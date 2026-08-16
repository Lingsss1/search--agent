# OpenStateSearch 长程奖励分配设计

状态：ABC 已实现，r11 单步门控前修正版  
记录日期：2026-08-16

## 1. 目标与边界

本项目继续使用 AReaL 作为多机 rollout、FSDP actor、权重同步和 PPO/GRPO
更新的执行后端。AReaL 本身不是 Search Agent 的奖励算法，不能自动解决长程
信用分配问题。

当前训练把轨迹最终奖励写到最后一次模型交互，再以
`turn_discount=1.0` 传播给所有交互。对最长 16 轮的
`SEARCH -> OPEN -> KEEP -> ANSWER` 轨迹，这会产生两个问题：

1. 任一终局协议错误可能使整条轨迹得到 `-1`，掩盖之前的有效搜索步骤；
2. 同一问题的四条 rollout 如果终局奖励相同，GRPO 组内优势为零。

本设计的目标是在不取消最终答案质量约束的前提下，把答案和支持证据反向分配到
各个工具步骤。过程奖励不得鼓励无意义的工具调用，也不得取代最终 ANSWER 奖励。

## 2. 参考方法与采用范围

### 2.1 Search Agent 基线

- [Search-R1](https://arxiv.org/abs/2503.09516) 和
  [ReSearch](https://arxiv.org/abs/2503.19470)：参考多轮搜索 rollout、检索结果
  masking 和端到端 RL 组织方式。它们主要使用 outcome reward，不直接作为本项目
  的长程信用分配方案。
- [WebAgent-R1](https://arxiv.org/abs/2505.16421)：作为强初始化加终局二值奖励可以
  有效的对照。当前项目出现过终局有效率坍缩，不满足直接依赖稀疏二值奖励的安全
  前提。

### 2.2 主方案：Answer-Backtracked Credit Assignment

[ABSeeker](https://arxiv.org/abs/2608.05102) 的 ABC 方法从正确答案回溯必要线索，
再按每一步是否发现、验证和保留线索给予 step-level supervision。本项目训练数据
已有正确答案和 gold supporting evidence，因此第一版使用确定性的 evidence/clue
匹配，不引入 LLM judge。

### 2.3 可选增强：TRACE

[TRACE](https://arxiv.org/abs/2607.13988) 在工具调用边界计算正确答案在冻结参考模型
下的 log-probability 变化，并以相邻状态的 TD 差作为 turn-level reward。该方法可
覆盖“有助于回答但未被标为 gold evidence”的检索结果，但每个状态都需要额外打分，
因此只在确定性 ABC 通过门控后加入。

### 2.4 暂不进入首版

- [SALT](https://arxiv.org/abs/2510.20022)：同一问题的多条轨迹构图，适合作为后续
  group-RL 消融，但搜索状态的公共前缀可能较少。
- [HCAPO](https://arxiv.org/abs/2603.08754)：使用 LLM 进行事后归因，成本较高且引入
  judge 偏差，暂不作为首版依赖。
- [Agent Lightning](https://arxiv.org/abs/2508.03680)：其 transition/credit-assignment
  抽象值得参考，但它同样偏训练框架，不构成切换 AReaL 的理由。

## 3. 首版奖励定义

### 3.1 线索阶段势能

对问题的每条唯一 gold supporting evidence `g`，定义其在状态 `s_t` 中达到的最高
阶段：

| 阶段 | 建议初始分值 |
|---|---:|
| 未出现 | 0.0 |
| 被 SEARCH 首次召回 | 0.1 |
| 被 OPEN 首次展示 | 0.3 |
| 被合法 KEEP 首次保留 | 1.0 |

状态势能为：

```text
Phi(s_t) = mean_g(stage_score(g, s_t))
```

这与按全部 gold evidence 的累计覆盖率书写完全等价：

```text
Phi(s_t) = 0.1 * C_search + 0.2 * C_open + 0.7 * C_keep
```

其中后一阶段的 evidence 也计入前一阶段覆盖。因此每发现、展示或保留一条新的 gold
evidence 都会增加势能，并非只奖励第一条命中。

过程奖励使用势能差：

```text
r_evidence,t = alpha * (Phi(s_{t+1}) - Phi(s_t))
```

约束：

- 每条 evidence 的每个阶段只计一次；
- 重复 SEARCH、OPEN 或 KEEP 不产生正奖励；
- 不因 JSON 格式正确或动作合法本身给正奖励；
- recoverable 非法动作给予固定小负奖励；
- 非法引用或未完成 ANSWER 保留显著终局负奖励；
- 正向证据奖励累计最多 `+0.25`；非法动作惩罚独立累计最多 `-0.10`，防止早期错误
  吃掉后续纠错产生的正向信用；
- 重复但合法的动作过程奖励为零。

阶段分值和 `alpha` 是需要通过消融确定的超参数，不应在没有门控结果时视为最终值。

### 3.2 终局奖励

合法完成时继续使用现有质量目标：

```text
R_terminal =
    2.0 * answer_f1
  + 0.8 * support_recall
  + 0.4 * citation_precision
  - 0.15 * duplicate_rate
  - bounded_protocol_penalty
```

Phase B 的搜索、OPEN 和 token 成本只在答案质量达到门槛后启用。首版长程信用分配
仍在 Phase A 验证，不同时改变成本课程。

### 3.3 GRPO 与逐步优势组合

同一问题继续采样 `G=4` 条轨迹。首先只对终局奖励计算 GRPO 组相对优势：

```text
A_episode,j = (R_terminal,j - mean_group) / (std_group + eps)
```

每个模型交互使用混合优势：

```text
A_j,t = A_episode,j + beta * clip(r_evidence,j,t, -c, c)
```

其中 `beta` 首版应较小，使终局答案质量保持主导。当四条轨迹终局奖励完全相同时，
`A_episode=0`，但新增有效证据的步骤仍可得到非零训练信号。失败轨迹中的有用步骤
不再自动与最终错误动作获得完全相同的优势。

终局奖励已经做 prompt 内组标准化，过程分也已限幅，因此 actor 不再做第二次 batch
标准差缩放。尤其不能在全失败低方差 batch 中除以很小的标准差，否则单个 KEEP 的
局部优势会被放大数倍。

每个 individual interaction 对应一个模型动作。actor loss 必须先在该动作的生成 token
内求平均，再对动作求平均；prompt、observation、检索文档和 tool response 全部由
`loss_mask` 排除。这样 SEARCH query 或 JSON 更长不会获得更大的总梯度权重。

首版直接使用每个 interaction 的 `A_j,t`，不再对 episode reward 调用
`turn_discount=1.0` 后复制到全部交互。若需要 return-to-go，应单独验证
`gamma=0.9--0.95`，不能与局部优势未经分析地重复传播。

## 4. TRACE 增强

确定性 ABC 通过门控后，可使用冻结的 SFT/reference policy 计算：

```text
V_t = mean_token_logp_ref(
    gold_answer | question, legal_retained_evidence(s_t)
)
r_trace,t = clip(V_{t+1} - V_t, -c_trace, c_trace)
```

组合优势变为：

```text
A_j,t = A_episode,j
      + beta_evidence * r_evidence,j,t
      + beta_trace * r_trace,j,t
```

`r_trace` 必须批量或离线计算，并记录额外 forward 次数和训练吞吐。参考模型冻结，
训练策略不能同时充当自己的奖励评估器。

## 5. AReaL 接入要求

现有实现存在一个不能忽略的兼容性问题：

- `openstatesearch/training/areal_agent.py` 当前调用 `set_last_reward()` 后执行
  `apply_reward_discount(turn_discount=1.0)`；
- `areal/infra/remote_inf_engine.py::_normalize_group_rewards()` 只读取每条 rollout
  最后一个 interaction 的 reward，然后把同一个归一化 reward 覆盖到该 rollout
  所有 interaction。

因此，仅在 workflow 中增加 `client.set_reward(response.id, step_reward)` 不足以实现
逐步信用分配，过程奖励会在组归一化时被覆盖。

需要把 rollout 数据语义拆分为：

```text
episode_reward       # 终局质量，用于同 prompt 的四轨迹归一化
process_reward       # 每个 interaction 独立保存
training_advantage   # normalized episode reward + beta * process reward
```

当前实现使用项目侧 workflow 加一个最小 AReaL grouped-rollout adapter：

1. agent 执行时记录每次 response ID 对应的状态转移和 `process_reward`；
2. 轨迹结束后单独保存 `episode_reward`；
3. 四条 rollout 到齐后只归一化 `episode_reward`；
4. 对每个 interaction 合成 `training_advantage`；
5. 导出 individual interactions，actor 将该标量放到本 interaction 的响应末 token；
6. 审计日志同时保留原始终局分、组相对分、过程分和最终训练优势。

最小 vendor patch 只增加 episode/process 元数据的合成和 `sequence_mean` actor loss，旧
workflow 与默认 `token_mean` 行为保持不变，并有独立单元测试。不能通过关闭组归一化、
改成普通 REINFORCE 来假称仍是 GRPO。

## 6. 实现顺序

1. **已完成：离线 scorer**：对历史轨迹重放 ABC 分数，确认 gold evidence 映射和状态
   阶段正确。
2. **已完成：奖励单元测试**：覆盖首次/重复 SEARCH、OPEN、KEEP、失败轨迹中的有效
   前缀，以及正负过程预算互不抢占。
3. **已完成：group adapter 测试**：四条终局同分时终局组优势为零而逐步优势非零；
   终局不同时局部奖励不反转排序。
4. **进行中：一训练步 smoke**：检查逐 interaction reward、动作等权 loss、权重更新和
   checkpoint 正常保存。
5. **10--20 步稳定性门控**：只比较 terminal-only 与 ABC，不启用 TRACE。
6. **通过后跑 ABC 与 ABC+TRACE 消融**，再决定完整 Phase A 配置。

## 7. 必须记录的评测指标

- 完整 ANSWER 率和联合合法率；
- answer F1、support recall、citation precision；
- 四条 rollout 终局奖励完全相同的 group 比例；
- 非零 process reward 的 interaction/trajectory 比例；
- 按 SEARCH、OPEN、KEEP、ANSWER 分解的平均优势；
- 失败轨迹中获得正过程分的有效前缀比例；
- 重复搜索、重复证据和平均工具轮数；
- KL、entropy、gradient norm、长度分布及每步 wall time；
- TRACE 启用后的额外 scoring 成本。

停止条件：联合合法率继续明显低于冻结 SFT 基线、ANSWER 完成率连续下降、终局全同
group 比例没有改善、过程奖励主要来自重复/无效操作，或局部奖励使最终质量倒退。

## 8. 不采用的简化方案

- 不给每个合法动作固定正奖励：会鼓励无意义的长轨迹；
- 不只把 `turn_discount` 从 1.0 改成 0.9：折扣只能改变远近权重，不能判断哪一步有用；
- 不取消非法引用门槛：引用合法性仍是最终任务定义；
- 不在 `0/50` 有效的状态下直接延长 terminal-only GRPO；
- 不因为 Agent Lightning 提供 credit-assignment 接口就立即更换训练后端。

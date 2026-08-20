# OpenStateSearch-36

OpenStateSearch-36 是一个可复现的多轮搜索 Agent 工程实现。它把原始对话历史压缩为机械维护的
`SearchState`，使用 BM25 + LRAT Dense 的混合检索，并通过可程序验证的答案、证据、引用和成本奖励训练
Policy。实现依据为项目实施规范 v1.0。

## 当前进度（2026-08-20）

正式 Phase A 多机 GRPO 已启动，实验为
`oss36-grpo-a-v8-logprob-fix / processedlogp-from-r21-5steps-gate-r1`。截至
2026-08-20 11:55（+08:00），已完成 `121/375` 个更新，step 122 正在 actor
forward/backward；step 120 的非阻塞周期评测已完成，step 200 的 adapter 归档、reward/ABC 审计和 held-out
联合门控 watcher 均在运行。

当前 50 条 held-out 结果尚未证明持续提升：step 100/110/120 的 answer F1 分别为
`0.4189 / 0.4407 / 0.4329`，step 120 相比 step 110 的 paired bootstrap 95% CI 为
`[-0.0598, 0.0430]`。优化器、importance ratio 和 ABC 信用分配审计正常，但训练 reward 不作为质量提升证据；
正式决策点仍是预先约定的 step-200 checkpoint gate。

完整进度、结果表、关键修复、验证证据与剩余交付项见
[`PROGRESS_REPORT_2026-08-20.md`](PROGRESS_REPORT_2026-08-20.md)。AReaL 基于固定 upstream commit 的
可复现修改见 [`patches/areal`](patches/areal/README.md)。当前项目测试为 `98 passed, 0 failed`。

> 状态说明：完整目标尚未完成。Phase A 375 步、Phase B、A--F 矩阵、四个正式数据集、最终 reward audit、
> failure cases、cost curve、精确回放及 acceptance audit 仍是必需交付项。

## 历史稳定性实验与设计演进（2026-08-17）

### 目标

当前目标是先验证 Qwen3.6-27B 的多轮 GRPO 是否能稳定学习
`SEARCH -> OPEN -> KEEP -> ANSWER`，同时保持动作、状态和引用协议合法；通过短程稳定性门控后，再扩大
rollout 并完成 Phase A、Phase B 及完整评测矩阵。最终目标是得到答案正确、证据完整、引用准确且搜索成本可控
的 Search Agent，而不是只降低训练 loss。

### 正式训练结构

- Actor：8×A800-80GB，FSDP data parallel 8，避免跨节点参数分片通信。
- Rollout：H800 上运行 vLLM TP4，生成同一策略版本的 on-policy 轨迹。
- Retriever：另一台 A800 机器提供冻结的 R4 HTTP 检索服务；训练不在 rollout worker 内重复加载 Dense
  模型和索引。
- Policy：Qwen3.6-27B（当前 Transformers 注册为 `Qwen3_5ForConditionalGeneration`），LoRA rank 16，
  仅训练 `q_proj/k_proj/v_proj/o_proj`。
- 已验证的稳定性配置每步为 16 prompts × 4 samples，共 64 episodes；每条最多 16 轮工具交互。
  先前 8-prompt 配置在 actor DP8 上每 rank 只有一个 prompt group，无法稳定平衡展开后的 action 数。
- 优化：GRPO/PPO，学习率 `1e-6`，`sequence_mean` 动作内 token 归一化，第二次 reward/advantage
  标准化关闭。
- 稳定性段保持严格同版本轨迹，不跨策略版本预取；确认正确后再扩展 H800 为 DP2×TP4 rollout。

### ABC 信用分配

终局奖励为：

```text
2.0 * answer_f1
+ 0.8 * support_recall
+ 0.4 * citation_precision
- 0.15 * duplicate_rate
- protocol_penalty
```

过程奖励按 gold evidence 覆盖势能增量分配：SEARCH/OPEN/KEEP 的累计阶段权重为
`0.1/0.3/1.0`，乘以 `alpha=0.25`；正过程奖励上限 `+0.25`，非法动作负奖励下限
`-0.10`。训练优势为 prompt 内终局组优势加局部过程奖励。这样，即使同组终局全部失败，只要失败轨迹包含
正确证据前缀，仍有可学习信号。历史回放中这类失败轨迹占 `69.91%`。完整设计见
[CREDIT_ASSIGNMENT_DESIGN.md](CREDIT_ASSIGNMENT_DESIGN.md)。

### 已完成结果

- AReaL/ABC 相关测试曾达到 60/60 通过；终局奖励符号、低方差 batch 放大和动作长度偏置已修正。
- r21 单步训练完成并成功合并；held-out 随机评测相对 SFT 基线有改善，但绝对门槛尚未全部通过。
- r24 原始 step0：31/32 合法，平均 reward `1.5298`，F1 `0.5748`，support recall
  `0.3464`，citation precision `0.3906`。
- r24 原始 step1：32/32 合法，平均 reward `1.6202`，F1 `0.4965`，support recall
  `0.5260`，citation precision `0.5625`；模型与优化器 recover checkpoint 已保存。
- 8192-token cap 的诊断 step2：32/32 合法，reward `2.1632`，F1 `0.6878`，support recall
  `0.6198`，citation precision `0.7448`。该结果未进入最终优化器 recover 谱系，只保留作诊断证据。

### 当前运行状态

r24 已完成到逻辑 GRPO update 4；随后 r25 使用 batch 16、actor micro-batch cap 12,288、eager rollout 和
逐步 recover，再完成两个优化器更新并干净退出。最终合并模型为
`artifacts/policy_grpo_r25_step6_merged_seed36`，即逻辑 update 6。r25 两个 rollout version 均为 61/64
联合有效，且 checkpoint、optimizer recover 与 teardown 都成功。

同一批 50 条、temperature=1.0 的 r21/r25 配对评测中，r25 completion 从 84% 提升到 90%，但严格
token-overlap answer F1 从 0.4324 变为 0.4004。逐样本审计显示均值差为 -0.0320，paired bootstrap 95%
区间为 `[-0.1451, 0.0788]`，不足以证明真实退化；两边都协议干净的 22 条上均值差为 +0.0243。负差主要受
2 条 r25 未完成轨迹和 7 条“包含完整 reference 但附带解释”的长度惩罚影响，r25 平均答案长度从 5.22
增至 8.42 tokens。审计 artifact 为
`artifacts/eval/grpo_r21_vs_r25_gate50_paired_audit.json`。

扩大到同一批 100 条后，r25 相对 r21 的 answer F1 为 `0.4372 vs 0.4670`，paired bootstrap 95%
区间仍跨零（`[-0.1155, 0.0554]`），但 completion、support recall、citation precision 和非法 doc 引用
也都向不利方向变化，且三个数据集的 F1 delta 均为负。因此当前 ABC checkpoint 没有通过“不退化”稳定性
门槛，不能直接启动完整 Phase A。

中间 r24/update4 的同批 100 条 F1 为 `0.4149`：相对 r21/update1 为 `-0.0521`，而 r25/update6 相对
r24 回升 `+0.0223`。协议完成率在 r24 达到 0.94，说明主要退化是答案/证据质量而非无法完成 ANSWER，且短程
变化非单调。ABC 回放显示 95% 以上 episode 有过程信号，过程回报与终局回报相关系数为 0.65--0.68；因此
问题不是奖励过于稀疏，而更可能是小 batch 更新方差和 rollout/training policy mismatch。配对 artifacts 为
`artifacts/eval/grpo_r21_vs_r24_gate100_temp10_paired_audit.json` 与
`artifacts/eval/grpo_r24_vs_r25_gate100_temp10_paired_audit.json`。

为隔离 importance-ratio 下尾，已从 r21 分别完成一次 `[0.5,5]` 双边门控和仅 upper=5 的更新。约
7.1%--7.3% token ratio 低于 0.5，因此下尾不是装饰性配置；但在相同 50 条 held-out、vLLM TP4、
temperature=1.0 和逐 prompt/turn 稳定 seed 下，两者都未通过多指标不退化门槛。r21/lower/upper 的 F1 为
`0.3990/0.4023/0.3916`；lower 相对 r21 的 F1 delta 仅 `+0.00325`，95% CI
`[-0.0543,0.0637]`，同时 support recall 从 `0.3767` 降至 `0.3550`、citation macro 从 `0.5124`
降至 `0.4147`、completion 从 `0.94` 降至 `0.92`。upper-only 同样降低 F1、support 和 completion。
因此当前保留 r21，不再为这组门控做一小时严格同轨迹重跑，也不启动 375 步正式训练。机器可读汇总见
`artifacts/eval/grpo_v7_icepop_one_step_ablation.json`。

### 已知问题与下一步

- 当前热缓存实测中，27B 权重读取约 40--45 秒；更慢的是 FSDP2 应用/广播约 105--109 秒，以及 eager
  vLLM rollout worker 初始化约 105 秒。框架已加入 actor 前向/反向 micro-batch 心跳和 ETA，便于区分
  “仍在计算”与真正卡死。
- 原评测脚本的 8 个分片会各自加载约 51 GB 权重并重复编译 kernel。现已支持持久化 vLLM TP4 服务；同一
  greedy 样本与直接 Transformers 路径的 7 个动作、token 数、答案和引用逐字段一致，正式评测可复用服务。
  H800 实测权重读取仅 4.86--6.36 秒，约两分钟冷启动主要来自 engine/compile/KV-cache/CUDA graph 与
  多模态 warmup。启动器已分离存储路径和 API served-model-name，并关闭该环境中必然编译失败后回退的
  FlashInfer allreduce-RMS fusion；三份 matched gate50 均完成且干净释放显存。
- 同策略版本的 vLLM behavior 与 FSDP/SDPA 重算 log-prob 仍有长尾差异。从 r21 完成的两个单步实验显示，
  约 7.1%--7.3% token 的 importance ratio 低于 0.5，而超过 5 的不足 0.04%；双边门控确实改变更新，
  但 held-out 多指标没有改善。框架现已增加逐请求稳定 seed，后续相同版本、prompt、sample index 和 turn
  可独立于异步请求顺序复现；下一步应先缩小 behavior/recompute 差异，而不是继续调 mask 阈值。
- 单步训练的主要瓶颈不是 rollout：64 episodes 展开为 595 条 action-level 序列，重复多轮前缀，actor 每步
  处理约 134 万逻辑 token，其中约 97% 位置被 mask；一次 PPO update 实测 1,250--1,265 秒。直接跑 375
  步会持续数天。episode packing/前缀复用可能显著提速，但会触及每动作 ABC advantage 的训练语义，必须
  先做等价性测试和小规模 A/B，不能作为纯工程重构直接启用。
- 8192-token cap 曾触发跨 rank 的不可行 micro-batch 数：某 rank 只有 16 个 sequence groups，却被同步要求
  执行 20 个非空 micro-batches。
- 异常后 actor destroy RPC 曾卡住超过 20 分钟；collective RPC 已改为不重试并有界清理，本轮两次 DP8
  单步均完成 checkpoint 与 teardown，所有 A800/H800 显存已释放。
- 上述问题、临时规避和最小框架修复记录在
  [AREAL_RUNTIME_ISSUES.md](AREAL_RUNTIME_ISSUES.md)。
- gate50 统计功效不足，且暴露了结构化 `answer` 字段冗长导致的 F1 长度惩罚。扩大稳定性评测样本并保持
  checkpoint、prompt、采样参数和检索器 provenance 可配对；只有合法率、完成率、F1、support recall 和
  citation precision 的更强证据不退化时，才扩展 10--20 步。
- 稳定性门控通过后再启动完整 Phase A；最终仍需完成 A--F、R0--R4、BrowseComp-Plus、中文零样本评测和
  奖励审计。

## 已实现的主链路

- 严格动作协议：`SEARCH / OPEN / KEEP / VERIFY / ANSWER`
- 外置状态、搜索/打开/Token 预算、重复查询短路和引用合法性校验
- 可复现语料构建、稳定 `doc_id`、分句、切块、manifest 与污染检测
- 无第三方依赖的 BM25、Dense 接口、RRF 混合检索与固定种子 Top-K
- Answer F1、Supporting Recall、Citation Precision、Duplicate Rate、成本和 RL-A/RL-B 奖励
- 轨迹回放、端到端指标、JSONL 审计日志和离线 Demo
- LRAT Retriever、Policy SFT、GRPO A/B 的冻结配置及可校验训练入口

真实的 27B 训练需要网页规范中的 8×A800-80GB 与外部数据。本仓库不提交数据或 checkpoint；下载后通过
manifest 固定 revision 和 SHA256。

## 快速开始

```bash
cd /code/openstatesearch
python -m openstatesearch.demo
python scripts/replay_demo.py
python -m unittest discover -s tests -v
```

Demo 使用 6 篇内置小语料运行完整的 `SEARCH → OPEN → KEEP → ANSWER` 流程，不联网、不下载模型。

## 数据与训练

```bash
# 0. 按 manifest 的不可变 commit 下载所需来源；可用 --only 仅取当前阶段
python scripts/fetch_sources.py --only LRAT-Train

# 1. 从统一 JSONL 构建冻结语料；每行至少含 title/text/source
python scripts/build_corpus.py --input data/raw.jsonl --output data/corpus.jsonl \
  --manifest data/manifests/corpus_manifest.json

# 2. 训练前校验所有配置与资源（不启动 GPU 任务）
python scripts/train_retriever.py --config configs/retriever_lrat.yaml --validate-only
python scripts/train_sft.py --config configs/policy_sft.yaml --validate-only
python scripts/train_grpo.py --config configs/grpo_a.yaml --validate-only
python scripts/train_grpo.py --config configs/grpo_b.yaml --validate-only

# 3. 运行冻结轨迹评测
python -m openstatesearch.eval.runner --predictions predictions.jsonl --output metrics.json
```

实际 SFT 命令增加 `--data <converted.jsonl> --output <adapter_dir>` 并通过 `torchrun` 启动 8 卡 FSDP；
Adapter 只有通过 `scripts/check_sft_gate.py` 的四项门槛才能启动 RL。实际 LRAT 启动命令为
`python scripts/train_retriever.py --config ... --pairs ... --output ...`；实际 GRPO
先用 `scripts/build_dense_index.py` 从自训 LRAT checkpoint 生成冻结 Dense 索引，再用
`scripts/merge_adapter.py` 把上一阶段 Adapter 合并为 AReaL 初始化 checkpoint。实际 GRPO 启动命令传入
冻结语料、模型和索引：`--corpus <jsonl> --dense-model <dir> --dense-index <npz>`；默认使用仓库自带的
`configs/areal_grpo_lora.yaml`，也可用 `--areal-config` 替换。
GRPO 采用 AReaL 官方推荐的 OpenAI-compatible proxy workflow，类路径为
`openstatesearch.training.areal_agent.OpenStateSearchAgent`。

`scripts/eval_retrievers.py` 会串行运行 R0–R4 并输出 Recall@5/20/100 与 nDCG@10；Dense 索引旁会自动
写入包含 SHA256、doc-id 摘要、维度、pooling 和 query instruction 的 manifest。

统一语料输入不能使用“每道题自带的候选段落”。HotpotQA、2WikiMultiHopQA 和 MuSiQue 的 train/dev
context 必须先取并集、去重、切块，再统一建 BM25/Dense 索引。测试题及 BrowseComp-Plus 不得进入训练。

## 仓库结构

```text
openstatesearch/
├── configs/                    # Retriever、SFT、RL-A、RL-B、评测冻结配置
├── data/manifests/             # 数据 revision、许可和 SHA256（不提交原始数据）
├── openstatesearch/
│   ├── agent/                  # schemas、SearchState、Harness
│   ├── data/                   # 规范化、污染检测、语料构建
│   ├── retriever/              # BM25、Dense 接口、RRF、LRAT loss
│   ├── rewards/                # 答案、证据、成本和聚合奖励
│   ├── training/               # 配置校验和训练计划
│   └── eval/                   # 指标、轨迹回放和 CLI
├── scripts/                    # 构建、训练与评测入口
└── tests/                      # schema、状态、奖励、检索和端到端测试
```

## 关键不变量

1. Gold answer/supporting facts 只进入 Reward，不进入 Policy observation。
2. `KEEP` 只能引用已经 `OPEN` 的 `doc_id` 和合法 `sent_id`。
3. 重复 query 返回 `duplicate=true`，不调用 Retriever，也不消耗搜索预算。
4. Retriever 权重、语料、索引和参数在 Policy SFT/RL 期间冻结。
5. RL-B 只对 `AnswerF1 >= 0.8` 的质量达标轨迹施加效率成本。

最终实验完成后，使用 `scripts/audit_acceptance.py --results results.json` 做缺失即失败的逐项验收；不能以
smoke 测试替代 A–F、R0–R4 或外部评测结果。

## 大规模运行边界

本地 smoke 证明程序链路与不变量；它不能证明网页定义的最终研究指标。完整验收仍需运行 A–F 六组实验、
R0–R4 Retriever 对照、BrowseComp-Plus 830 题、中文零样本评测和奖励审计，并达到
`configs/acceptance.yaml` 中的阈值。

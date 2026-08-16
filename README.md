# OpenStateSearch-36

OpenStateSearch-36 是一个可复现的多轮搜索 Agent 工程实现。它把原始对话历史压缩为机械维护的
`SearchState`，使用 BM25 + LRAT Dense 的混合检索，并通过可程序验证的答案、证据、引用和成本奖励训练
Policy。实现依据为项目实施规范 v1.0。

## 当前实验方案与进度（2026-08-16）

### 目标

当前目标是先验证 Qwen3.5-27B 的多轮 GRPO 是否能稳定学习
`SEARCH -> OPEN -> KEEP -> ANSWER`，同时保持动作、状态和引用协议合法；通过短程稳定性门控后，再扩大
rollout 并完成 Phase A、Phase B 及完整评测矩阵。最终目标是得到答案正确、证据完整、引用准确且搜索成本可控
的 Search Agent，而不是只降低训练 loss。

### 正式训练结构

- Actor：8×A800-80GB，FSDP data parallel 8，避免跨节点参数分片通信。
- Rollout：H800 上运行 vLLM TP4，生成同一策略版本的 on-policy 轨迹。
- Retriever：另一台 A800 机器提供冻结的 R4 HTTP 检索服务；训练不在 rollout worker 内重复加载 Dense
  模型和索引。
- Policy：Qwen3.5-27B，LoRA rank 16，仅训练 `q_proj/k_proj/v_proj/o_proj`。
- 每步：8 prompts × 4 samples，共 32 episodes；每条最多 16 轮工具交互。
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

r24 正从 `global_step=2` 恢复并重做 step2，目标完成到 step4。actor micro-batch 上限已从 8192 调整为
10240 tokens，recover 改为每一步保存。截止 2026-08-16 16:11 UTC，新 v2 rollout 已完成，recompute
log-probability 和 advantage 计算通过了此前的 micro-batch 同步故障点，8×A800 正在执行 PPO 更新。该状态是
运行快照，不代表 step2 已最终提交或 r24 已完成。

### 已知问题与下一步

- AReaL 冷启动约 490 秒：actor FSDP 构造约 266 秒，vLLM compile/CUDA graph 约 224 秒。
- 8192-token cap 曾触发跨 rank 的不可行 micro-batch 数：某 rank 只有 16 个 sequence groups，却被同步要求
  执行 20 个非空 micro-batches。
- 异常后 actor destroy RPC 曾卡住超过 20 分钟。
- 上述问题、临时规避和最小框架修复记录在
  [AREAL_RUNTIME_ISSUES.md](AREAL_RUNTIME_ISSUES.md)。
- r24 step4 完成后：合并最终 LoRA，运行 held-out temperature=1 gate50，并与 SFT、r21 做配对比较；只有
  合法率、完成率、F1、support recall 和 citation precision 未退化时，才扩展 10--20 步。
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

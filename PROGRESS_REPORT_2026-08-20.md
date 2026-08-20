# OpenStateSearch-36 progress report

Snapshot: **2026-08-20 11:55 +08:00**
Experiment: `oss36-grpo-a-v8-logprob-fix / processedlogp-from-r21-5steps-gate-r1`

## Executive summary

OpenStateSearch-36 now has a working end-to-end implementation for corpus
preparation, hybrid retrieval, protocol-constrained multi-turn search, SFT,
multi-node GRPO, checkpoint/recovery, periodic evaluation, reward/credit audit,
formal evaluation aggregation, and provenance verification.

The formal Phase A run is active and frozen at its audited configuration. At
this snapshot, **121 of 375 updates are complete (32.3%)** and step 122 is in the
actor update. The latest independent evaluation is step 120. Optimization and
ABC credit assignment are operating correctly, but the 50-example held-out
series has not shown a sustained or statistically significant answer-F1 gain.
The next predeclared decision point is the step-200 checkpoint gate.

The project is therefore **in progress, not complete**. Phase A, Phase B, the
full A--F matrix, named external datasets, final reward audit, failure cases,
cost curve, and final replay/acceptance audit remain required.

## Current training topology and method

- Policy: Qwen3.6-27B, registered by the installed Transformers version through
  the Qwen3.5 implementation.
- Actor: 8 x A800-80GB, FSDP data parallel.
- Rollout: H800, vLLM TP4, strict same-version trajectories.
- Retriever: frozen R4 hybrid HTTP service on a separate A800 host.
- Batch: 16 prompts x 4 samples = 64 episodes per update, up to 16 turns each.
- Adapter: LoRA rank 16 / alpha 16 on `q_proj`, `k_proj`, `v_proj`, `o_proj`.
- Optimizer: learning rate `1e-6`, PPO clip `0.4`, no KL penalty, upper
  importance-ratio rejection at `5.0`.
- Credit: prompt-group terminal GRPO advantage plus local ABC evidence-coverage
  increments, with action-internal token averaging.
- Recovery: optimizer-bearing recover checkpoint approximately hourly; an
  explicit watcher archives and audits step 200 without stopping Phase A.
- Evaluation: 50 held-out examples every 10 steps, on separate H800 GPUs, without
  pausing or mutating training.

## Phase A status

At the snapshot:

| Item | Status |
|---|---:|
| Required updates | 375 |
| Completed updates | 121 |
| Progress | 32.3% |
| Active work | step 122 actor forward/backward |
| Latest recover checkpoint | human step 121 |
| Periodic evaluation complete through | step 120 |
| Step-200 checkpoint watcher | active; continue after archive |
| Periodic-evaluation watcher | active |

Recent throughput is about 28--30 minutes per update. If unchanged, step 200 is
roughly 38 hours from this snapshot and step 375 roughly five days away. These
are operational estimates, not completion claims.

One infrastructure warning is being monitored: newly launched shell processes
cannot see `/dev/nvidia*` and `nvidia-smi`/`cudaGetDeviceCount` fail. The actor's
pre-existing CUDA contexts continued to advance step 122, and no training
Traceback, OOM, NCCL, or CUDA runtime error had appeared at the snapshot. The
step-121 recover checkpoint remains the recovery boundary if those workers fail.

## Held-out periodic results

All rows below use the same 50-example held-out set and evaluation protocol.

| Step | Answer F1 | Completion | Joint valid | Citation valid | Citation precision | Support recall |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 0.4541 | 0.96 | 0.52 | 1.00 | 0.3469 | 0.3550 |
| 10 | 0.4306 | 0.94 | 0.54 | 1.00 | 0.3750 | 0.3550 |
| 20 | 0.4190 | 0.92 | 0.52 | 1.00 | 0.3956 | 0.3550 |
| 30 | 0.4515 | 0.92 | 0.56 | 1.00 | 0.3830 | 0.3450 |
| 40 | 0.4473 | 0.96 | 0.56 | 1.00 | 0.3478 | 0.3383 |
| 50 | 0.4430 | 0.94 | 0.60 | 1.00 | 0.3558 | 0.3700 |
| 60 | 0.4531 | 0.94 | 0.54 | 1.00 | 0.3936 | 0.3700 |
| 70 | 0.4295 | 0.92 | 0.54 | 1.00 | 0.3721 | 0.3350 |
| 80 | 0.3882 | 0.90 | 0.56 | 1.00 | 0.3708 | 0.3350 |
| 90 | 0.4350 | 0.94 | 0.60 | 1.00 | 0.3500 | 0.3600 |
| 100 | 0.4189 | 0.94 | 0.50 | 1.00 | 0.3478 | 0.3350 |
| 110 | 0.4407 | 0.92 | 0.52 | 1.00 | 0.3696 | 0.3550 |
| 120 | 0.4329 | 0.92 | 0.56 | 1.00 | 0.3864 | 0.3467 |

Step 120 versus step 110 has an answer-F1 delta of `-0.00785`, with paired
bootstrap 95% CI `[-0.05979, 0.04302]`. The interval crosses zero. The highest
observed point remains step 6 at `0.4541`; no sustained or statistically
significant improvement is established through step 120.

This is why training reward is not used as the quality gate. The frozen run is
being preserved until the predeclared step-200 checkpoint evaluation can provide
stronger evidence.

## Step-100 joint gate

Step 100 was evaluated in two complementary ways.

### Independent held-out evaluation

- 50/50 rows produced.
- Answer F1 `0.418945`, EM `0.24`.
- Completion `47/50 = 0.94`.
- Joint-valid `25/50 = 0.50`.
- Citation validity `1.00`, citation precision micro `0.347826`.
- Support recall `0.335`.
- Versus step 90, F1 delta `-0.01604`, 95% CI
  `[-0.08512, 0.05097]`; not statistically significant.

### Reward and ABC mechanism audit

- 16 prompt groups / 64 episodes.
- All 64 episodes received non-zero process credit.
- Only 1/16 groups had identical terminal rewards.
- Six episodes had negative terminal reward; all six retained a positive evidence
  prefix, proving that failed episodes still carried localized learning signal.
- 50 raw reward records were linked to full rollouts with zero unmatched records
  and maximum reward delta below `1e-7`.
- Nine mappings remain interchangeable inside identical
  prompt/reward/action-signature equivalence classes because the runtime raw
  recorder did not store a sample-unique trajectory key. Aggregate reward
  components are exact; those nine row-level response assignments are not unique.
- The recorder keeps the first 50 asynchronously completed episodes, so this
  mechanism audit is not an unbiased quality estimate. Held-out evaluation is
  authoritative for model quality.

The gate decision was: **optimization and ABC are healthy, held-out quality is
not yet improved; continue the fixed run to step 200 without changing training.**

## Key design and correctness improvements

### 1. Reward and credit assignment

- Corrected the evidence terms to reward support recall and citation precision
  rather than subtracting them.
- Replaced first-hit process bonuses with coverage-based SEARCH/OPEN/KEEP
  potential increments over all gold evidence.
- Separated positive evidence-credit and negative protocol-penalty budgets so an
  early error cannot consume later correction credit.
- Prevented low-variance/all-failure batches from amplifying small process rewards
  through a second standard-deviation normalization.
- Averaged loss within each action's generated tokens and masked observations/tool
  responses, avoiding a gradient advantage for longer JSON actions.
- Made citation legality require cited evidence to be retained legal evidence,
  closing the opened-but-not-kept reward loophole.

### 2. PPO/log-probability correctness

- Aligned behavior log-probabilities with actor temperature/sampling semantics;
  raw pre-temperature vLLM log-probabilities are no longer treated as PPO behavior
  probabilities.
- Added importance-ratio tail, KL, effective-sample-size, rejection, timing, and
  micro-batch diagnostics.
- Added isolated two-sided-ratio experiments. About 7.1--7.3% of token ratios
  were below 0.5, proving the lower tail is material, but held-out evidence did
  not justify changing the frozen formal run.

### 3. Multi-node AReaL reliability

- Added exact synchronized micro-batch counts across FSDP ranks; the common count
  is now exact rather than a lower bound that FFD may exceed.
- Added fail-fast behavior when a whole workflow batch fails, preventing infinite
  replacement sampling.
- Bounded collective cleanup and disabled unsafe rank-local RPC retries after
  FSDP/NCCL collective failures.
- Made partial initialization and LoRA rollout cleanup recoverable.
- Restored counters from recover state while reapplying the current checkpoint
  cadence, preventing old cadence settings from silently overriding a resumed
  run.
- Added rollout pause/load/resume handshakes for disk LoRA weight updates.
- Added source-hash parity checks across nodes and explicit topology verification.

The AReaL changes are distributed as an auditable patch set under
[`patches/areal`](patches/areal/README.md), pinned to upstream commit
`1f966b1a9dac370fbecdd38f4eea974ba05cc4b5`.

### 4. Retrieval and async workflow

- Moved blocking HTTP retrieval off the async workflow event loop.
- Added explicit retriever health/provenance requirements and remote R4 service
  support, avoiding one dense model/index copy per rollout worker.
- Implemented stable hybrid BM25 + LRAT Dense/RRF retrieval and frozen corpus,
  index, and provenance manifests.

### 5. Evaluation throughput and reproducibility

- Added a persistent vLLM evaluation route so parallel shards share one model
  instead of loading a 27B checkpoint independently.
- Added deterministic per-prompt/per-turn seeds, independent of shard scheduling.
- Added non-blocking every-10-step evaluation and step-200 archive watchers.
- Added paired comparisons, failure categories, bootstrap confidence intervals,
  token/cost curves, and exact prompt-identity checks.
- Corrected shard-local gate enforcement: shards emit evidence, while only the
  merged result applies global thresholds.

### 6. Final-evidence integrity

- The final 50-trajectory reward audit must match the final GRPO model provenance,
  final checkpoint step, archived adapter inventory, and checkpoint audit metrics.
- External dataset manifests must use the final GRPO model provenance.
- The replay demo must come from the exact formal `F/8192` predictions and is
  freshly re-executed against a frozen SEARCH/OPEN environment action by action.
- Unrelated checkpoints and toy replay manifests can no longer satisfy final
  acceptance.

## Validation status

The current OpenStateSearch worktree passes:

```text
python -m pytest -q
98 passed, 0 failed
```

The only warning in the snapshot run was the local CUDA device-enumeration issue
described above; the tests themselves are CPU-safe and all passed.

The 18 AReaL test files covering the exported runtime changes pass as follows:

```text
289 passed, 1 deselected, 0 assertion failures
```

The deselected upstream test initializes a one-process Gloo group by opening a
local TCP socket. Socket creation is prohibited by the repository-validation
sandbox; the remaining tests, including every project-added regression, passed.

The exported AReaL patch set was also applied to a clean clone at the pinned
commit:

```text
expected changed/untracked paths: 46
applied changed/untracked paths:  46
file-content SHA mismatches:       0
```

## Local audit evidence

Large runtime artifacts and checkpoints are intentionally excluded from Git.
The following local evidence files define this snapshot:

| Evidence | SHA256 |
|---|---|
| periodic trend through step 120 | `cc2b45c2064cdf2f585e33d95a78978989621a76139f2a83f99f905d31141178` |
| step-100/eval-120 interim gate | `a6bd1cefb929ce74846968eb89335d073a784db6e0efa777f4221cec63117b2f` |
| step-100 linked reward summary | `3f8918f386d8fbd546f7d127f093ad7aa641616ea187d94efe49324cb2578d54` |
| step-100 ABC replay | `e1bae546c69973de8cd408b9b7839613790a4613d41cbf6913871a2ab70a6c6a` |
| completion-requirements snapshot | `6d307de213cfcca23fba6a8251199d55b2e3c46ecdd6969148c0ad8d3ce074bd` |
| formal-evaluation readiness | `19f3a21a9803ce398cc55a7262f0ffa5955730bf1fa608b63ba0c65fe026b566` |

These hashes identify local evidence; they do not imply the ignored artifact
files are present in a fresh clone.

## Remaining work before completion

1. Complete Phase A to update 375 while preserving periodic evaluation.
2. At step 200, archive the adapter and execute the held-out/reward/ABC gate;
   continue Phase A after the archive unless the declared failure criteria fire.
3. Resolve and pin Phase B's final topology, sampling temperature, action-token
   envelope, vLLM source provenance, final-step reward audit, and node placement.
4. Run Phase B from the audited Phase-A step-200 adapter and archive/audit its
   final step-187 checkpoint.
5. Merge the final adapter with immutable provenance.
6. Run all 12 A--F cells at 4096/8192 tokens with common prompt identities.
7. Run and aggregate in-domain dev, BrowseComp-Plus, XBench-DeepSearch, and
   BrowseComp-ZH evaluations.
8. Produce the accuracy/token-cost curve, failure cases, final linked reward
   audit, exact formal replay, and final acceptance report.
9. Audit every webpage requirement against an explicit artifact and SHA before
   declaring completion.

No training-success claim should be made until those items are complete and the
final acceptance audit passes with no missing evidence.

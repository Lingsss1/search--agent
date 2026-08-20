# AReaL runtime issues and mitigations

This file records failures observed during the Qwen3.6-27B multi-node GRPO
stability run. It distinguishes correctness fixes from performance tradeoffs so
that a short smoke configuration is not accidentally promoted to the formal run.

## 1. Cold-start latency

Observed on 2026-08-16 while recovering r24:

- launch to actor-ready: `16:00:17 -> 16:04:43`, about 266 seconds;
- rollout worker creation to vLLM-ready: `16:04:43 -> 16:08:27`, about 224 seconds;
- total launch to usable rollout: about 490 seconds.

The actor phase includes eight independent 27B checkpoint reads, FSDP2
construction, parameter broadcast, and optimizer creation. The rollout phase
includes vLLM `torch.compile`, CUDA-graph capture, multimodal warmup, and LoRA
registration. The H800 also attempts optional AWEX and FlashInfer fused
all-reduce components before falling back; these failures are noisy but were not
fatal in the validated runs.

Mitigations:

- Avoid restarting a healthy formal run. Save optimizer recovery state every
  step while stabilizing, then use a longer cadence after the path is proven.
- Use `scripts/train_grpo.py --rollout-enforce-eager` only for short integration
  smoke runs. It skips vLLM compile and CUDA-graph capture but lowers steady-state
  rollout throughput.
- Keep compiled mode for a long formal run, where the roughly four-minute vLLM
  setup cost is amortized.
- Longer term, preserve a rollout service across actor-only recovery or make the
  vLLM AOT cache serializable. Both require lifecycle work beyond a config tweak.

The corrected 2026-08-17 one-update run separated the remaining costs more
precisely. With the checkpoint already hot in the page cache, each actor rank
read/constructed the model in about 40--45 seconds, but FSDP2 application and
parameter broadcast took as much as 105 seconds. The eager H800 TP4 rollout
engine took about 105 seconds from initialization call to ready. Disk I/O is no
longer the dominant warm-start cost; FSDP construction/broadcast and inference
engine lifecycle are. Reusing healthy workers/engines is therefore the next
framework optimization, rather than adding another checkpoint cache layer.

## 2. Infeasible synchronized micro-batch counts

With `actor.mb_spec.max_tokens_per_mb=8192`, one r24 batch produced natural
micro-batch counts up to 20 while another FSDP rank owned only 16 sequence
groups. AReaL recursively set every rank's `min_groups` to 20, then failed with:

```text
Number of values 16 is smaller than min_groups 20
```

The constraint is genuinely infeasible if every micro-batch must be non-empty:
the rank with 16 values cannot execute 20 non-empty forwards. The framework bug
was that this was discovered independently inside the allocator, producing
rank-specific retries and an opaque failure.

Implemented mitigation:

- `allocate_balanced_mbs_synced` now gathers both natural micro-batch counts and
  local sequence-group counts once.
- Every rank raises the same diagnostic before recursive allocation when no
  common non-empty count exists.
- A later batch exposed a second bug in the feasible path: FFD's `n_mbs` is a
  minimum, not an exact target. The ranks agreed on 20 forwards, but rerunning
  FFD with `min_groups=20` returned 21 on one rank. The synchronized path now
  splits already-valid natural bins until it reaches the exact common count;
  subsets cannot exceed their parent's token capacity. A regression fixture
  covers the analogous `26 -> target 27 -> old FFD 28` case.
- The exact-count fix was exercised by a real DP8 Qwen3.6-27B update on
  2026-08-17: all ranks received exactly 16 synchronized micro-batches,
  recomputed log probabilities, completed the PPO update, published the LoRA
  checkpoint, and exited cleanly. This is the first end-to-end confirmation of
  the fix beyond the allocator regression test.

Run-level workaround:

- 10,240 tokens was sufficient for r24 version 2, but version 3 still produced
  a `17 > 16` mismatch. It is therefore not a stable cap for a batch containing
  only eight prompt groups on actor DP8.
- The short stability run should use at least two prompt groups per DP rank
  (`train_dataset.batch_size=16` for DP8), plus the validated 12,288-token cap.
  The larger batch amortizes the variance in actions per sampled episode; the
  cap controls memory per forward rather than total optimizer-step tokens.

Root-cause detail and longer-term fix:

- r24 version 3 contained eight prompt groups / 32 episodes / 272 action
  sequences. Individual prompt groups expanded to 16--56 actions. With exactly
  one prompt group per DP rank, the controller has no remaining freedom to
  balance that post-expansion variance.
- A correct structural fix should balance the post-expansion action counts as
  well as token totals, split prompt groups without changing GRPO normalization,
  or support masked dummy micro-batches. It must preserve episode/group
  semantics and be validated with a distributed FSDP test before replacing the
  batch-size/cap workaround.
- The aborted version 3 rollout was not consumed by the optimizer and was moved
  to `rollout_aborted_cap10240/3`. Its independent audit found 32/32 valid
  episodes, mean terminal reward 2.05343, and mean 8.5 interactions. The crash
  is therefore a framework batching failure, not policy/reward collapse.

## 3. Cleanup can hang after a worker error

After the 8192-token allocation failure, rollout cleanup completed but actor
cleanup remained at `Destroying engines on all workers...` for more than 20
minutes. The controller used the scheduler's default 7,200-second HTTP timeout
and three retries for a best-effort destroy RPC.

Implemented mitigation:

- engine destroy calls now use a 120-second HTTP timeout and one attempt;
- the aggregate destroy phase has a 125-second asyncio timeout;
- worker deletion still runs afterward in reverse rank order, leaving rank 0
  (the TCPStore owner) until last.
- aggregate TrainController calls now use one RPC attempt. Retrying only the
  HTTP request that surfaced an error is unsafe for FSDP/NCCL methods: the other
  ranks have already entered the collective and cannot participate in a
  rank-local replay. The trainer now fails once and performs coordinated
  teardown/recovery instead of waiting through three futile retries.

This keeps cleanup bounded without changing the successful training path.

## 4. Recovery state overrides a new checkpoint cadence

The r24 restart requested `recover.freq_steps=1`, and the resolved Hydra config
contained that value. After loading the old step1 recovery state, however, step2
did not save DCP state. `RecoverHandler.load()` restored the complete historical
`FrequencyControl` state, including its old `frequency_steps=2`, silently
overriding the current run configuration.

Implemented fix:

- restore the historical counters and last-trigger position;
- reapply `freq_epochs`, `freq_steps`, and `freq_secs` from the current
  `RecoverConfig` after loading state;
- cover the `2 -> 1` resumed-cadence change with a unit test.

The affected process predated this patch. Later launches honor a changed cadence
immediately.

## 5. Parallel gate shards incorrectly enforce a global threshold

`scripts/run_parallel_gate_model.sh` previously ran the frozen Go Gate on every
6--7-example shard. A shard can legitimately miss a 98% global threshold even
when all 50/100 examples finish and the merged run passes. The wrapper therefore
returned exit status 1 after producing every required record, making a completed
evaluation look like an inference failure.

Implemented fix:

- shard workers now use `--skip-gate-enforcement` while still writing their
  local diagnostics;
- the merged artifact remains the only place where the global gate is enforced;
- the wrapper records the tag as `experiment` and, when the model contains
  `merge_manifest.json`, requires and hashes that model provenance manifest.

The paired r21/r25 audit also records an evaluation limitation rather than
hiding it: exact token-overlap F1 penalizes explanatory text inside the
structured `answer` field. In the 50-example run, r25 predictions averaged 8.42
normalized tokens versus 5.22 for r21; seven regressions still had 100% reference
token recall. This does not make the strict project metric invalid, but requires
reporting length-attributed regressions separately from wrong or incomplete
answers.

## 6. Text-only evaluation unnecessarily loads a multimodal processor

The first provenance-enabled r21 gate launch on H800 failed before inference:
`AutoProcessor.from_pretrained()` could not rediscover the nested Qwen image
processor from the merged checkpoint, despite the text tokenizer and model
weights being valid. The evaluation path never supplies an image or video, so
this dependency was accidental.

Implemented fix:

- `scripts/run_sft_gate.py` now loads `AutoTokenizer` directly;
- the rendered chat-template token IDs are byte-for-byte identical to those
  obtained from `AutoProcessor.tokenizer` on the validated local checkpoint;
- `AutoModelForMultimodalLM` remains the model loader because that is the
  registered Qwen3.5 architecture, but no visual preprocessing is initialized.

This removes a cross-host processor-discovery failure and avoids initializing
unused image/video processing in every evaluation shard.

## 7. Eight independent evaluation workers repeat model load and kernel setup

The 100-example checkpoint comparison exposed two separate cold-path costs:

- on the local A800 host, eight workers reading the same cold 51 GB checkpoint
  took almost three minutes before the first record (the immediately preceding
  hot-cache r25 run started much sooner);
- on H800, the optional FLA path made eight workers independently autotune
  Qwen3.6's Qwen3.5-implementation GatedDeltaNet Triton kernels and created
  1,116 cache files. The first
  records took about four minutes and steady progress was worse than the
  validated A800 torch path for this low-batch interactive workload.

Implemented mitigations:

- `run_sft_gate.py` has an auditable
  `--disable-flash-linear-attention` compatibility switch selected before the
  Qwen modeling module is imported;
- gate, matrix, dataset, and formal-suite wrappers accept a persistent
  `--generation-url` / `OSS36_*_GENERATION_URL`, so many lightweight trajectory
  clients can share one vLLM model instance instead of loading one copy per
  shard and per evaluation cell;
- HTTP and local sampling use a per-prompt/per-turn seed, making results
  independent of request scheduling and shard completion order.

Two H800-specific launch requirements were found while validating the persistent
route:

- the venv `bin` directory must be in `PATH`; invoking its Python by absolute
  path alone left the bundled `ninja` invisible to child JIT processes;
- FlashInfer 0.6.6's SM90 GDN prefill source did not compile against the host
  CUDA/CCCL (`cuda::ptx::tensormap_replace_global_dim` missing). Launching vLLM
  with `--gdn-prefill-backend triton` selects the supported FLA/Triton path.
- vLLM auto-discovered the optional AWEX plugin even though this FSDP path has
  no Megatron installation, printing a non-fatal import traceback on every
  engine start. Training and evaluation launchers now set `VLLM_PLUGINS` to the
  documented empty allow-list; AReaL's worker extension is passed explicitly
  and does not depend on plugin discovery.

With both settings, the TP4 r21 vLLM service remained healthy for more than
three hours and completed a seven-turn greedy trajectory in 9.35 seconds with
valid JSON, legal references, and a final ANSWER. A direct Transformers replay
of the same prompt then matched all seven raw actions, tool results, token
counts, KEEP operations, final answer, and citations exactly; only run metadata
(experiment name, service URL, and run-config hash) differed. The persistent
service is therefore validated as the formal matrix backend, while the direct
path remains available as a correctness fallback.

## 8. Rollout/training log-probability tails are under-observed

The r24/r25 runs are synchronous (`max_head_offpolicyness=0`) and use one PPO
minibatch, but vLLM behavior log-probabilities still differ from the FSDP/SDPA
recomputation made from the same LoRA version:

- mean behavior importance weight was 0.94--0.96 (close to, but systematically
  below, the ideal 1.0);
- mean absolute log-probability difference was 0.13--0.20;
- the maximum absolute difference was 14.6--18.6 and the minimum observed
  importance weight was approximately `8e-9`;
- the configured upper-only ratio mask rejected just 0.01%--0.05% of tokens.

Different inference/training kernels can legitimately produce a distribution
mismatch, and AReaL's decoupled loss is designed to correct it. These aggregate
values therefore do not by themselves prove a framework bug or explain the
held-out regression. They do show that `avg/min/max` plus an upper-only filter
cannot answer how much gradient support is lost in the low-ratio tail. AReaL's
documented IcePop configuration uses a two-sided `[0.5, 5.0]` range.

Implemented diagnostic fix:

- the PPO loss now reports pre-filter fractions below 0.5 and above 2/5;
- it reports applied first/second importance-weight moments, allowing normalized
  effective sample size to be computed, plus a numerically bounded K3 statistic;
- `scripts/train_grpo.py` exposes audited `--rejection-lower/upper` overrides so
  a two-sided mask can be tested in an isolated trial without changing the
  frozen formal configuration;
- 81 functional/rejection-sampling tests pass, including a deterministic test
  that confirms an upper-rejected token remains visible in pre-filter tail
  diagnostics.

The lower-bound treatment completed one update from the r21 checkpoint with
`[0.5, 5.0]` on 2026-08-17. It observed:

- 7.2975% of pre-filter token ratios below 0.5, versus 0.0359% above 5;
- 7.2976% total rejection, applied importance-weight mean 0.92291, and
  normalized effective sample size 0.91084;
- mean absolute behavior/recompute log-probability difference 0.19675, maximum
  16.395, bounded K3 0.11821;
- successful update, actor loss 0.012236, gradient norm 0.097967, and no PPO
  clipping at this first update.

The lower tail is therefore material, not a cosmetic configuration change. The
result is recorded in
`artifacts/eval/grpo_v7_lower05_from_r21_one_step_training_summary.json`.

The upper-only control also completed one update from r21. It observed 7.0546%
of token ratios below 0.5 but, because no lower bound was active, filtered only
0.022382% in total; its normalized effective sample size was 0.93597 and its
actor update took 1,265.2 seconds. The treatment and control both exported 595
action-level records from 64 episodes. Their rollout F1 values were 0.45432 and
0.51138 respectively, but the asynchronous server produced different sampled
trajectories despite the shared global seed. The rollout metrics therefore
measure different batches and cannot estimate the causal effect of the lower
bound. The machine-readable comparison is
`artifacts/eval/grpo_v7_icepop_one_step_ablation.json`.

Both resulting policies were then evaluated against r21 on the same 50 held-out
prompts with vLLM TP4, temperature 1.0, and deterministic per-prompt/per-turn
request seeds. The r21/lower/upper F1 values were 0.39901, 0.40226, and 0.39160.
The paired lower-minus-r21 delta was only +0.00325 with 95% bootstrap interval
`[-0.05433, 0.06366]`; moreover support recall fell from 0.37667 to 0.355 and
citation precision macro fell from 0.51238 to 0.41467. Upper-only also reduced
F1, support recall, and completion. Neither update passes the multi-metric
non-degradation gate. A new same-trajectory rerun is therefore not justified;
r21 remains the selected checkpoint while the behavior/recompute mismatch is
investigated.

## 9. Individual action export multiplies actor work

The corrected one-update treatment made the training-side bottleneck explicit:

- 64 rollout episodes expanded into action-level training sequences with
  repeated multi-turn prefixes;
- AReaL reported a 1,338,700-token logical processing denominator for the actor
  update, while `masked_token_ratio=0.96993` means about 96.993% of model-input
  positions were excluded from the policy loss (context, padding, or rejected
  positions); only roughly 3% were active action-token positions;
- rollout plus log-probability recomputation completed in roughly seven
  minutes, while the PPO update alone took 1,250.3 seconds;
- the complete one-step job took 1,709.1 seconds (28.5 minutes), excluding final
  teardown.

This is both a performance issue and a method-design risk. Exporting every
action with its complete history repeatedly recomputes the same prefix, and an
episode with more actions contributes more action-level sequences. The latter
can implicitly weight long trajectories more heavily even when each action loss
is token-normalized.

Required work before a 375-step formal run:

- benchmark a larger token cap within the measured 13--23 GB/card free-memory
  margin, while preserving exact synchronized micro-batch counts;
- measure action count and processed-token expansion per episode;
- investigate an episode-packed implementation that masks observations and
  trains only generated action tokens without duplicating prefixes, while
  preserving per-action ABC advantages (the exact limitations of the current
  observation format and Qwen3.6 hybrid architecture are recorded in section
  12);
- benchmark H800 `DP2 x TP4` rollout separately, but do not expect rollout-only
  scaling to solve the actor bottleneck.

Implemented observability fix: FSDP rank 0 now logs the start of each
forward-only or forward-backward batch and reports micro-batch progress at about
25%, 50%, 75%, and 100%, including elapsed time and a simple ETA. This does not
add collectives or alter tensor execution, so it makes long updates visible
without changing training semantics.

At the measured rate, directly starting all 375 steps would take multiple days
and would be methodologically premature. The upper-only control and held-out
quality gate come first, followed by the packing/throughput benchmark.

## 10. Rollout dumps were remote-local and sampling was not restart-reproducible

Although `rollout.dump_to_file=true`, trajectory files generated by the H800
worker were not visible under the actor host's identically named fileroot. The
warning that `/code/openstatesearch/artifacts/areal` is not shared was literal:
the treatment produced 16 task files under the H800-local log directory. They
contained 595 action-level JSONL records and had to be copied back explicitly.
The formal launcher must either use a truly shared artifact root or collect and
hash remote rollout/log directories before teardown.

Implemented collection fix: `scripts/train_grpo.py` now accepts
`--rollout-artifact-source`. After the trainer exits it rsyncs the remote
`rollout/` tree and `rollout.log`, hashes every collected file, and writes
`rollout_collection_manifest.json`. A successful training process followed by
a failed collection returns a non-zero wrapper status instead of silently
claiming a complete artifact set.

The upper-only control also showed that a common engine seed is insufficient
for a strict paired ablation. Both jobs used the same r21 model, dataset seed,
vLLM seed, and sampling parameters, but async request ordering consumed the
server RNG differently; their synchronized micro-batch token layouts already
differed before either optimizer update. The current control is an independent
same-distribution replicate, not a same-trajectory counterfactual.

Implemented reproducibility fix for subsequent runs:

- `GenerationHyperparameters` carries an optional per-request seed and the
  remote vLLM payload forwards it;
- grouped rollout assigns stable, distinct sample indices `0..group_size-1` via
  workflow context;
- OpenStateSearch derives each request seed from the configured base seed,
  model version, stable trajectory ID, group sample index, and turn index;
- request-building, group-index, and stable-hash tests pass.

This preserves four distinct GRPO samples per prompt while making them
independent of async request order. The upper-only control was launched before
this patch and remains an independent same-distribution replicate; only future
launches may be labeled restart-reproducible. Both treatment and control rollout
trees have now been collected locally, each with 16 JSONL files and 595
action-level records, and each has a SHA-256-bearing
`rollout_collection_manifest.json`.

## 11. Persistent evaluation startup and model-name compatibility

The H800 TP4 evaluation path now distinguishes actual weight I/O from engine
startup. Across three Qwen3.6-27B models, safetensors weight reads took only
4.86--6.36 seconds. End-to-end service readiness took about two minutes because
process construction, tokenizer/multimodal setup, torch.compile, KV-cache
profiling, CUDA-graph capture, and multimodal warmup dominate. Warm compilation
reduced engine initialization from 57.0 to about 40 seconds, but cannot remove
the remaining fixed lifecycle cost. Keeping one service alive for all eight
evaluation shards avoids multiplying this cost by eight.

Two additional framework issues were reproduced and fixed:

- the first H800 launch attempted FlashInfer TRT-LLM allreduce-RMS fusion,
  emitted about 90 CUDA compilation errors, and only then fell back. The
  evaluation launcher now disables that known-broken fusion through vLLM's
  compilation pass config while retaining working custom all-reduce. The
  choice is recorded in the launch manifest and may be restored explicitly for
  a compatibility benchmark;
- a model loaded from an absolute H800 path was registered under that absolute
  API name, while local evaluators requested the corresponding relative model
  name. vLLM returned HTTP 404 even though `/v1/completions` existed. The
  launcher now accepts a distinct served-model name and records it in the
  manifest, so storage paths and API identity cannot be conflated.

The repaired persistent path completed three matched 50-example evaluations
with one model load per policy, verified retriever/model provenance, stable
per-turn seeds, and clean teardown. All H800 and actor GPUs returned to their
idle memory baselines afterward.

## 12. Generic tree/concat packing is unsafe or ineffective for this policy

AReaL advertises tree training as a way to share repeated prefixes, but it
cannot be enabled for Qwen3.6-27B. The model's Transformers implementation is
registered as `qwen3_5` and its 64 text layers comprise 48
`linear_attention` (GatedDeltaNet) layers and 16 `full_attention` layers. The
FSDP tree implementation replaces only the full-attention forward path with a
branch-aware mask. A GatedDeltaNet layer would still process the flattened trie
as one causal recurrence, allowing one branch to alter the hidden state of a
sibling. This can run without an exception while optimizing the wrong model.

Implemented safety fix: FSDP initialization now inspects `layer_types` on both
the outer Hugging Face config and `text_config`. When tree training is enabled,
any explicit type other than `full_attention` raises before distributed/model
initialization. Four unit tests pass, and the cached Qwen3.6 config is rejected
with `unsupported layer_types=['linear_attention']` as intended. Models that do
not expose `layer_types` retain the existing behavior rather than being blocked
without evidence.

The less invasive OpenAI `concat` export also does not solve this workload.
Every OpenStateSearch turn deliberately calls the policy with a new two-message
snapshot: the fixed system prompt plus a complete JSON serialization of the
current search state. It does not append the previous assistant action and tool
observation as a conversation. Consequently successive actions are independent
roots to AReaL's exact message-prefix matcher, so selecting `concat` leaves all
actions as leaf sequences and saves no actor forwards.

Changing the policy to an append-only dialogue could permit natural
episode-level packing, and per-action advantages can in principle be recovered
by assigning token weights so each action retains its own token mean. It is not
an exact runtime-only optimization here: it changes what the model conditions
on, changes truncation behavior, and requires a matching SFT/legal-space
adaptation plus a new gate. A correct exact optimization for the present
snapshot policy would require branch-aware recurrent-state handling in all 48
GatedDeltaNet layers (or a different model architecture), not merely an AReaL
configuration toggle.

## 13. Actor callback traffic inherited the cluster HTTP proxy

The formal Phase-A run completed step 68, then failed while publishing the next
LoRA version. `actor/0` sent the private callback request for
`continue_generation` through `oversea-squid2.ko.txyun:11080`; the proxy reset
the connection. This was not an OOM, numerical failure, or rollout timeout.
Rollout workers already had explicit proxy bypass variables, but actor
`SchedulingSpec.env_vars` was empty. Ray's worker launcher does not propagate
arbitrary parent environment variables, so setting `NO_PROXY` only on the
driver was insufficient.

Implemented runtime fix:

- actor workers now receive both `NO_PROXY` and `no_proxy` for localhost, the
  Ray head/callback host, the H800 worker, and the retriever host;
- the existing full DCP checkpoint was retained, and the run resumed at step
  68 without changing reward, data, sampling, or optimizer configuration;
- recovered `weight_update_v68` completed actor export, remote rollout load,
  callback, and generation resume. Training then advanced through step 81.

The failure, input hashes, recovery state, and verification timestamps are
recorded in
`artifacts/runtime/oss36-grpo-a-v8-logprob-fix_processedlogp-from-r21-5steps-gate-r1_incident_20260819_step68.json`.

## 14. Long jobs must not depend on transient driver and Ray sessions

After the recovered run completed step 81, the driver and Ray head disappeared
without a new AReaL traceback. The local eight actor GPUs were idle and the Ray
GCS port was unavailable, while the H800 vLLM subprocess on GPUs 0--3 remained
alive as an orphan. The exact external termination cause is not proven; the
evidence only establishes loss of the control plane rather than a model or
training exception.

Runtime mitigation:

- the orphan was identified by its exact trial command, port, and process tree
  and only that vLLM parent was terminated; the independent evaluation service
  on H800 GPUs 4--7 was preserved;
- the Ray head and training driver now run in named persistent `tmux` sessions,
  and the H800 Ray worker uses a detached `nohup` process;
- the periodic-evaluation watcher also runs in a persistent session and is
  attached to the actual training pane PID;
- the rebuilt cluster was verified as local DP8 actor plus H800 TP4 rollout,
  then recovered from `StepInfo(... global_step=81 ...)`. All four actor ranks
  completed `weight_update_v81`, including the remote callback, before step 82
  sampling began.

This is a lifecycle change only; training semantics and checkpoint lineage are
unchanged. The evidence is recorded in
`artifacts/runtime/oss36-grpo-a-v8-logprob-fix_processedlogp-from-r21-5steps-gate-r1_incident_20260819_step81_control_plane.json`.

## 15. Remote disk-weight transfer requires a persistent SSH listener

Step 82 completed all 19 actor forward/backward micro-batches and exported a
complete local LoRA adapter, but the H800 rollout worker could not pull it from
`10.82.123.139:22222`. All three rsync attempts failed with `Connection
refused`, the remote weight callback returned HTTP 500, and the trainer exited
before the rollout accepted v82 or a new durable DCP state was saved. This was
not an optimizer, reward, or numerical failure.

The actor host's ordinary SSH endpoint on port 2222 remained reachable with the
same key. The dedicated port-22222 listener used by
`AREAL_DISK_WEIGHT_SSH_PORT` had disappeared. Recovery therefore restored a
separate sshd listener with its PID and log under `artifacts/runtime`, then
proved the path with an isolated H800-to-actor rsync and matching SHA-256 hashes
for both adapter files. The long intervention window also outlived the old Ray
GCS, so the stale two-node cluster was rebuilt before resuming.

Because the failed v82 adapter did not include recoverable optimizer state, it
was not promoted as a checkpoint. The authoritative DCP state resumed as
`StepInfo(... global_step=81 ...)`. H800's v81/v82 cache directories were
removed before replay so the remote staging guard could not mistake stale
same-name adapters for the newly exported versions. Recovery verification
observed an actual remote v81 rsync, a 1.43-second vLLM load, and 64 concurrent
step-82 rollout requests. The periodic Step-80 evaluation remained intact.

The replay subsequently completed all 19 actor update micro-batches. H800
staged and loaded `oss36-policy-v82` in about 1.47 seconds, all eight actor
ranks completed the update callback, and a full recovery checkpoint including
optimizer state was saved at 17:04:47 with zero-based `global_step=81`
(`Train step 82/375 done`). The same listener then staged and loaded every
version through v90 without intervention; steps 83--90 completed and the
Step-90 periodic evaluation started in parallel with continued training. This
consecutive v82--v90 evidence closes the incident as resolved rather than
merely reachable after an isolated probe.

The full failure evidence, adapter hashes, recovery lineage, and verification
timestamps are recorded in
`artifacts/runtime/oss36-grpo-a-v8-logprob-fix_processedlogp-from-r21-5steps-gate-r1_incident_20260819_step82_weight_sync.json`.

## 16. Retry-appended rollout files are mutable audit inputs

The live H800 rollout tree continued growing after the through-step-81 trend
audit was generated. In the current step-81 source, task files `0..15` each
contain eight rollout starts (two appended four-sample batches), while files
`208..223` each contain four starts. The current mutable path therefore exposes
192 episode starts. The earlier audit observed 128, selected the latest complete
64-episode file batch, and remains a valid hash-bound snapshot of what was
visible when it was generated; it is no longer byte-for-byte reproducible from
the subsequently appended path.

This also exposes a boundary mismatch in the audit pipeline. The loader rejects
any individual file whose rollout-start count is not exactly four, before the
retry selector can choose a complete batch. Re-running the full trend now fails
on `rollout/81/0.jsonl` with eight starts. This does not affect training,
checkpointing, or the independent 50-example periodic evaluator.

It initially affected the future automated step-200 archive: that watcher
invokes `audit_phasea_checkpoint.sh`, which rebuilds the full
version-1-through-200 trend before archiving the adapter. The audit-only loader
now splits a physical file containing any positive multiple of four starts into
virtual four-rollout sources. The selector first keeps the newest complete
virtual source for each task, then chooses the newest required cross-task batch.
Partial appended batches still fail loudly.

No training or source-rollout change was made while the run is frozen. As a
non-mutating workaround, versions 82--92 were audited independently: 665/704
episodes were valid (94.46%), no episode failed after a single interaction, and
the output plus manifest hash successfully verified. The exact observation,
hashes, impact boundary, and deferred loader/snapshot fix are recorded in
`artifacts/audits/oss36-grpo-a-v8-logprob-fix_processedlogp-from-r21-5steps-gate-r1_rollout_append_audit_20260819.json`.

The repaired loader observed all 192 step-81 episode starts, represented them
as 48 virtual sources, and selected exactly 64 episodes from batch index 1 of
tasks 0--15 while recording 128 discarded retry episodes. The complete
version-1-through-92 trend then rebuilt successfully over 5,888 selected
episodes. Ten targeted tests, 23 related audit/final-result tests, and Ruff all
pass. The immutable resolution record and source/output hashes are in
`artifacts/audits/oss36-grpo-a-v8-logprob-fix_processedlogp-from-r21-5steps-gate-r1_rollout_append_resolution_20260819.json`.

## Validation status

Completed locally:

- 128/128 sequence-packing tests, 37/37 TrainController tests, and 81/81
  functional/rejection-sampling tests pass. These cover exact/infeasible
  synchronized allocation, bounded/non-retried collective RPC behavior, and
  rejection-tail diagnostics;
- the corrected lower-bound treatment completed a real DP8 FSDP2 update and
  published `epoch0epochstep0globalstep0` successfully;
- Ruff passes for every modified Python file;
- CLI validation confirms the eager-rollout override and manifest field.
- r25 completed two DP8 optimizer updates with batch size 16 and a 12,288-token
  cap, saved HF and DCP state every step, and tore down cleanly;
- the paired gate comparison CLI and shard merge tests pass, and the parallel
  wrapper passes `bash -n`.

Still required before calling the lifecycle changes production-proven:

- one multi-GPU failure-injection smoke using the patched teardown path;
- a distributed test before implementing or enabling a structural
  post-expansion dispatcher/dummy-microbatch change.

# AReaL runtime issues and mitigations

This file records failures observed during the Qwen3.5-27B multi-node GRPO
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
- Feasible mismatches are reallocated once to the common target and checked.

Run-level workaround:

- r24 resumes with a 10,240-token cap. This is below the original 12,288-token
  peak-memory configuration while keeping the observed common group count at or
  below the smallest local sequence count.

Longer-term fix:

- Multi-turn episodes are dispatched atomically across DP ranks, then expanded
  into a variable number of action sequences. A correct structural fix should
  balance the post-expansion action counts as well as token totals, or support
  masked dummy micro-batches. It must preserve episode/group semantics and be
  validated with a distributed FSDP test before replacing the cap workaround.

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

This keeps cleanup bounded without changing the successful training path.

## Validation required

- CPU unit tests for feasible/infeasible synchronized allocation;
- TrainController unit tests confirming bounded destroy RPC arguments;
- CLI validation for eager rollout manifest/override;
- one multi-GPU failure-injection smoke before treating teardown as fully fixed;
- one 8-rank batch with the structural post-expansion dispatcher fix before
  returning to an 8192-token cap.

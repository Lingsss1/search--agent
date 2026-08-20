# AReaL compatibility patch set

OpenStateSearch-36 pins AReaL at:

```text
repository: https://github.com/inclusionAI/AReaL.git
commit:     1f966b1a9dac370fbecdd38f4eea974ba05cc4b5
```

The project-specific runtime/correctness changes are exported here instead of
vendoring the AReaL checkout, model environments, or caches. Patch `0001`
contains changes to files already present at the pinned commit. Patches
`02`--`11` add new validation and regression-test files.

Apply from a clean AReaL checkout:

```bash
git checkout 1f966b1a9dac370fbecdd38f4eea974ba05cc4b5
for patch in /path/to/openstatesearch/patches/areal/*.patch; do
  git apply --check "$patch"
  git apply "$patch"
done
```

The exported set was verified by applying all 11 patches to a clean local clone:

```text
expected changed/untracked paths: 46
applied changed/untracked paths:  46
file-content SHA mismatches:       0
```

The behavior and rationale for the changes are documented in
[`AREAL_RUNTIME_ISSUES.md`](../../AREAL_RUNTIME_ISSUES.md). The most important
correctness fixes cover:

- processed sampling log-probabilities matching actor-temperature semantics;
- fail-fast handling when an entire workflow batch fails;
- exact cross-rank micro-batch counts for FSDP action-sequence training;
- bounded collective cleanup and partial-initialization teardown;
- recover-checkpoint cadence restoration;
- rollout pause/load/resume synchronization for disk LoRA updates;
- richer importance-ratio, KL, ESS, and timing diagnostics;
- Qwen3.6/Qwen3.5 tree-training compatibility validation.

Do not apply this patch set to a different AReaL revision without a fresh
three-way review and regression run.

# GPT-OSS `matmul_ogs` Stage Policy

## Problem

The native-frontier GPT-OSS `matmul_ogs` fixtures captured from SGLang replay
with the recorded Triton options:

```text
num_warps=8
num_stages=2
BLOCK_M=64
BLOCK_N=256
BLOCK_K=128
```

Native gfx942 replay succeeds for those fixtures, but the Salmon replay path
forces Triton to compile a gfx1250 wave32 kernel and then translates that code
object to gfx942. With the recorded two-stage pipeline, Triton reports:

```text
generated target: gfx1250:32
execution target: gfx942:64
shared memory required: 71648 bytes
hardware limit:         65536 bytes
```

That failure happens inside Triton's handle initialization before HIP launches
the kernel and before Salmon translates the code object. It is a target
resource-limit failure for the generated kernel, not an arithmetic mismatch.

## Native MXFP4 Versus Decomposed-BF16

There are two GPT-OSS `matmul_ogs` modes that must not be conflated:

- Native MXFP4 / mixed-dot path: exercises Triton's native MXFP4 lowering and
  the `matmul_ogs` epilogue used by the native-frontier fixtures.
- Decomposed-BF16 path: upcasts MXFP4 weights and scales to BF16 before
  `matmul_ogs`. This is useful control coverage, but passing it does not prove
  the native MXFP4 kernel.

Older decomposed-BF16 fixtures can pass while the native-frontier fixtures fail
resource accounting. Treat those as different coverage points.

## `num_stages=1` Policy

For forced-gfx1250 Salmon replay, the principled runnable policy is to compile
`matmul_ogs` with `num_stages=1`. This keeps the same tile shapes and reduces
the generated kernel's LDS requirement from `71648` bytes to `16384` bytes for
the assigned native-frontier fixtures.

This is not a Salmon metadata workaround and not a global matmul rewrite. The
stage count must be selected where Triton-kernels chooses `matmul_ogs` options:

```python
target = triton.runtime.driver.active.get_current_target()
if target.backend == "hip" and target.arch.startswith("gfx125"):
    num_stages = 1
```

The fixture replay tool also has `--salmon-num-stages 1` so exact recorded
fixtures can validate the forced-target policy without mutating the captured
fixture files.

## E2E Driver Requirements

For full GPT-OSS E2E validation of the native MXFP4 path, the Salmon run needs
to force the Triton target and use a fresh cache:

```bash
HSA_SALMON_CACHE_DISABLE=1 \
TRITON_ALWAYS_COMPILE=1 \
TRITON_CACHE_DIR=/tmp/triton_cache_after_matmul_ogs_lds_limit \
TRITON_CORPUS_FORCE_TARGET=gfx1250:32 \
HSA_HOTSWAP_ISA_OVERRIDE=gfx942 \
HSA_HOTSWAP_IR_RAISER=1 \
HSA_SALMON_STRICT=1 \
python3 compare_gpt_oss_sglang.py --native-mxfp4 ...
```

Do not enable `GPT_OSS_SGLANG_DECOMPOSE_MXFP4_FOR_GFX1250=1` when the goal is
to prove the native MXFP4 frontier. That flag intentionally selects the
decomposed-BF16 control path.

## Current Evidence

After selecting `num_stages=1` for the forced-gfx1250 replay path and fixing
Salmon's `v_minmax_num_f32` lowering, all four assigned native-frontier fixtures
match native with `mismatches=0`:

```text
matmul_ogsdflt_swiglu.matmul_ogs.2195296.000.pt
matmul_ogsdflt_swiglu.matmul_ogs.2195296.002.pt
matmul_ogsdflt_dflt.matmul_ogs.2195296.001.pt
matmul_ogsdflt_dflt.matmul_ogs.2195296.003.pt
```

The decomposed-BF16 control fixtures also pass with the same explicit
`--salmon-num-stages 1` replay policy.

## Runtime Scratch Follow-Up

The native-MXFP4 E2E exposed a separate runtime integration issue after the
fixture blockers moved: translated Salmon code objects can have a non-zero
target `private_segment_fixed_size` from spills while the submitted AQL dispatch
packet still carries `private_segment_size=0`. ROCR now repairs that mismatch in
the insufficient-scratch handler by reading the loaded kernel descriptor before
allocating scratch.

Follow-up is still needed to move that reconciliation earlier in the dispatch
path, ideally when packet resource metadata is produced or intercepted. The
async scratch handler should remain a safety net with structured diagnostics,
not the first place a valid translated kernel's scratch requirement becomes
visible to the runtime.

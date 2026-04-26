# Triage snapshot — 2026-04-20

Working notes for two pre-existing salmon failures surfaced by
`transpiler_tests` and `compare_correctness` while hardening the
wave32→wave64 cross-widening pipeline. This doc exists as a handoff
point so the next iteration does not start from scratch.

> **Current-state note (post-April 23):** the Matmul128 half of this
> triage is historical. The `Gfx1250Gpu.Matmul128x128*` family
> graduated from XFAIL after the V_CMP → V_CNDMASK per-lane-i1 shadow
> fix documented in `hotswap/docs/learnings.md` (2026-04-21), and
> `transpiler/tests/xfail.cmake` no longer marks Matmul tests as
> expected failures. The Softmax half remains current: `Gfx1250Gpu.Softmax`
> is still `WILL_FAIL` in `xfail.cmake`, now reaching launch and
> producing invalid `+inf` output.

## Status at a glance

| Test                                            | At pristine HEAD | With WaveNative + `ttmp8` lift stashed-in | Classification |
|-------------------------------------------------|------------------|-------------------------------------------|----------------|
| `Gfx1250Gpu.Vecadd`                             | Pass             | Pass                                      | Baseline       |
| `Gfx1250Gpu.Matmul64x64_*`                      | Pass             | Pass                                      | Baseline       |
| `Gfx1250Gpu.WmmaProbe_*`                        | Pass             | Pass                                      | Baseline       |
| `Gfx1250Gpu.Softmax`                            | **Fail** (historically no-write; current xfail is `+inf` output after launch) | Still XFAIL | Current open item |
| `Gfx1250Gpu.Matmul128x128_1tile`                | **Fail** (16 238 errors, `maxErr≈45`) | **Fixed post-triage** | Historical; see `learnings.md` 2026-04-21 |
| `Gfx1250Gpu.Matmul128x128_1tile_UniformDiag`    | Not built        | **Fixed post-triage**                   | Historical diagnostic |
| `corpus_add_fp32` (`compare_correctness`)       | `ALL_MATCH`      | `ALL_MATCH`                              | Baseline       |
| `corpus_softmax_fp32` (`compare_correctness`)   | Historical `CRASH` | Later notes report loud-failure / wrong-output variants | Needs fresh sweep |

Net at the time of this triage: the two Softmax failure modes
(test-data silently writes nothing, corpus crashes) and the Matmul128
numerical failure were orthogonal to the wave-size-translation work
done in that session. Current state diverges: Matmul128 is fixed,
`Gfx1250Gpu.Softmax` remains the current gtest XFAIL, and
`corpus_softmax_fp32` needs a fresh sweep before quoting a single
current status.

## Failure 1 — `Gfx1250Gpu.Softmax` silently writes nothing

This section records the historical no-write failure mode before the
`runPipeline` overload fix. The current gtest XFAIL reaches launch and
produces invalid `+inf` output on finite inputs; use the `xfail.cmake`
comment as the current one-line verdict.

### What we know

- `softmax_gfx1250.hsaco` is a Triton-generated kernel (see
  `test_data/gfx1250/README.md` provenance). It uses raw
  `buffer_load_b32` / `buffer_store_b32` through V# descriptors, loads
  and stores through `s_setreg_imm32_b32 HWREG(MODE,25,1)` at entry,
  does row-wise reduction via `v_permlanex16_b32` (emulated on gfx942
  through `ds_bpermute`), and `s_bfe_u32 sDST, ttmp8, 0x50019`
  boilerplate for `wave_id_in_workgroup`.
- The transpiler raises all of the above. The resulting raised IR
  (snapshotted at `/tmp/softmax.ll`) contains `@llvm.amdgcn.init.whole.wave`,
  valid `llvm.amdgcn.raw.buffer.store.i32` calls with well-formed
  `<4 x i32>` SRDs, and the usual `emitUnderExec` per-lane gating.
- The harness launches `grid=(1,1,1) wg=(maxFlatWorkgroupSize,1,1)`
  with `maxFlatWorkgroupSize` taken from AMDGPU metadata (128 for a
  Triton `num_warps=4` wave32 kernel, or 1024 by default).
- Preallocating the output buffer with `hipMemset(dOut, 0xA5, …)` or a
  `7.77f` sentinel proves the kernel **does not write at all** — the
  sentinel survives the launch bit-for-bit.
- The reported `maxErr ≈ 1.95e-3 = 1/512` is an artefact of the test
  comparing zero-initialised output against the CPU reference for an
  all-ones input (`softmax(1…) = 1/N`), not a signal that the kernel
  writes tiny values.

### What we suspect (graded by evidence)

1. **Cross-wave reduction collapsing under doubled participation**
   (≈60% confidence).
   Under `WaveNativeProjection`, `saved_exec = ballot.i64(init_whole_wave)`
   equals `-1` on any full-wave dispatch — including the 2×wave64
   halves of a 128-thread workgroup. That makes `emitLaneActiveBit`
   return true for all 64 target lanes in every `emitUnderExec`
   diamond. For a wave-size-oblivious pointwise kernel (vecadd) this
   is fine because the per-lane bounds check saturates the extra
   lanes. For softmax, the row-wise max / sum reductions now include
   32 "phantom" lanes per wave whose LDS and `v_permlanex16_b32`
   contributions the kernel never intended; the reduction state can
   saturate to `+inf` or wrap to `0`, which in turn makes the
   `exp(x - max)/sum` stage produce `NaN` / `0` outputs that a lane
   predicate strips before the store.
2. **Dropped `s_setreg_imm32_b32 HWREG(MODE,25,1)` at entry**
   (≈20% confidence).
   `handle_sopk.cpp:classifyHwreg()` routes `HWREG::MODE` through
   `HwregWrite::WarnDrop`. The corresponding warning prints during
   every Softmax transpile. The MODE register carries FP rounding /
   denormal / clamp bits that Triton writes to disable denormals; the
   downstream `v_exp_f32` / `v_div*` behaviour can differ between
   "denormal-flush" and "denormal-preserve". This alone would not
   explain "writes nothing", but it could produce `NaN`s in the
   reduction that a later `isfinite`-gated store drops.
3. **V# construction bug in `buildMubufSRD`** (≈20% confidence —
   currently the prior theory, now downgraded).
   `mubuf_addr.cpp:buildMubufSRD()` masks source `dw1` to `& 0xFFFF`
   to keep only `base_hi[47:32]`, discarding the source-ISA stride
   and flag bits in the upper 16 bits, and hardcodes `dw3 = 0`. For
   an untyped `BUFFER_STORE_DWORD` on gfx942 this should be
   sufficient: `TYPE=0`, `num_records=0x00FFFFFF` keeps the bounds
   check happy, and `DST_SEL` is unused for untyped ops. "Writes
   nothing at all" with a V# that is only *imperfect* would show up
   as partial OOB drops, not a clean no-op; a fully wrong V# would
   likely produce SIGBUS, not silent-drop. This is still worth
   probing, but it is no longer my best guess.

### Reproducer authored and wired in

- **Kernel**:
  `projects/rocr-runtime/runtime/hsa-runtime/hotswap/transpiler/tools/compare_correctness/kernels/mubuf_store_b32.hip`
  — minimal HIP kernel that forces a `buffer_store_b32` via
  `__builtin_amdgcn_raw_buffer_store_b32` with a hand-built SRD from
  `__builtin_amdgcn_make_buffer_rsrc`. Does one `global_load`, one
  arithmetic op, one `buffer_store_b32`. No cross-wave reductions, no
  LDS, no `permlanex16`.
- **Recipe**: `makeMubufStoreB32Recipe()` in
  `compare_correctness.cpp` (~line 2057). Registered in `allRecipes()`
  and added to `KERNELS` in the `Makefile`.
- **Run**:

  ```bash
  cd projects/rocr-runtime/runtime/hsa-runtime/hotswap/transpiler/tools/compare_correctness
  make ROCR_BUILD=$HOME/rocm-systems/projects/rocr-runtime/build
  LD_PRELOAD=./libsalmon_intercept.so \
    ./compare_correctness --recipe=mubuf_store_b32
  ```

  **What the result tells us:**

  | Outcome                                | Conclusion                                                                                 |
  |----------------------------------------|--------------------------------------------------------------------------------------------|
  | Salmon produces correct output         | MUBUF store path is fine; Softmax failure is in reduction / MODE / lane-doubling.          |
  | Salmon produces zeros or no writes     | MUBUF store path is the bug; `buildMubufSRD` / `handle_mubuf.cpp` is the fix site.         |
  | Salmon produces garbage (wrong values) | V# lowering is *close* but wrong on some lanes; probably `dw3` format bits for this shape. |

### Next concrete actions for Softmax

1. Run the MUBUF probe per the command above.
2. Based on the outcome, follow the branch in the table above.
3. If the MUBUF path is exonerated, the next probes to author are:
   - `mode_register_probe.hip`: pure `s_setreg_imm32_b32 HWREG(MODE,25,1)`
     + `v_pk_add_f32` using the lowest-denormal inputs that the mode
     bit affects (subnormal f16 pack).
   - `permlanex16_reduce.hip`: a 2-warp reduction using
     `__builtin_amdgcn_permlanex16_b32` under a large workgroup, to
     test whether the `ds_bpermute`-based emulation in
     `handle_valu_cross_lane.cpp` matches native behaviour when both
     wave64 halves are live.

## Historical Failure 2 — `Gfx1250Gpu.Matmul128x128_1tile` numerical mismatch

This section is retained as the pre-fix investigation trail. The root
cause was later identified as V_CMP → SGPR → V_CNDMASK wave-mask
truncation under cross-widening, and the Matmul128 family now passes
under the per-lane-i1 shadow fix documented in `learnings.md`.

### What we know

- `matmul_f16_large_gfx1250.hsaco` is an HIP kernel generating a
  128×128 output tile with WMMA accumulation.
- Against the passing `Matmul64x64` kernel, `matmul_f16_large` uniquely
  uses two instructions:
  - `s_set_vgpr_msb` (3 occurrences) — extends the VGPR-index bit
    width for the next ALU instruction so that operands can address
    VGPRs 256..1023 on gfx1250.
  - `v_bitop3_b32` — ternary bitwise LUT op. Handled in
    `handle_valu.cpp:V_BITOP3_B32` / `V_BITOP3_B16`.
- The `s_set_vgpr_msb` path in the transpiler is
  `handle_sopp.cpp:S_SET_VGPR_MSB` (stores the 8-bit low immediate
  into `ctx.vgprMSBs`) + `raise_context.cpp:computeVGPRAdjust()`
  (expands the 8 bits into per-operand-slot +256-bank offsets before
  each instruction dispatch) + `raiser.cpp:501` (one-shot reset to 0
  after each instruction). `AllocaRegFile::kVGPRCap = 1024`, so the
  upper banks have storage.

### What we suspect (graded by evidence)

1. **`s_set_vgpr_msb` adjustment dropped along one of the decoder
   paths** (≈65% confidence).
   The one-shot reset means the adjustment has to fire on the very
   next instruction. If that next instruction goes through a decoder
   path that does not consult `currentVGPRAdjust[]` (e.g. a VMEM
   address-computation branch, a VOPD half with a slot-mismatch, a
   DPP-modifier'd VALU source, or a cross-lane op that routes the
   operand through `permlanex16` emulation), the MSB is silently
   lost and the kernel reads/writes the wrong VGPR bank.
2. **`v_bitop3_b32` operating on wrong bits under cross-widening**
   (≈20% confidence).
   The LUT expansion in `handle_valu.cpp` is bit-serial: it ORs
   `minterm[i] AND expand(LUT[i])` for `i∈0..7`. That shape is
   wave-size-agnostic at the per-lane level, but if a source operand
   is a per-lane wave mask (v_bitop3 gets used on EXEC-derived i32
   masks in some AMD-lite patterns), the wave32→wave64 widening could
   corrupt the intermediate.
3. **An instruction on the matmul128 path that the transpiler
   handles but whose behaviour does not survive the WMMA →
   cross-target lowering** (≈15% confidence, residual).

### Reproducer authored and wired in

- **Kernel**:
  `projects/rocr-runtime/runtime/hsa-runtime/hotswap/transpiler/tools/compare_correctness/kernels/s_set_vgpr_msb.hip`
  — minimal HIP kernel with a long pointwise dependency chain (64
  live f32 values, then 32 pairwise products summed) designed to
  pressure the gfx1250 register allocator into using VGPRs ≥ 256,
  which forces `s_set_vgpr_msb` emission. No WMMA. No cross-lane ops.
- **Recipe**: `makeSSetVgprMsbRecipe()` in `compare_correctness.cpp`
  (~line 2147). Registered in `allRecipes()` and added to `KERNELS`
  in the `Makefile`. The CPU reference is kept in lock-step with the
  kernel's chain.
- **Run**:

  ```bash
  cd projects/rocr-runtime/runtime/hsa-runtime/hotswap/transpiler/tools/compare_correctness
  make ROCR_BUILD=$HOME/rocm-systems/projects/rocr-runtime/build
  llvm-objdump -d kernels/build/s_set_vgpr_msb.gfx1250.co | rg s_set_vgpr_msb
  LD_PRELOAD=./libsalmon_intercept.so \
    ./compare_correctness --recipe=s_set_vgpr_msb
  ```

  **What the result tells us:**

  | Outcome                                | Conclusion                                                                                                              |
  |----------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
  | Compiled .hsaco has no `s_set_vgpr_msb`| Repro is not pressing the allocator hard enough; bump live-state depth in the kernel's dependency chain.                |
  | Compiled .hsaco has `s_set_vgpr_msb` AND salmon matches | `s_set_vgpr_msb` alone is handled correctly; Matmul128 failure is in its interaction with WMMA lowering.  |
  | Compiled .hsaco has `s_set_vgpr_msb` AND salmon mismatches | The MSB path is broken; bisect by instruction class of the instruction immediately after the MSB site.    |

### Next concrete actions for Matmul128

1. Run the commands above.
2. If the probe *does* mismatch on salmon, inspect the raised IR and
   compare against the source disassembly for the instruction *after*
   each `s_set_vgpr_msb` — the VGPR indices in the IR should carry
   the `+256` bank offset. If they don't, trace back to whichever
   decoder path elided the adjustment.
3. If the probe matches on salmon, the failure is in the interaction
   between MSB-banked VGPRs and the WMMA → MFMA cross-target
   lowering, which points at `wmma_lowering.cpp`'s per-dword WWM
   marker logic (see commit `4944a33698`).

## Pending uncommitted work (orthogonal to both failures)

The following changes are staged in the working tree and are
independently useful (they unblock Matmul64×64 / WmmaProbe / etc.):

- `WaveNativeProjection` + `init_whole_wave`-based entry EXEC save in
  `wave_projection.cpp` and `reg_file.cpp`. Supersedes the prior
  per-MFMA `@llvm.amdgcn.strict.wwm` strategy that ran into
  `SIPreAllocateWWMRegs` pressure on 128×128 tiles.
- `s_bfe_u32 sDST, ttmp8, 0x50019` lift to `(workitem.id.x >> 5) & 0x1F`
  in `handle_sop2.cpp`, exempted from `TtmpWaveIdLeak` refusal via
  `isCanonicalWaveIdBfe()` in `wave_size_obstruction.cpp`, with a new
  `lit` fixture `c1_ttmp_wave_id_lift`. Replaces the broader
  `TtmpWaveIdLeak` refusal that swept in ~12 unrelated kernels.
- Test additions in `tests/gfx1250_gpu_test.cpp`
  (`Matmul128x128_1tile_UniformDiag` probe).
- Doc edits in `hotswap/docs/wave-size-translation.md` (§5.6.2
  wave_id lift rescue, §6 C1 update, §7 TtmpWaveIdLeak row).

These should be committed as their own unit before the Softmax /
Matmul128 work; they do not cause either residual failure.

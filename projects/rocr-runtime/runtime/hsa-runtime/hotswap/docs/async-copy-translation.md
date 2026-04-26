# Async-Copy Translation — FLAT `global_load_async_to_lds_b*` on gfx942

> **Status (2026-04-24): cross-target synchronous emulation landed.**
>
> The FLAT async global→LDS DMA family
> (`GLOBAL_LOAD_ASYNC_TO_LDS_B{8,32,64,128}`) and its companion wait
> counter (`S_WAIT_ASYNCCNT`) are now supported end-to-end on
> gfx1250 → gfx942 cross-target lifts via a synchronous per-lane
> emulation.  The emulation is principled but has a **documented
> semantic trade-off**: it loses the async-pipelining overlap the
> native DMA provides, not the per-lane memory state.  Every line
> of this doc, the handler code, the SemOp docstring, and the lit
> fixture spells out what that means in practice.
>
> **Scope of this document.**  The FLAT 1-D per-lane async DMA and
> its wait counter.  The sibling VIMAGE TENSOR descriptor-driven
> family (`TENSOR_LOAD_TO_LDS` / `TENSOR_STORE_FROM_LDS` plus
> `S_WAIT_TENSORCNT`) is still cross-target-refused — its
> descriptor encoding has no analog on gfx942 and a correct
> decomposition requires the work tracked in `tdm-translation.md`.
> `S_WAIT_TENSORCNT` is pre-registered as a SemOp with a no-op
> handler so that when the TENSOR emulation lands on the same
> posture as this one, the wait-counter canonicalisation is
> already in place.
>
> **North Star.**  The four GPT-OSS MoE expert-GEMM kernels
> (`_matmul_ogs_06d912ce88af`, `_matmul_ogs_0af655e6ea2b`,
> `_matmul_ogs_16674cb4d384`, `_matmul_ogs_25359f86c8d1`) use this
> opcode family to prefetch A/B tiles into LDS before the MFMA
> accumulation loop.  Pre-change they refused end-to-end on gfx942
> at the first async load; post-change all four lift to completion
> (`1670 / 1232 / 860 / 389` instructions respectively).

---

## 1. Problem in one paragraph

gfx1250 introduced `GLOBAL_LOAD_ASYNC_TO_LDS_B{8,32,64,128}` — a
per-lane 1-D global→LDS DMA that issues against the `ASYNCcnt`
hardware unit (distinct from `LOADcnt` / `STOREcnt` / `DScnt` /
`TENSORcnt` / `XCNT`) and synchronises via `S_WAIT_ASYNCCNT`.
The LLVM intrinsic family
`int_amdgcn_global_load_async_to_lds_b{8,32,64,128}` is gated by
`FeatureGFX1250Insts` and has no isel coverage on earlier
subtargets.  gfx942 has no asynccnt unit, no matching intrinsic,
and no equivalent burst path in the VFLAT encoding — a direct
cross-target lift via the intrinsic would fail at isel; a
refusal blocks every kernel that uses it.  Every GPT-OSS
MoE expert-GEMM variant (the "north star" above) hits this
opcode before reaching any of its MFMA body; without a
cross-target path there is no GPT-OSS MoE inference on gfx942.

## 2. ISA source-of-truth (verbatim pragma)

From `docs/manuals/instruction_manual.pdf §13.6.{9,10,11,12}`
(MI400 Shader Instruction Set, sp3 — identical across all four
widths, the LDS-store byte count is the only variation):

```
pragma "vector" do
  dsaddr  = LDS_BASE.b32 + VGPR[laneId][VDST.u32] + INST_OFFSET.b32;
  memaddr = ADDR;   // CalcGlobalAddr(VADDR, SADDR, IOFFSET)
  LDS[dsaddr].bN = MEM[memaddr].bN   (N = 8 / 32 / 64 / 128)
endpragma
```

Three observations the emulation relies on, each directly
readable off the pragma:

1. **Per-lane execution.**  `pragma "vector" do` iterates over
   the active lane set.  Both the LDS and global addresses are
   computed from per-lane VGPR values (`VGPR[laneId][...]`) and
   the memory effect is a per-lane write.  The synchronous
   emulation wraps the `load` + `store` pair in `emitUnderExec`
   so inactive lanes skip the entire memory round-trip —
   identical to the hardware's EXEC gating.

2. **`INST_OFFSET` applies to both pointers.**  `dsaddr` is
   computed as `LDS_BASE + vdst + INST_OFFSET` and `memaddr` is
   computed via `CalcGlobalAddr(VADDR, SADDR, IOFFSET)` where
   `IOFFSET` is the same `INST_OFFSET` immediate.  The
   same-target intrinsic folds this onto the operand bank via
   its `offset` immarg; the cross-target emulation materialises
   two `i8`-GEPs explicitly (one per pointer) before the
   `load`/`store` pair.  Zero-offset instructions elide both
   GEPs to keep the IR compact (the common case in the
   observed corpus).

3. **Per-lane LDS state is purely a function of the load's
   source data.**  There is no cross-lane interaction; no lane
   reads another lane's LDS slot or another lane's loaded
   value.  The emulation can lower each lane independently and
   preserve the source's semantic exactly — modulo completion
   ordering, addressed in the trade-off below.

## 3. The emulation

### 3.1 Operand decode (shared between both arms)

Both the same-target (gfx1250 → gfx1250) intrinsic-emit path
and the cross-target (gfx1250 → gfx942) synchronous path share
the operand decoder in `handle_flat.cpp::handleFLAT`.  Any shape
that the source compiler produces on gfx1250 is accepted by both
arms with identical parsing; the only divergence is in the
emission tail after the `globalPtr` / `ldsPtr` / `flatOffset` /
`cpolImm` quartet is computed.

The `FLAT_Global_Load_LDS_Pseudo<…, IsAsync=1>` InOperandList
(FLATInstructions.td:391-417) carries two shapes across all
four widths:

```
plain (4 srcs): vdst:VGPR_32, vaddr:VGPR_64,            offset, cpol
SADDR (5 srcs): vdst:VGPR_32, saddr:SReg_64, vaddr:VGPR_32, offset, cpol
```

`vdst` is an INPUT in the TableGen schema (because
`has_vdst = IsAsync = 1`) — it carries the per-lane LDS i32
offset, not a register written by the instruction.  The
emulation casts it to `ptr addrspace(3)` via `inttoptr i32`,
matching the LLVM intrinsic's `local_ptr_ty` operand shape.

The SADDR form's global address is `saddr + voff` where `voff`
is the zero-extended per-lane VGPR offset.  If the cpol
immediate's `SCAL` bit (`AMDGPU::CPol::SCAL = 0x400`) is set,
the hardware scales `voff` by the access element size before
adding to `saddr`.  The same-target arm relies on the intrinsic
folding this via the cpol immarg on isel; the cross-target arm
has to materialise the multiply explicitly in IR (see §3.2
below), because plain LLVM `load` / `store` has no encoding
bit for this.

### 3.2 Per-arm emission

**Same-target (`hasTensorOps == true`).**  Emit the native
intrinsic with `IntrInaccessibleMemOrArgMemOnly` preventing
downstream reorder across companion `S_WAIT_ASYNCCNT` barriers;
the backend re-emits the `global_load_async_to_lds_b*_SADDR`
real on isel, including the `scale_offset` cpol bit when it
was set in the source.  No change from the pre-2026-04-24
implementation.

**Cross-target (`hasTensorOps == false`).**  Emit a synchronous
per-lane `load` + `store` chain, wrapped in `emitUnderExec`:

```llvm
%ldsPtr   = inttoptr i32 <vdst-VGPR>          to ptr addrspace(3)
%gAddr    = <saddr + voff (with explicit mul by N if scale_offset)>
%gPtr     = inttoptr i64 %gAddr                to ptr addrspace(1)
%gPtrOff  = getelementptr i8, ptr addrspace(1) %gPtr,   i64 <flat_offset>
%lPtrOff  = getelementptr i8, ptr addrspace(3) %ldsPtr, i64 <flat_offset>
%loaded   = load  <T>, ptr addrspace(1) %gPtrOff, align <N>
            store <T> %loaded, ptr addrspace(3) %lPtrOff, align <N>
```

Width → `<T>` / `N`:

| Source SemOp | `<T>`         | `N` (bytes) |
| ------------- | -------------- | ----------- |
| `_B8`          | `i8`           | 1           |
| `_B32`         | `i32`          | 4           |
| `_B64`         | `<2 x i32>`    | 8           |
| `_B128`        | `<4 x i32>`    | 16          |

Vector types for b64 / b128 (not `i64` / `i128`) mirror the
choice made by `GLOBAL_LOAD_DWORDX{2,3,4}` for the same
aggregate shape: they carry the aligned-load alignment
attribute cleanly and let the backend pick
`global_load_{dwordx2,dwordx4}` / `ds_store_{b64,b128}` on
gfx942 from the alignment-derived codegen path.

### 3.3 `scale_offset` handling (per-arm gate)

The SADDR-form `scale_offset` multiply is materialised
conditionally on `!hasTensorOps`.  The gate is load-bearing:

*   On the same-target arm, the intrinsic is passed a
    `%gaddr = saddr + voff` (unscaled) together with the cpol
    immarg that includes the `SCAL` bit.  On isel the backend
    matches the unscaled pattern and emits the
    `global_load_async_to_lds_*_SADDR ... scale_offset` real,
    and the HARDWARE performs the scale at execution.
    Pre-multiplying in IR would produce `saddr + voff * N` and
    *either* (a) confuse the pattern match, silently dropping
    back to the plain form (correctness-preserving but
    bypassing the SADDR optimisation), OR (b) match the pattern
    anyway and apply SCAL on top, producing
    `saddr + voff * N * N` (a silent 4x / 16x / 64x
    miscompile).  The same-target lit fixture
    (`global_load_async_to_lds_same_target.ll`) pins the
    `inttoptr i64` shape with no intervening multiply — any
    regression that drops the gate fails that fixture
    immediately.

*   On the cross-target arm, there is no intrinsic; the
    `load` / `store` pair operates on the fully-computed
    pointer.  The multiply MUST be materialised here for the
    lane address to match the source's per-lane semantics.
    When `hasScaleOffset` is false (the matmul_ogs form) the
    multiply is elided; the HIP fixture (`scale_offset` form)
    exercises the explicit-multiply path.

### 3.4 cpol bits other than `scale_offset`

The cpol immediate also carries `th` (temporal hint:
`NT`/`HT`/`BYPASS`) and `scope` (CU/DEV/SYS).  Neither has a
representable mapping to the gfx942 cache-control encoding
(which uses `glc` / `slc` / `dlc` / `scc` bits on the atomic
or load instruction itself).  The cross-target arm drops both
deliberately and relies on the backend's default memory-model
cache behaviour — the same posture taken by
`sync-translation.md §5.3` for atomic scope recovery on
pre-gfx12 targets when the decoded modifiers have no target
mapping.  The common case in the GPT-OSS corpus is cpol = 0
(or cpol = 0x400 for `scale_offset` only) — every load
observed disassembles with default cache policy.

### 3.5 `S_WAIT_ASYNCCNT` handling

Before this change, LLVM's `AMDGPU::S_WAIT_ASYNCCNT` MC opcode
was not in `opcode_map.cpp` and fell through to
`raiser.cpp`'s generic `Unsupported instruction` refusal
(not `SemOp::Unknown`'s silent drop — the raiser reports loud
unknown-opcode diagnostics).  The change:

1.  Adds `SemOp::S_WAIT_ASYNCCNT` and `SemOp::S_WAIT_TENSORCNT`
    to the SOPP SemOp block in `semop.hpp`, with an
    inline-comment block describing the cross-target
    correctness argument.
2.  Canonicalises both in `opcode_map.cpp` alongside the
    existing wait-counter entries.
3.  Adds an **explicit** no-op arm in `handle_sopp.cpp` (not
    just a fall-through to the catch-all at the bottom of the
    function).  The explicit arm's inline comment is the
    single anchor a future reviewer can grep for to understand
    why the cross-target emulation doesn't need a real wait
    IR-node.

**The correctness argument (on both arms):** the IR's
load/store dataflow already carries the happens-before the
native counter was enforcing.  On the cross-target arm the
synchronous `load` + `store` has already retired by the time
the wait site is reached, so the wait is naturally a no-op.
On the same-target arm, the intrinsic's
`IntrInaccessibleMemOrArgMemOnly` annotation prevents
downstream passes from reordering loads/stores across the
wait site, and the gfx942 / gfx1250 backends re-emit the
appropriate native wait (`s_waitcnt lgkmcnt(0)` on gfx9,
`s_wait_asynccnt` on gfx12+) from that IR-level ordering
constraint.  Both arms emit IR-identically for the wait
itself (an empty lowering); the explicit branch in
`handle_sopp.cpp` exists purely for auditability.

## 4. The documented semantic trade-off

The **only** information the emulation loses is **pipelining
overlap** between the source's async DMA and unrelated VMEM /
LDS operations in the wave's own instruction stream.  On
gfx1250 the DMA unit fires while the wave's ALU path runs;
the synchronous emulation blocks the wave until the global
`load` retires before the `store` publishes the data to LDS.

**This is a throughput regression, not a correctness
regression.**  Every active lane's final LDS state is
bit-identical to the source's state as observed after
`s_wait_asynccnt 0`.  Kernels that depend on observable
effects *under a partially-elapsed asynccnt* — e.g. a
hand-written pipeliner reading back counter state during its
async window — are:

* not expressible in LLVM IR at all (there is no IR model for
  hardware counter state),
* not in the GPT-OSS corpus (all async DMAs observed fit the
  "prefetch then wait-before-read" pattern where the wait
  semantics are exactly what a synchronous `load` + `store`
  provides), and
* explicitly out of scope for this emulation.

The scoping argument — and the project-wide precedent — is
that a documented emulation with a named, audited trade-off
is preferable to a loud refusal when the alternative is "an
entire MoE inference workload cannot run on gfx942".  The
sibling TDM axis (descriptor-driven TENSOR ops) reaches a
different conclusion (still refused) because its source
encoding has no faithful 1-D decomposition — but the FLAT
async family IS already 1-D per-lane, so the decomposition is
direct.

## 5. Regression coverage

* **Lit.**
  `lit_tests/global_load_async_to_lds/global_load_async_to_lds.ll`
  pins the cross-target IR shape for all four widths
  (`b8` / `b32` / `b64` / `b128`).  It asserts the
  per-width `load <T>` + `store <T>` pair, the
  `scale_offset`-derived multiplier, the LDS-base
  `inttoptr i32` cast, AND a negative assertion that the
  intrinsic is NOT emitted on the cross-target arm
  (`IR-NOT: @llvm.amdgcn.global.load.async.to.lds.b*`).
  The companion
  `global_load_async_to_lds_same_target.ll` fixture (unchanged)
  keeps the same-target intrinsic-emit shape pinned.  The
  `global_load_async_to_lds_offset/` fixture closes the
  non-zero `flat_offset` branch directly: it compiles a b32
  async load with `offset = 16` and requires BOTH the global
  pointer and the LDS pointer to carry matching i8-GEPs before
  the load/store pair.
* **Batch raise.**
  `BatchRaise.Gfx1250TestData` and `BatchRaise.AiterGfx950`
  both continue to raise 100 % of their kernels (20 / 20 and
  27 / 27 respectively, identical to pre-change).  Neither
  corpus contains `global_load_async_to_lds_b*` instances
  today, but they catch any regression in shared handler code
  this commit touches (the `opcode_map` entry,
  `handle_sopp.cpp`'s new explicit arm, the shared operand
  decode in `handle_flat.cpp`).
* **Empirical (GTest-gated).**  `BatchRaise.ScopeDiscoveryGptOss`
  parses `scope_discovery/kernels/manifest.jsonl`, keeps only
  `status:"ok"` `_matmul_ogs_*.hsaco` entries, and raises the
  four GPT-OSS MoE expert-GEMM kernels end-to-end on
  `--isa=gfx1250 --target-isa=gfx942` with `expectedFailures = 0`.
  This turns the manual `raise_cli <file> ...` sweep
  (`OK ... (N/N)` for N ∈ {1670, 1232, 860, 389}) into a CI
  regression gate: removing the FLAT async-load SemOp mapping
  or synchronous cross-target emulation makes the GTest fail
  loudly.

## 6. Known gaps / follow-ups

* **Matmul_ogs lift gate closed.**  `BatchRaise.ScopeDiscoveryGptOss`
  now wires `scope_discovery/kernels/` into the GTest binary,
  discovers the four manifest-ok `_matmul_ogs_*.hsaco` kernels
  from `manifest.jsonl`, and requires a 0-failure gfx1250 →
  gfx942 raise.  The test is intentionally narrower than the
  full scope-discovery directory because that directory also
  contains unrelated Triton/GPT-OSS surfaces with pre-existing
  non-async-copy failures; the north-star async-copy consumers
  are now CI-gated directly.
* **Non-zero `flat_offset` lit gap closed.**  The
  `global_load_async_to_lds_offset/` fixture now pins the
  `async_gptr_off` / `async_lptr_off` GEP branch with a literal
  `offset = 16`.  This directly tests the ISA contract from
  §2 / §3.1: `INST_OFFSET` contributes to both `dsaddr` and
  `memaddr`, so the cross-target synchronous emulation must
  apply the same byte offset to both pointers.
* **b8 width lit gap closed.**  The
  `global_load_async_to_lds/` fixture now drives
  `__builtin_amdgcn_global_load_async_to_lds_b8` directly and
  checks the cross-target `load i8` / `store i8` shape with
  align 1, plus the same-target b8 intrinsic call.  The b8 case
  is therefore no longer covered only by inspection of the
  width switch in `handle_flat.cpp`.
* **Numerical-correctness probe still open.**  A lift-only
  GTest is now in CI, but no native-vs-salmon numerical gate
  exists for a real `_matmul_ogs_*` consumer.  Best-effort probe
  status: `tools/compare_correctness/kernels/triton/` has no
  `_matmul_ogs` recipe; the documented `~/anush_am/...`
  `triton_kernels` source path is absent on this checkout; the
  local Triton tree exposes the much larger `_matmul` /
  `_p_matmul` implementations whose captured specialisations
  require dozens of pointer, optional, scalar, and constexpr
  arguments plus metadata tensors.  Driving one captured
  `_matmul_ogs_*` shape through `compare_correctness` would
  therefore require a dedicated harness/schema effort, not a
  small async-copy audit patch.  The existing hand-authored
  `matmul_fp16` recipes remain useful `tl.dot` probes, but they
  are not the captured async-copy consumers.
* **cpol `th` / `scope` bits silently dropped.**  Per §3.4,
  these are tuning hints without a gfx942 equivalent.  If a
  future corpus kernel turns out to depend on them for
  correctness (rather than just performance), the gate needs
  revisiting — but the trade-off is explicit in the code and
  this doc.
* **VIMAGE TENSOR cross-target remains refused.**  That family
  is tracked in `tdm-translation.md` and is out of scope here.
  The `S_WAIT_TENSORCNT` SemOp is pre-registered with a no-op
  handler so the future TENSOR emulation can follow the same
  wait-counter posture this one landed.

## 7. Relationship to other axes

* **Sync (`sync-translation.md`):**  this is the async-axis
  instance of `§5.2.b`'s "no-op collapse path for
  pre-gfx12 targets" principle — the wait-counter is lowered
  to a no-op because the underlying async op has been emulated
  synchronously.  The doc's closing paragraph anticipated
  exactly this path ("emulation lowers TDM to synchronous
  buffer loads. Once async_copy is natively lowered,
  `s_wait_xcnt` has to become a real dependency in the IR");
  we take the opposite direction here (no native lowering, so
  the wait stays a no-op).
* **TDM (`tdm-translation.md`):**  the FLAT async family
  documented here is the 1-D per-lane sibling of the
  descriptor-driven 2-D / 4-D tile ops in
  `tdm-translation.md`.  The FLAT arm lifts now; the TDM arm
  does not.  `S_WAIT_TENSORCNT` is canonicalised as part of
  this change so that TDM, when it lands, inherits the same
  wait-counter posture.
* **Cross-cutting capability dispatch
  (`target-capability-dispatch.md`):**  the per-arm split in
  `handle_flat.cpp` is a literal instance of the project-wide
  "emit native when the target supports it, decompose only
  when it does not" dispatch pattern.  The capability bit is
  `ISAProfile::hasTensorOps` (the same bit used for the
  intrinsic-emit arm of the TENSOR family and the
  `GLOBAL_PREFETCH_B8` refusal); no new bit was introduced.

## 8. Implementation pointers

File-and-symbol map for the reviewer who wants to audit in
one pass.  Line numbers omitted — grep the symbol:

* `transpiler/semop.hpp` — SemOp enumerators and docstrings:
  `GLOBAL_LOAD_ASYNC_TO_LDS_B{8,32,64,128}`,
  `S_WAIT_ASYNCCNT`, `S_WAIT_TENSORCNT`.  The async-load doc
  block is the canonical reference for the trade-off
  argument; everything in this doc is consistent with it.
* `transpiler/opcode_map.cpp` — canonicalisation entries for
  `S_WAIT_ASYNCCNT` and `S_WAIT_TENSORCNT` (new in this
  commit).
* `transpiler/handle_flat.cpp::handleFLAT` —
  `GLOBAL_LOAD_ASYNC_TO_LDS_B*` shared operand decode +
  per-arm emission tail (same-target intrinsic, cross-target
  sync).
* `transpiler/handle_sopp.cpp::handleSOPP` — explicit no-op
  arm for `S_WAIT_ASYNCCNT` / `S_WAIT_TENSORCNT` with the
  correctness-argument comment block inlined.
* `transpiler/semop.cpp` — name switch entries for the two
  new SemOps.
* `transpiler/lit_tests/global_load_async_to_lds/` — the
  lit fixture directory holds both the same-target
  (`_same_target.ll`) and the cross-target
  (`global_load_async_to_lds.ll`) CHECK blocks against the
  same `.hip` source.

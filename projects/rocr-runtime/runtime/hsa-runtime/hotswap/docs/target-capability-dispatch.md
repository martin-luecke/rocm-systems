# Target-capability dispatch

> **Status:** design principle; one instance implemented today (WMMA
> in `handle_valu_vop3p.cpp`). The per-axis docs (`matrix-`,
> `sync-`, `abi-translation.md`) each reference this doc
> from their §5.0 / §4.0. §6's implementation-level question is
> resolved — see
> [`target-capability-dispatch-investigation.md`](target-capability-dispatch-investigation.md).

---

## 1. The principle

Every source SemOp whose target lowering is non-trivial must branch
on a **target-capability check** before emitting IR:

1. If the target ISA exposes the same primitive → emit it natively
   (one intrinsic call, no decomposition).
2. If the target ISA does not → run the axis-specific decompose /
   emulate / refuse path.

This is the only policy we run for handler dispatch. There is no
"fast path" in the byte-patching sense; the native branch **is** the
fast path. It falls out of the same IR-based pipeline every other
translation uses, with one extra `if` on an `ISAProfile` flag.

## 2. Why this is the right shape

Three properties follow from making the capability check the only
policy point:

1. **Same-family retargets are free.** gfx1251 → gfx1250 is identity
   on every capability flag relevant to the 170-kernel corpus. Every
   handler takes the native branch. The matrix, sync, and ABI
   axes collapse to "raise to IR, lower to target" with no
   axis-specific work at all.
2. **Cross-family correctness stays principled.** gfx1250 → gfx950
   has the decompose branches firing where the target lacks the
   primitive (WMMA→MFMA, async / tensor-copy emulation,
   split→monolithic barriers),
   and each branch is specified in the axis doc.
3. **Unsupported targets refuse cleanly.** A SemOp whose target
   neither supports it natively nor has a registered decompose path
   returns a `RaiseFailure`. No silent bad lowering.

The pre-conditions are declared per-SemOp in data (`SemOpAttrs` +
per-axis registration tables), not re-derived per-handler-call from
subtarget info.

## 3. The existing instance — WMMA handler

The one end-to-end implementation today:

```157:170:projects/rocr-runtime/runtime/hsa-runtime/hotswap/transpiler/handle_valu_vop3p.cpp
    Value *result_val;
    if (ctx.targetIsa.hasWMMA12) {
      Function *wmmaFn = Intrinsic::getOrInsertDeclaration(
          &ctx.M, Intrinsic::amdgcn_wmma_f32_16x16x32_f16,
          {v8f32Ty, v16f16Ty});
      result_val = ctx.B.CreateCall(wmmaFn, {
          ctx.B.getFalse(), a,
          ctx.B.getFalse(), b,
          ConstantInt::get(Type::getInt16Ty(ctx.C), 0), c,
          ctx.B.getFalse(), ctx.B.getFalse()
      }, "wmma");
    } else {
      result_val = emitWMMAtoMFMA(ctx, a, b, c);
    }
```

`ctx.targetIsa.hasWMMA12` is an `ISAProfile` field backed by
`FeatureWMMA128bInsts` / `FeatureWMMA256bInsts`. The native branch
costs one intrinsic call; the decompose branch runs the full
`ds_bpermute`-based lowering in `wmma_lowering.cpp`.

Every new translation we add — async / tensor-copy families, split
barriers, split waitcnt, scaled MFMA, cluster barriers — follows this
same shape.

## 4. Capability bits we need

Today `ISAProfile` carries five flags (`waveSize`, `hasAGPR`,
`hasMFMA`, `hasVOPD`, `hasScalarFP`, `hasWMMA12`). Per-axis docs
enumerate the rest:

| Axis | New bits | Backing `FeatureX` |
|---|---|---|
| Matrix (`matrix-translation.md §5.0.1`) | `hasWMMA1250`, `hasScaledMFMA` | `FeatureGFX1250Insts`, (new `FeatureScaledMFMA`) |
| Async / tensor-copy families | future `hasTDM`, `hasAsyncCopyGlobalToShared` | `FeatureGFX1250Insts` |
| Sync (`sync-translation.md §5.0.1`) | `hasSplitBarriers`, `hasSplitWaitCnt`, `hasSWaitAlu`, `hasClusterBarriers` | `FeatureGFX12Insts`, `FeatureClusters` |
| ABI (`abi-translation.md §4.0.1`) | `hasGFX12UserSGPRLayout`, `hasFlatScratchArchitected`, `hasFlatLDSArchitected` | `AMDGPU::isGFX12Plus(STI)`, `FeatureArchitectedFlatScratch`, `FeatureArchitectedSGPRs` |

Each field is a one-line `STI.hasFeature(...)` read in
`ISAProfile::fromSubtarget` — no new infrastructure.

## 5. Where the capability check lives

Two coexisting mechanisms, both driven by `ISAProfile`:

1. **Per-SemOp handler branch.** The WMMA-style `if
   (ctx.targetIsa.hasX) native; else decompose;` inside each handler.
   This is where new axes extend today.
2. **Per-kernel pre-translation classifier.** The existing
   `buildObstructionReport` (wave-size axis) runs before any handler
   and reports refusals. It should generalise into a
   `TranslationReport` that folds in the target-capability coverage
   for *all* axes, so handlers consume a pre-computed decision
   instead of re-deriving it.

Until (2) exists, every handler does its own check. This is the
status quo for WMMA, and each new axis doc §5.0 documents its
equivalent. Promoting the shared classifier is specified in
[`target-capability-dispatch-investigation.md §5`](target-capability-dispatch-investigation.md);
the T-series task breakdown lives in §7 of that doc.

## 6. Resolved — lift LLVM's existing per-opcode predicate query

For each source SemOp we need to answer "does the target ISA
support this primitive natively?" as a function of
`(sourceSemOp, targetFeatureSet)`.

**Resolved.** `llvm/utils/TableGen/InstrInfoEmitter.cpp::
emitFeatureVerifier` already generates
`llvm::AMDGPU_MC::isOpcodeAvailable(unsigned Opcode, const
FeatureBitset &Features)` and `computeRequiredFeatures(unsigned
Opcode)` into `AMDGPUGenInstrInfo.inc`, reading the same
per-instruction `Predicates = [...]` lists declared in
`AMDGPU.td`. The functions are gated behind
`GET_AVAILABLE_OPCODE_CHECKER`, which is not defined in any LLVM
TU, so the symbols never ship in the public LLVM build — but we
can define the gate on our side and pull the functions into one
Salmon-owned TU with zero upstream changes.

`ISAProfile::supports(SemOp)` dispatches via `OpcodeMap`'s inverse
(`SemOp → SmallVector<unsigned>` of MC opcodes already maintained
by the raiser): for each mapped MC opcode, call
`isOpcodeAvailable(opc, featureBits)`; return true if any opcode
is available. Unregistered SemOps hard-refuse via
`report_fatal_error` — silent defaults are rejected.

Full option evaluation (hand table, TableGen backend extension,
feature-set intersection, backend-error fallback), API sketch,
migration plan for `buildObstructionReport → TranslationReport`,
and concrete task breakdown live in
[`target-capability-dispatch-investigation.md`](target-capability-dispatch-investigation.md).

## 7. How each axis extends this doc

| Axis doc | Section | Scope |
|---|---|---|
| `matrix-translation.md` §5.0 | Per-shape dispatch (`hasWMMA12`, `hasScaledMFMA`, …); same-family retarget uses the native branch exclusively |
| `sync-translation.md` §5.0 | Split-barrier / split-waitcnt / cluster-barrier dispatch; collapse branches are pre-gfx12 only |
| `abi-translation.md` §4.0 | User-SGPR layout + hidden-arg compatibility; every capability is identity on same-family |

## 8. What this doc is *not*

- Not a replacement for the per-axis specs. Each axis still owns its
  decompose templates, refusal criteria, and gate formulations.
- Not a specification of the `TranslationReport` generalisation of
  `buildObstructionReport` — that specification lives in
  [`target-capability-dispatch-investigation.md §5`](target-capability-dispatch-investigation.md).
- Not an argument for a byte-level fast path. The whole point of the
  native branch is that it subsumes what a fast path would do while
  staying inside the IR pipeline.

# hotswap / transpiler design docs

This folder collects the prescriptive design specs for the
hotswap/transpiler pipeline. Each document addresses one translation
concern in depth and states its own decision procedure and refusal
criteria, so kernels that fall outside the framework fail loudly
instead of silently miscompiling.

## Reading order

Start here if you are new to the pipeline:

1. `hotswap-architecture.md` — end-to-end pipeline, components, and
   data flow (load-time transpilation, MC-level decode, IR raise,
   target lowering).
2. `gfx1250-on-gfx950-analysis.md` — the source/target ISA delta
   (encodings, capabilities, wave-size gap).
3. `wave-size-translation.md` — the wave-size axis: SIMT Predicated
   Execution model, modulo-replication projection, obstruction classes
   C1–C4, decision procedure, and the cross-lane primitive rewrite
   table.
4. `gpt-oss-derisking.md` — corpus-level audit: which GPT-OSS
   kernels do / do not fall within the wave-size framework.

## Per-axis translation design docs

Each of these specifies one concern. They are written to be read
independently but cross-reference each other where the axes interact.

| Doc | Axis | Status |
|---|---|---|
| `wave-size-translation.md` | EXEC divergence (SPE), wave32 → wave64 projection, cross-lane primitive rewrites, C1–C4 obstruction classifier | implemented end-to-end for the GPT-OSS-scoped cross-lane surface; the broader Triton `compare_correctness` corpus still has tracked residuals in `learnings.md` |
| `modrep-predicate-chain.md` | Cross-widening failures where the kernel's own predicate chain reads `workitem.id.x()` unmasked — outside the existing C1–C4 classification | O1 loud-refuse classifier landed for MODREP; default WaveNative structurally suppresses the main examples (`canary_bpermute_scan_fp32`, `rmsnorm_fp32`, `swiglu_fp32`, `corpus_layernorm_fp32` match in the current evidence) |
| `abi-translation.md` | Kernel descriptor, kernarg layout, hidden args, embedded V# / T# | design proposal; 80% path works today |
| `sync-translation.md` | Barriers (incl. gfx12 split), waitcnt, atomic scopes, cache ops, cluster sync | design proposal; barrier + waitcnt lowered today, scopes conservative |
| `matrix-translation.md` | WMMA → MFMA, per-shape fragment tables, MXFP scaled path | F16 WMMA cross-target path implemented for the current Matmul64/128 gtest family; additional shapes and MXFP scaled paths remain designed / staged |
| `buffer-store-lowering.md` | `BUFFER_STORE_*` → `amdgcn.raw.buffer.store`; implicit ABI coupling via `addrspace(5)` allocas (R1 fix) | implemented; cross-references `abi-translation.md` §3.4 / §4.2 |

Each doc has the same high-level structure: problem statement,
source/target models, translation design, principled fail-loudly
gates, decision procedure, engineering tasks, open questions,
cross-axis relationships.

## Cross-cutting principles

- `target-capability-dispatch.md` — every non-trivial SemOp branches
  on a target-capability flag before emitting IR: native intrinsic
  when the target supports it, decompose / emulate / refuse when it
  does not. Each per-axis doc's §5.0 (or §4.0 for ABI) is an instance.
  Captures the rationale for why same-family retargets (e.g.,
  gfx1251 → gfx1250) reach a native path on every axis without a
  separate fast code path.
- `target-capability-dispatch-investigation.md` — LLVM-reuse study
  that resolves §6 of the principle doc. Recommends lifting the
  per-opcode feature query that
  `llvm/utils/TableGen/InstrInfoEmitter.cpp` already generates
  (`AMDGPU_MC::isOpcodeAvailable`) into a Salmon-owned TU and
  exposing it on `ISAProfile::supports(SemOp)`. Includes the
  migration plan that generalises `buildObstructionReport` into a
  multi-axis `TranslationReport` and a T-series task breakdown with
  LoC estimates.

## Historical / supporting docs

- `hotswap-design-proposal.md`, `hotswap-design-meeting-notes.md` —
  pre-transpiler (same-family) hotswap design context.
- `hotswap-rewrite-rules.md`, `hotswap-wheel-integration.md` —
  integration concerns for the same-family path.
- `ORIGINAL_HOTSWAP_DEEP_DIVE.md` — deep dive on the pre-refactor
  implementation; kept for reference, superseded by the
  per-axis docs above for cross-family work.
- `transpiler-challenge-questions.md` — unresolved questions the
  per-axis docs are intended to answer.

## How the per-axis docs map to code

All axes hang their per-SemOp contracts off the central extension
point in
`projects/rocr-runtime/runtime/hsa-runtime/hotswap/transpiler/sem_op_attrs.hpp`:

- `SemOpAttrs` carries per-SemOp attribute bits.
  - `routesExecThroughStoreExec` (SPE gate, implemented).
  - `isScheduleHint` (sync axis, proposed).
  - `requiresUnifiedBarrierSemantics` (sync axis, proposed).
  - `requiresUniformExec` (matrix and future collective axes, proposed).
- Each attr has a startup coverage verifier
  (`verifyExecAttrCoverage` today; others proposed in each axis
  doc) that fails loud if new LLVM-emitted SemOps slip through
  without a declared classification.
- Per-handler-file registrations (`getHandlerSOP1Attrs`,
  `getHandlerSOP2Attrs`, `getHandlerVALU_VcmpAttrs`, …) aggregate
  into the O(1) lookup table.

The per-axis docs below each add one-to-few bits to
`SemOpAttrs` plus a corresponding verifier and per-kernel gate. No
axis reaches outside the `SemOp` + attr pattern.

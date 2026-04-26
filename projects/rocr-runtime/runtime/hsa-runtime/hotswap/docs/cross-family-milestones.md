# HotSwap Cross-Family Translation — Week-by-Week Milestones

**Scope.** gfx1250 → gfx942 / gfx950. Program-level window April 17 – July 15,
2026 (13 weeks, 7 engineers). This document specifies one component of the
HotSwap program: the **cross-family translation** workstream. Same-family
retarget (B0-to-A0 stepping) is a sibling workstream owned by @harsh-menon
and cited by PR number in §3; it is not re-specified here.

**Relation to program-level doc.** Leadership's program plan
(`hotswap-milestones.pdf`) frames cross-family translation as two
candidate schedules, A (binary patching) and B (raising to LLVM IR), with
a May 1 decision gate. This document lays out schedule B in detail. If
the May 1 gate resolves in favour of schedule A, the week-by-week
milestones in §4 are discarded and the legacy byte-level transpiler
(`hotswap/transpiler.cpp`) is ported instead.

**Authorship.** Design and schedule authored by M. Luecke. Owned by
@harsh-menon at the program level.

---

## 1. End-State Goals (July 15)

1. Run OpenAI GPT-OSS 120B through vllm / sglang on gfx942 / gfx950 using
   gfx1250 kernels transpiled by HotSwap.
2. All B0-to-A0 patches landed on main in COMGR (sibling workstream;
   tracked here only as a dependency — §3.1).
3. Cross-family execution validation on the North Star corpus: every
   kernel in the GPT-OSS Triton 15 and the hipBLASLt 147 GEMM subsets
   either executes correctly on gfx942 / gfx950, or is classified into
   a loud-fail refusal class documented in the per-axis design docs
   (`hotswap/docs/*-translation.md`).

Goal 3 replaces leadership's ATT-trace goal. The trace pipeline is
orthogonal to the translation-engine choice (leadership's §3.1
workstream C) and is out of scope for this document.

---

## 2. Structure of This Document

- §3: Current state of both cross-family prototypes as of April 22, 2026,
  with post-April-23 result updates folded in where the checked-in
  evidence supersedes the original snapshot.
  This replaces leadership's §3 ("shared weeks 1–2"), because (a) workstream
  A — sibling same-family stepping — is already in flight under its own
  schedule, and (b) this document's workstream is not a weeks-1–2 fork
  prototype but the current `hotswap/transpiler/` mainline code.
- §4: Cross-family IR-raising schedule, weeks 3–13. Mirrors leadership's
  §5 structure.
- §5: Factual side-by-side of the two cross-family prototypes. This is
  the May-1-gate evidence section, built from an executable
  canary + corpus harness (§5.1, §5.2).
- §6: Risk summary (mirrors leadership's §6).
- §7: What to do if the cross-family plan hits a wall (mirrors
  leadership's §7). The legacy byte-level transpiler stays available as
  the named fallback.

---

## 3. Current State — April 22, 2026 Snapshot, Updated With Latest Results

### 3.1 Workstream map

Letters follow leadership's §3.1 scheme for continuity with the program
plan. This document specifies workstream D; workstreams A–C are
referenced only where they interact with D.

| Workstream | Owner | State | Reference |
|---|---|---|---|
| A — B0-to-A0 landing (COMGR, same-family) | @harsh-menon | Apr-22 snapshot: 1/3 merged, 2/3 + 3/3 in flight | `harsh-amd/llvm-project` PRs #2202 (merged), `hotswap-infra-elf`, `hotswap-infra-b0a0` |
| B — Binary-patching transpiler port (cross-family, legacy) | per leadership §3.1 (2 eng) | Apr-22 snapshot: source in `hotswap/transpiler.cpp` (mainline); 5,780 LOC monolithic; 42/42 branch unit tests (Mar 25); no cross-family commits since Mar 25 | `hotswap/transpiler.cpp` |
| C — Trace pipeline | per leadership §3.1 (2 eng) | Out of scope for this document | — |
| D — LLVM IR raising (cross-family, this document) | This document (2 eng pre-May-1; §4.1 at full reallocation) | In `hotswap/transpiler/` (mainline); 39 `.cpp` files, ~20k LOC translation core, 7 per-axis design specs, full GoogleTest suite; post-Apr-23 evidence fixes the Matmul128 gtest family and moves the Triton compare corpus to 76/104 matching recipes | `hotswap/transpiler/`, `hotswap/docs/`, `hotswap/docs/learnings.md` |

Both cross-family prototypes (B, D) are in-tree and can be invoked on
the same kernel in the same process via the `HSA_HOTSWAP_IR_RAISER=1`
environment switch (`hotswap/transpiler/README.md`). This makes §5's
side-by-side reproducible end-to-end without porting either side.

### 3.2 Workstream D — IR raising (this document)

State from executable evidence, with post-April-23 updates:

- **Raise rate.** 27 / 27 production gfx950 kernels (corpus
  `hotswap/test_data/rocm-hotswap-testing/`); GPT-OSS Triton 15 all
  raise cleanly (corpus audit in `hotswap/docs/gpt-oss-derisking.md`).
- **Cross-family execution.** `Gfx1250Gpu.Vecadd`, `Matmul64x64_*`,
  `Matmul128x128_*`, and `WmmaProbe_*` pass end-to-end gfx1250 → gfx942
  in `tests/gfx1250_gpu_test.cpp`. `Softmax` remains the current
  `Gfx1250Gpu` XFAIL and is tracked by
  `hotswap/docs/triage-2026-04-20-softmax-matmul128.md` plus the
  `xfail.cmake` comment.
- **EXEC divergence.** SIMT Predicated Execution + WaveNative projection
  landed 2026-04-21 (`hotswap/docs/wave-size-translation.md`,
  `hotswap/docs/modrep-predicate-chain.md`). Closes the
  "EXEC-mask-as-scalar-boolean" silent-miscompile risk the program-level
  design proposal listed as HIGH.
- **Per-axis design.** Seven design specs under `hotswap/docs/` covering
  wave-size, matrix (WMMA→MFMA incl. MXFP scaled), sync, ABI, MODREP,
  and async / tensor-copy follow-up work
  predicate chain, and target-capability dispatch. Each spec states its
  own refusal criteria and decision procedure.
- **Corpus derisking.** GPT-OSS Triton 15 audited against the §6
  obstruction classification in `wave-size-translation.md`: zero
  unrewritable C1 sites, zero non-commutative atomics (C3), zero
  `v_permlane64`. Every obstruction maps to outcome (a) or (b) — i.e.
  this workstream will not refuse a GPT-OSS kernel on wave-size grounds.

### 3.3 Workstream B — binary patching (legacy)

State as of April 22, from `git log`:

- `hotswap/transpiler.cpp` last substantive cross-family commit:
  `50519f7ecc` — "42/42 ALL PASS! Fix RSRC1 VGPR/SGPR field swap +
  SGPR ceiling division" — 2026-03-25.
- No cross-family commits to `transpiler.cpp` in April.
- Harsh's three COMGR PRs (`hotswap-infra-elf`, `hotswap-infra-mc`,
  `hotswap-infra-b0a0`) land **B0-to-A0 same-family stepping only**.
  Quoting the 3/3 commit body: *"Add the GFX1250 B0-to-A0
  silicon-stepping rewrite policy."* None of the three touch
  cross-family translation.
- Leadership's §3.1 workstream B — *"Port branch transpiler to mainline
  behind COMGR"* — is not in flight in any public branch we can see.
  The branch transpiler is the present `hotswap/transpiler.cpp`; the
  separate COMGR port has not begun.

These are observations; scheduling workstream C is not in our scope.

### 3.4 Relation to the May 1 decision gate

Leadership's §3.4 commits the team to one cross-family path at May 1
based on workstream D's written report. This document contributes to
that decision as the raising-path evidence package: §4 states the
week-by-week schedule the raising path would follow if chosen, and §5
provides the current-state side-by-side between the two prototypes. If
the gate resolves in favour of schedule A (binary patching),
leadership's §4 stands as the plan and this document is retired; §5's
harness remains useful for regression tracking of the legacy path
going forward.

---

## 4. Schedule — IR raising, weeks 3–13

This section mirrors leadership's §5 ("Schedule B: All-In on Raising
to LLVM IR"). It states the schedule the raising path would follow from
May 1 onward if the decision gate resolves in its favour.

### 4.1 Team reallocation at week 3

Leadership's §5.1 renames the raising engine to workstream **E** and
keeps trace as workstream **C** (2 engineers). This document drops the
trace goal (see §1), so all 7 engineers go to **E**. E is subdivided
into six sub-workstreams, one per translation axis or infrastructure
concern. Sub-letter ordering matches the design-doc reading order in
`hotswap/docs/README.md`.

| Sub-workstream | People | Window | Focus |
|---|---:|---|---|
| E1 — Wave-size axis hardening | 1 | W3–W6 | P1/P2/P4/P5/P6 intrinsic lifts, C1 operand-range validator, MODREP predicate-chain class fix |
| E2 — Matrix axis | 1 | W3–W10 | Remaining WMMA/SWMMAC shapes; MXFP scaled path; per-shape fragment tables |
| E3 — TDM cross-target | 2 | W3–W11 | Tensor-descriptor decode, per-lane distribution, async-issue / waitcnt pairing (largest open design item) |
| E4 — Sync + ABI | 1 | W3–W9 | Split-barrier lowering, atomic-scope round-trip, named-barrier refusal gate, hidden-arg surface, embedded V# refusal, user-SGPR preload verification |
| E5 — Infrastructure + COMGR upstreaming | 1 | W3–W13 | Size-sweep harness (W3 deliverable), `TranslationReport` generalisation, `ISAProfile::supports(SemOp)`, in-process `llc`, **upstreaming to `ROCm/llvm-project:amd/comgr/`** — COMGR API surface, directory relocation, ROCR loader refactor (see §4.3) |
| E6 — Validation + reject-gate coverage | 1 | W3–W13 | Canary + corpus sweeps consuming E5's harness; AITER regression follow-up; triage items; cross-family execution expansion |

Sub-workstream entry criteria: the current state of `hotswap/transpiler/`
mainline — no upstreaming, no fork-to-mainline port, no prototype-to-
product transition. §4.3 explains why.

### 4.2 Week-by-week

Dates are Friday-Friday to match leadership's §4.2 / §5.2.

- **Week 3 (May 1–8):** E1 lands P1 (`ds_bpermute`) + P2/P4 (permlane) intrinsic lifts and the C1 operand-range validator. E5 stands up the size-sweep harness AND opens the upstreaming conversation with the COMGR code owner (@jacob-lambert): API surface for `amd_comgr_hotswap_*` cross-family entry point, reusing the pattern Harsh's workstream A is landing. E3 delivers the TDM cross-target design proposal (IR shape, fence ordering, refusal criteria). E2 begins BF16 shape. *Milestone: size-sweep harness online; COMGR API proposal circulated.*
- **Week 4 (May 8–15):** E1 lands P5 (DPP intrinsic lift), closing the largest open silent-miscompile (5 GPT-OSS Triton kernels). E2 BF16 shape executes cross-family. E4 lands split-barrier lowering. E3 TDM design reviewed; implementation begins. E5 first upstreaming PR — `amd/comgr/` directory shell for the cross-family entry point, stub-only, reusing the boundary pattern from `comgr-hotswap-b0a0.cpp`. *Milestone: first GPT-OSS GEMM executes correctly gfx1250 → gfx942.*
- **Week 5 (May 15–22):** E1 MODREP predicate-chain hardening continues from the already-landed O1 loud-refuse classifier / WaveNative default evidence (`canary_bpermute_scan_fp32`, `swiglu_fp32`, `rmsnorm_fp32`, `corpus_layernorm_fp32` match or fail loud in the current default path). E2 I8 shape. E4 atomic-scope round-trip + named-barrier refusal gate. E3 TDM cross-target skeleton IR under unit test. E5 begins translation-core relocation to `amd/comgr/` (MC state + OpcodeMap + ISAProfile — the pieces with no ROCR dependencies). *Milestone: GEMM + Attention correct on gfx942 (matches leadership §5.2 W5).*
- **Week 6 (May 22–29):** E1 P6 (`ds_swizzle`) lift (closes `sum_bitmatrix_rows`). E2 F8/F6/F4 shapes in progress. E3 TDM cross-target lands for the VIMAGE family. E6 full GPT-OSS Triton 15 canary sweep clean. E5 handler relocation continues. *Milestone: GPT-OSS forward-pass inference runs end-to-end on gfx942.*
- **Week 7 (May 29–Jun 5):** E5 `TranslationReport` generalisation lands (per `target-capability-dispatch-investigation.md §5`). E4 ABI hidden-arg surface. E2 MXFP scaled path in progress. E6 first hipBLASLt 147 subset sweep published. E5 second upstreaming PR — translation core in `amd/comgr/` with COMGR-internal unit tests mirroring `HotswapElfTest` / `HotswapMCTest`. *Milestone: hipBLASLt 147 baseline; failing kernels catalogued.*
- **Week 8 (Jun 5–12):** E3 TDM cross-target executes on Gluon MXFP GEMM. E2 MXFP scaled path lands. E4 ABI embedded V# refusal gate. E5 third upstreaming PR — ROCR loader calls the COMGR hotswap API instead of Salmon directly; `HSA_HOTSWAP_IR_RAISER=1` routes through COMGR. *Milestone: Gluon MXFP GEMM correct gfx1250 → gfx942.*
- **Week 9 (Jun 12–19):** E3 TDM cross-target executes on the FLAT `global_load_async_to_lds` family (FA K/V-cache tiling). E6 AITER regression follow-up (`test_jit_dir_with_enum` salmon-only hang). E5 in-process `llc` prototype — replaces the `std::system` subprocess call so the COMGR entry point has no fork/exec dependency. *Milestone: Gluon MXFP FA correct gfx1250 → gfx942.*
- **Week 10 (Jun 19–26):** E2 remaining shapes (F6/F4 scaled). E4 reject-gate pass over every sync SemOp; every unregistered SemOp fails loud. E6 closes remaining triage items (currently Softmax reduction and Triton sort/topk residuals; Matmul128 MSB is already fixed). E5 upstreaming PRs under review; respond to code-owner feedback. *Milestone: every gfx1250 WMMA/SWMMAC shape in the corpus either lowered or rejected loudly.*
- **Week 11 (Jun 26–Jul 3):** E5 in-process `llc` production integration lands. E6 full AITER corpus re-run. E4 reject-gate pass over every ABI SemOp. E5 upstreaming PRs merged; the `hotswap/transpiler/` tree in `rocm-systems` becomes a thin shim that invokes the COMGR API. *Milestone: zero silent miscompiles in the size-sweep across GPT-OSS Triton 15, hipBLASLt 147, Gluon 4, `test_data_gfx1250/`; COMGR-hosted translation passes the same test matrix.*
- **Week 12 (Jul 3–10):** Hardening, reproducibility, documentation pass, per-axis failing-kernel catalogue. *Milestone: demo-ready.*
- **Week 13 (Jul 10–15):** Final validation and delivery.

### 4.3 Upstreaming scope

Leadership's §5.3 flags a 2-week upstreaming cost for workstream D
between weeks 3 and 4, describing it as "Port lifter code into
comgr-hotswap module". The underlying obligation is real and this
document carries it, but the scope differs from leadership's framing
in one specific way, which allows it to be spread across the schedule
rather than executed as a single pre-axis-work port:

Leadership's §5.3 lumps two distinct steps into the 2-week estimate:

1. **Fork-prototype → mainline port.** Moving the raising prototype out
   of a fork and into a review-friendly shape on LLVM mainline.
2. **ROCR → COMGR boundary integration.** Placing the translation
   engine behind the `amd_comgr_hotswap_*` API surface that
   workstream A is establishing, so ROCR stays thin.

Step (1) does not apply to this document: the raising pipeline is in
`hotswap/transpiler/` in `rocm-systems` mainline today, builds against
the stock ROCm LLVM build tree (`hotswap/transpiler/README.md`), and
ships with a GoogleTest suite. Step (2) does apply and is load-bearing
— the COMGR boundary is what lets Salmon ship in a ROCm release and
what keeps ROCR's surface area unchanged. Without it, Salmon remains a
research fixture in the ROCR tree rather than a product deliverable.

The upstreaming scope for workstream D therefore comprises:

- **COMGR API surface.** A cross-family entry point
  (`amd_comgr_hotswap_retarget_cross_family` or equivalent) that
  mirrors the shape of workstream A's
  `amd_comgr_hotswap_rewrite`. Design in concert with
  @jacob-lambert (COMGR code owner) in W3; first stub PR in W4.
- **Directory relocation.** Moving the translation core from
  `rocm-systems/.../hotswap/transpiler/` into
  `ROCm/llvm-project:amd/comgr/`, alongside Harsh's
  `comgr-hotswap-*.cpp` files. Done in stages (MC layer first in W5,
  handlers + orchestrator in W6, ROCR loader in W8) so each PR is
  independently reviewable.
- **ROCR loader refactor.** `HSA_HOTSWAP_IR_RAISER=1` is redirected
  from calling `raiseToIR` in-process to calling the COMGR hotswap
  API; the remaining `rocm-systems/.../hotswap/transpiler/` tree
  becomes a thin invocation shim.
- **In-process `llc`.** Replaces the `std::system("llc …")`
  subprocess call so the COMGR entry point has no fork/exec
  dependency. Scheduled W9 (E5).
- **LLVM-upstream dependencies.** Salmon reaches a small number of
  target-private interfaces (`MCSubtargetInfo` feature queries, the
  `GET_AVAILABLE_OPCODE_CHECKER`-gated helpers documented in
  `target-capability-dispatch-investigation.md §6`, and
  `wmma_lowering.cpp`'s pre-backend pass over WMMA intrinsics not yet
  selectable upstream). Each is either already public or pulled into
  a Salmon-owned TU with zero upstream changes; no patches to
  `llvm/`-proper are on the critical path for July 15. The WMMA
  lowering is a post-July-15 upstream concern.

Spread across W3–W11 of E5, this is approximately 5–6 engineer-weeks
of dedicated upstreaming work, interleaved with the axis workstreams
rather than blocking them. The net schedule effect — compared to
leadership's §5.3 2-week upfront port — is that first GPT-OSS GEMM
correctness lands at W4 (matching leadership's Schedule A W3 milestone
one week later, and one week ahead of leadership's Schedule B W4), but
the COMGR-hosted version of that same milestone only lands at W11 when
the relocation PRs merge.

### 4.4 Advantages

These are leadership's §5.4 advantages, unchanged:

- Operates on LLVM IR: CFG, SSA, types, and liveness are available
  without reconstruction.
- Reuses LLVM backend passes: instruction selection, register
  allocation, scheduling, waitcnt insertion, hazard recognition.
- No per-instruction translation rules in the translation engine
  proper; target-specific lowering is the backend's responsibility.
- Better scalability to future target pairs: a new target is a new
  backend triple, not a new rule table.
- Re-lowered code is native compilation for the target, not transpiled
  code with emulation artefacts.

Plus two state-specific points that differ from leadership's §5
framing:

- **Per-axis design already published.** Every non-trivial translation
  decision in E1–E4 has a written spec with its own refusal criteria
  (`wave-size-translation.md`, `matrix-translation.md`,
  `sync-translation.md`, `abi-translation.md`,
  `modrep-predicate-chain.md`, `target-capability-dispatch.md`).
  Schedule slippage therefore manifests as design-doc-scheduled work
  not landing on time, not as a design decision pending.
- **Fail-loud is the default.** `buildObstructionReport` (today) and
  `TranslationReport` (E5 W7) refuse any kernel the decision procedure
  cannot classify. "Silent miscompile" converts to "loud refusal"
  before a kernel reaches the translation engine.

Upstreaming is not eliminated by this document (§4.3); the net
schedule effect is that correctness milestones land one week earlier
than leadership's Schedule B and the COMGR-hosted version of those
milestones lands at W11.

### 4.5 Risks

Leadership's §5.5 risks, with status updated from the latest checked-in
evidence:

1. **Binary-to-IR lifting is lossy — incorrect output for complex kernels.** *Partially mitigated.* SPE + WaveNative projections landed 2026-04-21, and the MODREP predicate-chain class now fails loud or matches in the default evidence. Current correctness residuals are more specific: Softmax reduction, Triton `tl.sort` N=16, and streaming-topk merge / index-set divergence.
2. **Upstreaming to COMGR takes longer than budgeted.** *Open, re-scoped.* Not a fork-to-mainline port (the translation engine is in mainline today) but a ROCR-to-COMGR boundary move, spread across W3–W11 of E5 (~5–6 engineer-weeks; §4.3). Risks: (a) COMGR code-owner review turnaround stalls one of the staged PRs; (b) directory relocation surfaces a dependency that does not transplant cleanly (e.g. a ROCR-side header that Salmon reaches through `loader/`); (c) in-process `llc` integration uncovers LLVM API friction. Mitigations: W3 API-design engagement with @jacob-lambert; staged PRs that each land independently; axis-work (E1–E4) is decoupled from the COMGR boundary so it continues in parallel.
3. **Re-lowered code does not match original register allocation or scheduling.** *Unchanged from leadership.* The July-15 goal is functional correctness, not performance parity; this becomes a post-July-15 concern.
4. **LLVM backend waitcnt/hazard passes conflict with lifted IR.** *Mitigated.* Waitcnts lift as no-ops (`handle_sopp.cpp:94`); backend re-derives for the target. Residual cases are specified in `sync-translation.md §5`.
5. **Lifter does not handle all gfx1250 instruction families.** *Bounded by derisking.* `gpt-oss-derisking.md` establishes that GPT-OSS Triton 15 has zero unrewritable C1/C3 sites and no `v_permlane64`; remaining gaps are TDM cross-target (E3, W3–W11) and the MXFP scaled path (E2, W3–W10). Both are scheduled.
6. **No equivalent of the 42-kernel validation.** *Addressed.* BatchRaise + Corpus tests run over the 170-kernel corpus catalogued by `hotswap/kerneldex/`. The size-sweep harness (E5 W3) extends this beyond a single input per kernel.

Schedule-specific open risks:

7. **MODREP predicate-chain class no longer silently miscompiles the named default-path recipes.** O1 loud-refuse landed for MODREP, and WaveNative is the default for the checked-in evidence. Keep the risk open as a classifier/regression surface, but the current failures have moved to Softmax, Triton sort/topk residuals, and other per-axis gaps.
8. **TDM cross-target is the largest open design item.** If E3 slips past W8, Gluon MXFP GEMM and FA K/V-cache tiling milestones slip correspondingly. Mitigation: W3 TDM design proposal gates the implementation; if the proposal does not resolve by W4, the path escalates per §7.2.
9. **Triage items consume E6 bandwidth in W3–W5.** `softmax_gfx1250` remains a current XFAIL, now reaching launch and producing invalid `+inf` output. `Matmul128x128_1tile` is historical: the family graduated after the V_CMP → V_CNDMASK per-lane-i1 shadow fix in `learnings.md` (2026-04-21).

---

## 5. Side-by-side: current state of both cross-family prototypes

This section is the May-1-gate evidence package. §5.1 and §5.2 started
as the April 22 size-sweep snapshot; the D-column below has been
updated where newer checked-in evidence supersedes that snapshot. Cells
marked *TBD* still mean the harness has not produced comparable
workstream-B data.

### 5.1 Canary matrix — one row per obstruction-class canary

Each canary is a hand-authored kernel that exercises one failure mode.
Three cell states: **pass** (output matches reference), **loud-fail**
(path refuses with a classified diagnostic), **silent wrong** (path
runs to completion and emits incorrect output).

| Canary | Axis / class | D — IR raising | B — binary patching |
|---|---|---|---|
| `canary_bpermute_scan_fp32` | C2 `ds_bpermute` / C5 predicate chain | match under default WaveNative; loud-refused under MODREP narrow-O1 | TBD (harness W3) |
| `canary_dpp_reduce_fp32` | C2 DPP | pass | TBD |
| `canary_dpp_compound_add_fp32` | C2 DPP | pass | TBD |
| `canary_permlanex16_rowmax_fp32` | C2 permlane | pass | TBD |
| `corpus_softmax_fp32` | reduction + cross-widening safety net | current notes report loud failure / wrong-output variants; needs fresh sweep before promoting a single status | TBD |
| `corpus_layernorm_fp32` | reduction + WaveNative | match in the current MODREP-predicate evidence; older meeting sweep had a layernorm hang | TBD |
| `mubuf_store_b32.hip` | MUBUF V# path | probe added for Softmax triage; MUBUF store path no longer the leading Softmax suspect | TBD |
| `s_set_vgpr_msb.hip` | gfx1250 VGPR-MSB bank | historical probe; Matmul128 family fixed by V_CMP/V_CNDMASK mask-shadow work | TBD |
| `v_cmpx_gt_i32.hip` | C4 `v_cmpx` | pass | TBD |
| `s_and_saveexec_b32.hip` | C4 saveexec | pass (Triton bounds-check form) | TBD |
| `wmma_f32_16x16x4_gemm.hip` | matrix axis | pass (baseline F16 shape) | TBD |
| `v_bfi_b32.hip` | bitfield-insert | pass | TBD |
| `v_cmp_cndmask_sgpr.hip` | VALU compare + SGPR cndmask | pass | TBD |

### 5.2 Corpus sweep — per-kernel, shape-varying

The corpus sweep varies the input-shape parameters each kernel's
driver exposes (M/N/K for GEMM, sequence length for attention, hidden
dimension for softmax / layernorm) and records per-cell pass /
loud-fail / silent-wrong for both paths. Corpora included:

| Corpus | Count | Source |
|---|---:|---|
| GPT-OSS Triton (North Star) | 15 | `scope_discovery/kernels/`, audited in `gpt-oss-derisking.md` |
| hipBLASLt Tensile GEMMs | 147 | `tensilelite/` |
| `test_data_gfx1250/` smoke tests | 4 | pre-existing gfx1250 transpiler probes |
| Gluon reference invocations | 4 captured + 3 reasoned | `triton-lang/triton/third_party/amd/python/examples/gluon/`; `gpt-oss-derisking.md §8` |

The April 22 state (pre-harness) for workstream D was bounded by the
three `Gfx1250Gpu` families then known green. Current evidence expands
that to include `Matmul128x128_*`; `Gfx1250Gpu.Softmax` remains the
headline gtest XFAIL.
Workstream B has no corresponding `Gfx1250Gpu`-style cross-family
execution test on record. The E5 harness (W3) closes both gaps.

### 5.3 Open items per path

**Workstream D (IR raising):**

- MODREP predicate-chain class: O1 loud-refuse landed for MODREP and WaveNative is default; keep regression coverage, but this is no longer the named four-recipe silent-miscompile blocker.
- `Gfx1250Gpu.Softmax` reduction failure. Triage authored; MUBUF probe checked in; current xfail reaches launch and produces invalid `+inf` output.
- Triton sort/topk residuals: `canary_tl_sort_fp32_n16`, `topk_forward_bf16`, and `topk_forward_bisect_m2_strict` remain open in `learnings.md`.
- AITER `test_jit_dir_with_enum` salmon-only hang on `fmoe_stage1_bf16_pertokenFp8_g1u1_32x64_4tg_pf3` (2242 instructions). Scheduled E6 W9.

**Workstream B (binary patching, legacy):**

- Size-variance coverage unknown pending E5 harness runs against `hotswap/transpiler.cpp`.
- Port to mainline behind COMGR (leadership's §3.1 workstream B focus) not observed in a public branch as of April 22.
- LLVM MC state bug (one code object per process) documented in the program-level design proposal (`hotswap-design-proposal.md §3c`) as a HIGH risk for the batch fallback path.

### 5.4 Dimensional comparison

| Dimension | D — IR raising (workstream D) | B — binary patching (workstream B, legacy) |
|---|---|---|
| Current location | `hotswap/transpiler/` (mainline) | `hotswap/transpiler.cpp` (mainline) |
| Translation core LOC | ~20k (handler-decomposed; 921 LOC orchestrator `raiser.cpp`) | 5,780 (monolithic `transpiler.cpp`) |
| Cross-family execution tests green (Apr 22) | `Vecadd`, `Matmul64x64_*`, `WmmaProbe_*` gfx1250 → gfx942 | 42/42 branch unit tests (Mar 25); cross-family execution suite not in tree |
| Last cross-family substantive commit | continuous through Apr 22 | `50519f7ecc`, Mar 25 |
| Same-family Tier 1 dependency | shares workstream A COMGR infra | shares workstream A COMGR infra |
| Per-axis design specs | 7 in `hotswap/docs/` | none published |
| EXEC divergence handling | SPE + WaveNative projection (`wave-size-translation.md`) | explicit text patterns in `transpiler.cpp` |
| Scaling to N target pairs | O(source + target) | O(source × target) |
| LLVM MC global-state bug | not affected (uses `llc` subprocess; in-process migration is E5 W9) | blocks batch fallback per `hotswap-design-proposal.md §3c` |
| Shape-sweep coverage | E5 W3 deliverable; §5.1/§5.2 populated | E5 W3 deliverable; §5.1/§5.2 populated |

---

## 6. Risk summary

This section supplements leadership's §6 side-by-side risk table with
an updated status for the raising-path rows, as of April 22:

| Leadership row | Status Apr 22 |
|---|---|
| §6 "Correctness confidence — whole-program lifting, harder to debug" | Partially mitigated. Per-axis design specs localise each lowering decision; the obstruction classifier runs per-kernel before translation. |
| §6 "Transpilation artefacts in trace" | Out of scope (trace goal dropped per §1). |
| §6 "Scalability to future targets — backend target triples" | Unchanged. This remains the central architectural advantage. |
| §5.5 risk #1 "lifting is lossy" | Partially mitigated by SPE + WaveNative and MODREP loud-refuse coverage. Current residuals are `Gfx1250Gpu.Softmax`, `canary_tl_sort_fp32_n16`, and streaming-topk strict recipes; see `learnings.md` / `xfail.cmake`. |
| §5.5 risk #2 "upstreaming > 2 weeks" | Open, re-scoped. Not a fork-to-mainline port but a ROCR-to-COMGR boundary move spread across W3–W11 of E5 (~5–6 engineer-weeks); see §4.3 and §4.5 #2. |
| §5.5 risk #3 "re-lowered scheduling differs" | Unchanged; July-15 goal is correctness, not performance parity. |
| §5.5 risk #4 "backend waitcnt conflict" | Mitigated; waitcnts lift as no-ops. |
| §5.5 risk #5 "instruction family coverage" | Bounded by `gpt-oss-derisking.md` + E2/E3 schedule. |
| §5.5 risk #6 "no 42-kernel validation" | Addressed; BatchRaise + Corpus + 5 GPU test suites + E5 size-sweep. |

For the corresponding Schedule A picture, refer to leadership's §4.4
and §6 without modification; no workstream-B-specific observations
from this document are added.

---

## 7. What If the Chosen Path Hits a Wall

### 7.1 Principle: do not switch paths mid-project

This mirrors leadership's §7.1 verbatim. Switching cross-family paths
after May 1 costs two to three weeks of retooling and discards work.
The July-15 deadline does not have that slack. Instead:

1. Scope-reduce before path-switching. Deliver GPT-OSS GEMM correctness only if full inference is blocked.
2. Reject and document kernels that the chosen path cannot handle, rather than building heroic workarounds.
3. Accept a reduced corpus coverage target (GPT-OSS Triton 15 alone, deferring hipBLASLt 147) rather than slipping the date.

### 7.2 Escalation triggers

Calendar aligned with leadership's §7.2; thresholds adapted to this
document's milestones.

| When | Trigger | Action |
|---|---|---|
| Week 4 | First GPT-OSS GEMM not executing correctly after P5 lands | Pull 1–2 engineers from E3 / E4 into E1 for targeted triage |
| Week 5 | GEMM + Attention not both green | All-hands triage; identify root cause before adding people |
| Week 7 | GPT-OSS forward pass still failing in W6 carries into W7 | Scope-reduce: hipBLASLt 147 subset deferred, GPT-OSS Triton 15 only |
| Week 8 | First COMGR-hosted PR (W4 stub) still not reviewed / merged | Escalate to @harsh-menon. If blocked >1 week, E5 re-scopes: land Salmon's COMGR-facing API as a ROCR-side shim against a declared COMGR interface, so downstream work unblocks while upstream review continues. |
| Week 9 | TDM cross-target not executing on Gluon MXFP GEMM | Scope-reduce: MXFP corpus dropped from §1 goal 3; coverage limited to F16 Gluon |
| Week 11 | Size-sweep shows residual silent miscompiles, OR upstreaming PRs not merged | For silent miscompiles: accept a reduced "zero silent miscompiles in GPT-OSS Triton 15" scope in §1 goal 3; do not slip July 15. For upstreaming: July 15 deliverable is demonstrated from the `rocm-systems` tree; COMGR-hosted version lands post-July-15. |
| Any week | Fundamental blocker in the lifter | Add to the reject list. The legacy branch becomes a post-July-15 plan, not a mid-project switch. |

### 7.3 The value of the other prototype

Regardless of which cross-family path is chosen at May 1, the other
prototype's work product is not wasted:

- **If IR raising is chosen:** the legacy byte-level transpiler
  (`hotswap/transpiler.cpp`, 5,780 LOC, 42/42 branch unit tests)
  remains a known-working fallback and a reference for instruction-
  family coverage. It can be reactivated if the lifter encounters a
  kernel class it cannot handle, per leadership's §7.3.
- **If binary patching is chosen:** this document's workstream
  establishes the feasibility and cost of an alternative for future
  target pairs. The per-axis design specs, the 170-kernel corpus
  classification, and the size-sweep harness (E5) remain useful
  independent of which engine produces target-ISA binaries.

Both prototypes reduce long-term risk even if only one is used for the
July-15 deliverable.

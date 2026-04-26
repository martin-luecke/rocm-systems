# Target-capability dispatch — LLVM-reuse investigation

> **Status:** investigation, recommendation, API sketch. Resolves the
> §6 open question in `target-capability-dispatch.md`. No code has
> been changed; the implementation is tracked in §7 below as a
> T-series task breakdown.
>
> **Scope:** how should `ISAProfile::supports(SemOp)` decide whether
> a target ISA natively supports a given source SemOp, for all four
> axes listed in `target-capability-dispatch.md §4` (matrix, TDM,
> sync, ABI)?

---

## 1. The question, restated

Every non-trivial translation branches on a target-capability check
(`target-capability-dispatch.md §1`). Today that check reads one
hand-picked `ISAProfile` field per axis (`hasWMMA12`, `hasTDM`,
`hasSplitBarriers`, `hasGFX12UserSGPRLayout`, …). As we add axes the
list grows monotonically, and every new AMDGPU generation forces us
to re-audit which bit gates which SemOp.

We want a single decision procedure:

```cpp
bool ISAProfile::supports(SemOp op) const;
```

whose answer is sourced from LLVM's per-instruction `Predicates`
list (the same data that the assembler / disassembler / matcher
already use) rather than a transpiler-side parallel table that
shadows it. The question is which LLVM surface to hook into.

## 2. What LLVM gives us, mapped to each option

### 2.1 `MCSubtargetInfo` public surface (the floor)

`llvm/include/llvm/MC/MCSubtargetInfo.h` exposes only the two queries
`ISAProfile::fromSubtarget` uses today:

```cpp
bool hasFeature(unsigned Feature) const;
const FeatureBitset &getFeatureBits() const;
```

`FeatureBitset` (in `llvm/TargetParser/SubtargetFeature.h`) is a
`constexpr` `std::bitset` reimplementation with bitwise AND / OR /
XOR and element access. It supports exactly the operation the
subset-intersection option needs:

```cpp
bool isSubsetOf(const FeatureBitset &Required, const FeatureBitset &Have) {
  return (Required & Have) == Required;
}
```

There is no public accessor that takes an `MCInst` / `MCInstrDesc`
/ opcode and returns its required `FeatureBitset`. `MCInstrDesc`
carries `TSFlags`, `SchedClass`, operand info, and implicit-def /
implicit-use lists, but **no** feature-bit dependency. `MCInstrInfo`
and `MCInstrAnalysis` likewise have no per-opcode feature query.

> Evidence: grepping the MC headers for `Feature`, `Predicate`, or
> `FeatureBitset` against `llvm/include/llvm/MC/*.h` surfaces no
> hit at the `MCInstrDesc` / `MCInstrInfo` / `MCInstrAnalysis`
> level; the `FeatureBitset` type only appears on
> `MCSubtargetInfo`.

### 2.2 Per-target `AMDGPUBaseInfo` helpers

`llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.{h,cpp}` exposes a
growing family of subtarget predicates — `isGFX12Plus(STI)`,
`hasMAIInsts(STI)`, `hasVOPD(STI)`, `isGFX1250(STI)`, etc. These
are the style of helpers `ISAProfile::fromSubtarget` already uses
(see `AMDGPU::hasMAIInsts(STI)` in `isa_profile.hpp`). They answer
"does this subtarget have feature family X?" — not "does this
subtarget natively implement SemOp S?" For any SemOp whose
requirement is a single generation-level bit (e.g. TDM
= `FeatureGFX1250Insts`) these are sufficient; for anything finer
they fall into the same hand-picked bucket as option 1 below.

### 2.3 Generated assembler / disassembler tables

`AMDGPUGenAsmMatcher.inc` carries a `static const MatchEntry
MatchTable0[]` with `RequiredFeaturesIdx` columns into a
`static constexpr FeatureBitset FeatureBitsets[]`. Similarly
`AMDGPUGenDisassemblerTables.inc` ships `OPC_CheckPredicate` tables
with `static bool checkDecoderPredicate(unsigned Idx, const
FeatureBitset &FB)`. Both encode the same per-instruction predicate
information but are keyed off parser state (token sequences) /
decoder state (encoded bits), not off the AMDGPU opcode enum we
want to query. They are correct but not the right shape for a
keyed-on-opcode runtime query.

### 2.4 `InstrInfoEmitter::emitFeatureVerifier` (the critical find)

`llvm/utils/TableGen/InstrInfoEmitter.cpp`, function
`emitFeatureVerifier` (L694), already emits exactly the per-opcode
feature query we need. Its output into `AMDGPUGenInstrInfo.inc` is:

```cpp
// Around L167289+, guarded by GET_COMPUTE_FEATURES
namespace llvm::AMDGPU_MC {

inline FeatureBitset computeAvailableFeatures(const FeatureBitset &FB);

inline FeatureBitset computeRequiredFeatures(unsigned Opcode) {
  static constexpr FeatureBitset FeatureBitsets[] = { ... };
  static constexpr uint16_t RequiredFeaturesRefs[] = { ... };
  assert(Opcode < <NumOpcodes>);
  return FeatureBitsets[RequiredFeaturesRefs[Opcode]];
}

} // namespace llvm::AMDGPU_MC

// Around L224575+, guarded separately by GET_AVAILABLE_OPCODE_CHECKER
namespace llvm::AMDGPU_MC {

bool isOpcodeAvailable(unsigned Opcode, const FeatureBitset &Features) {
  FeatureBitset AvailableFeatures = computeAvailableFeatures(Features);
  FeatureBitset RequiredFeatures  = computeRequiredFeatures(Opcode);
  FeatureBitset MissingFeatures   =
      (AvailableFeatures & RequiredFeatures) ^ RequiredFeatures;
  return !MissingFeatures.any();
}

} // namespace llvm::AMDGPU_MC
```

The `FeatureBitsets[]` rows are derived from each
`Instruction`'s `Predicates` list in TableGen, i.e. the same data
that produced the `AssemblerPredicate` declarations referenced by
`AMDGPU.td` (e.g. `AssemblerPredicate<(all_of FeatureWMMA128bInsts)>`
for `HasWMMA128bInsts`, used in `Predicates = [HasWMMA128bInsts,
isGFX1250]` on every gfx1250 WMMA instruction def). `Inst->TheDef
->getValueAsListOfDefs("Predicates")` harvests them on L723 and
L776 of `InstrInfoEmitter.cpp`.

**This is exactly the runtime query we want.** What stops us using
it today is how it is gated:

```cpp
// AMDGPUGenInstrInfo.inc, L169318
#if (defined(ENABLE_INSTR_PREDICATE_VERIFIER) && !defined(NDEBUG)) || \
    defined(GET_AVAILABLE_OPCODE_CHECKER)
#define GET_COMPUTE_FEATURES
#endif
```

`AMDGPUMCTargetDesc.cpp` defines `ENABLE_INSTR_PREDICATE_VERIFIER`
before including the `.inc`, so in a **debug** LLVM build (with
`!NDEBUG`) these functions are compiled into `AMDGPUMCTargetDesc.o`
— but with `inline` linkage, inside `llvm::AMDGPU_MC`, and the only
non-inline symbol exposed out of that TU is
`verifyInstructionPredicates` (which wraps them and
`report_fatal_error`s on mismatch). In **release** LLVM
(`NDEBUG` set) the whole block is gone because
`GET_AVAILABLE_OPCODE_CHECKER` is never defined anywhere in tree.

So the query *function* exists in-emitter, but LLVM ships no
public symbol we can link against.

## 3. Option evaluation

### 3.1 Option 1 — hand-maintained `SemOp → FeatureBit` table

**Shape.**

```cpp
static constexpr llvm::AMDGPU::Feature kRequiredFeature[SemOp_COUNT] = {
    [SemOp::V_WMMA_F32_16x16x32_F16] = AMDGPU::FeatureWMMA128bInsts,
    [SemOp::TENSOR_LOAD_TO_LDS]      = AMDGPU::FeatureGFX1250Insts,
    [SemOp::S_BARRIER_WAIT]          = AMDGPU::FeatureGFX12Insts,
    // …
};
```

**What LLVM gives us.** Nothing at this level — `FeatureBit`s only.
We pair them by hand with SemOps.

**What extension is needed.** None in LLVM. One `constexpr` array
in Salmon; one table entry per new SemOp.

**Maintenance cost.** One row per SemOp; per-ISA reasoning required
("what feature flag should gate this?") and re-reasoning whenever
LLVM renames a feature bit. Today that is ~250 SemOps and 10+
feature bits across the four axes.

**Cross-generation robustness.** Poor. gfx1300 will rename /
reshape some feature bits; every such rename is a transpiler
patch. A SemOp whose requirement is a disjunction (e.g. "FP8 conv
intrinsics exist on gfx940 *or* gfx1250") becomes a nested list
with custom scalar — i.e. we drift toward option 3 as pressure
increases.

**Where it breaks.** Two specific patterns we already have in the
axis docs:

1. `GLOBAL_ATOMIC_PK_ADD_F16` is gated by
   `FeatureAtomicGlobalPkAddF16Insts` on gfx942 and
   `FeatureAtomicBufferGlobalPkAddF16Insts` on gfx12+. A scalar
   `FeatureBit` column cannot express this; the table has to store
   a `SmallVector<FeatureBit>` and a reducer op per row.
2. `V_MFMA_F32_16x16x128_F8F6F4` is gated by the conjunction
   (`FeatureFP8ConversionInsts` ∧ `FeatureGFX950Insts`). Same
   problem — one-column is insufficient.

Both of those would work with a richer table (option 3), but then
we are back to hand-maintenance with no LLVM-side data source.

### 3.2 Option 2(a) — TableGen backend extension

**Shape.** Add a new TableGen backend or extend `InstrInfoEmitter`
to emit `AMDGPUGenSemOpSupport.inc` keyed off either the MC opcode
enum or a Salmon-side `SemOp` name.

**What LLVM gives us.** All the primitives already exist
(`Inst->TheDef->getValueAsListOfDefs("Predicates")`,
`SubtargetFeatureInfo::getAll(Records)`, `FeatureBitset` emission).

**What extension is needed.** A custom backend alongside
`InstrInfoEmitter.cpp`, or a third preprocessor gate added to the
existing emitter. Either approach means upstreaming an LLVM patch
(`llvm/utils/TableGen/`) and waiting for it to land in the LLVM
drop Salmon builds against.

**Maintenance cost.** In LLVM: small and one-time. In Salmon:
nothing after the LLVM-side patch lands.

**Cross-generation robustness.** Best possible — identical to
option 2(b), since both hit the same `Predicates` data.

**Where it breaks.** Upstream friction. Project Salmon cannot merge
on LLVM's cadence; blocking our progress on an LLVM patch is a
scheduling risk. Worse, the right API (keyed on Salmon's
internal `SemOp` enum) is not upstream's concern, so the
upstream-facing extension has to be keyed on MC opcode — which
reduces it to exactly what option 2(b) gives us today with zero
LLVM changes.

### 3.3 Option 2(b) — lift the existing generated table into Salmon

**Shape.**

```cpp
// transpiler/isa_profile_query.cpp  (one TU, one inclusion)
#define GET_AVAILABLE_OPCODE_CHECKER
#include "AMDGPUGenInstrInfo.inc"
// → llvm::AMDGPU_MC::isOpcodeAvailable(unsigned, const FeatureBitset&)
//   and llvm::AMDGPU_MC::computeRequiredFeatures(unsigned) are now
//   defined in this TU.
```

Then `ISAProfile::supports(SemOp)` dispatches through the existing
`OpcodeMap`'s inverse (`SemOp → {MC opcodes}`):

```cpp
bool ISAProfile::supports(SemOp op) const {
  auto opcodes = opcMap_->opcodesFor(op);   // SmallVector<unsigned>
  if (opcodes.empty())
    llvm::report_fatal_error(
        "ISAProfile::supports: SemOp has no registered MC opcodes; "
        "OpcodeMap coverage is broken");
  for (unsigned opc : opcodes) {
    if (llvm::AMDGPU_MC::isOpcodeAvailable(opc, featureBits_))
      return true;
  }
  return false;
}
```

The inverse map is trivially built from the existing
`OpcodeMap::build`: we already iterate every AMDGPU opcode and
collapse it onto a canonical pseudo keyed on `SemOp`. The same
traversal populates a `DenseMap<SemOp, SmallVector<unsigned>>` for
the reverse direction. One-shot work at raiser init; O(1) per
query.

**What LLVM gives us.** `InstrInfoEmitter` emits
`computeAvailableFeatures`, `computeRequiredFeatures`,
`isOpcodeAvailable` already — full closed-form answer to our
question, parameterised on `FeatureBitset`. Only the gating macro
stops us using it today.

**What extension is needed.** None in LLVM. In Salmon: one new TU
that includes `AMDGPUGenInstrInfo.inc` with
`GET_AVAILABLE_OPCODE_CHECKER` defined, plus the inverse map
build-out in `OpcodeMap`.

**Maintenance cost.** Zero per-SemOp rows. The data source is the
same TableGen `Predicates` list the assembler and disassembler
already rely on. When LLVM renames / adds a feature bit, the
generated `.inc` tracks it automatically; Salmon rebuilds against
the new `.inc` and picks it up.

**Cross-generation robustness.** As strong as TableGen itself.
Whatever predicates `AMDGPU.td` declares for gfx1300's WMMA are
picked up with zero Salmon edits; the only Salmon-side change for
a new ISA is adding new `SemOp`s when they are needed (i.e. when
the new ISA introduces a primitive we did not previously lift).

**Where it breaks — three boundary conditions, all explicit.**

1. **MC-opcode coverage.** If `OpcodeMap` does not have a row for
   every MC opcode that means `SemOp::X`, `supports(X)` will miss
   some target-side encodings. This is already the failure mode of
   the existing startup coverage verifiers (`verifyExecAttrCoverage`,
   `verifyMFMACoverage`); the same startup pass extends to check
   that the inverse map is non-empty for every SemOp referenced in
   the handler registry.
2. **Pseudo vs. real opcode.** Several AMDGPU instructions have a
   target-agnostic pseudo whose `Predicates = []` and a
   subtarget-specific real (`_gfx1250`, `_gfx90a`, …) whose
   `Predicates` list is populated. `isOpcodeAvailable` on the
   pseudo says "available on every target" which is too permissive.
   The disjunction over `opcodesFor(op)` solves this by favoring
   the subtarget-specific real — if a real variant is available on
   the target's feature bits, the pseudo's permissiveness is moot.
   This also matches how the assembler / matcher resolve instruction
   dispatch today, so we are reusing the same contract.
3. **GET_AVAILABLE_OPCODE_CHECKER is not a documented ABI.** The
   gate is a private build-time knob used by `llvm-tblgen` tests;
   nothing stops LLVM from changing the emitter signature. We
   mitigate by pinning one TU that includes the `.inc`, compiling
   it as part of CI in both debug and release LLVM configurations,
   and keeping a small golden-file test
   (`isOpcodeAvailable(V_WMMA_F32_16X16X32_F16_w32_twoaddr, gfx950.features)
   == false` and `== true` for gfx1250.features) so that an
   upstream rename breaks the build, not the runtime behavior.

### 3.4 Option 3 — feature-set intersection (hand-maintained minimal `FeatureBitset`)

**Shape.**

```cpp
struct SemOpRequirement { FeatureBitset required; };
static const SemOpRequirement kReq[SemOp_COUNT] = {
    [SemOp::V_WMMA_F32_16x16x32_F16] = { bits(FeatureWMMA128bInsts) },
    // …
};
bool ISAProfile::supports(SemOp op) const {
  return (kReq[(size_t)op].required & featureBits_) == kReq[(size_t)op].required;
}
```

**What LLVM gives us.** `FeatureBitset` operators. We supply the
per-SemOp required bitset ourselves.

**What extension is needed.** None in LLVM. In Salmon: a table of
`FeatureBitset` literals, one row per SemOp.

**Maintenance cost.** Same per-SemOp-row burden as option 1, with
a richer cell type. Conjunctions and disjunctions are expressible
(conjunction = multi-bit row, disjunction = list of rows with a
`std::any_of` reducer). Still hand-maintained, still requires
per-ISA reasoning for each row.

**Cross-generation robustness.** Same weakness as option 1 — a
new ISA that changes the granularity of a feature bit forces a
Salmon edit. Better than option 1 in that renames are typically
bit-level and do not cascade.

**Where it breaks.** Exactly the same places as option 1, just
with a wider cell. Nothing structural is better; we just defer
when it breaks.

### 3.5 Option 4 — rejected

The "emit native unconditionally, catch backend errors, fall back
to decompose" posture violates the project rule that forbids
fallback solutions (the same rule that motivates
`target-capability-dispatch.md §1`'s refusal-or-native decision
procedure). No further evaluation; recording only that the
`isOpcodeAvailable` query we recommend is a **principled** check,
not a best-effort emit followed by recovery. A SemOp whose target
lacks both native and decompose paths fails loudly via
`RaiseFailure::unsupportedShape` at the handler site, which the
classifier in §5 below surfaces before any IR is produced.

## 4. Recommendation — option 2(b)

Use `AMDGPU_MC::isOpcodeAvailable` via one Salmon-owned TU that
includes `AMDGPUGenInstrInfo.inc` with
`GET_AVAILABLE_OPCODE_CHECKER` defined, plus the SemOp →
MC-opcodes inverse of `OpcodeMap`.

**Rationale.**

- **Single source of truth.** The TableGen `Predicates` list
  declared on each `Instruction` def is already the data the
  assembler and disassembler rely on. Our runtime query reads the
  same data, through the same emitter-generated function. There is
  no parallel table in Salmon that can drift.
- **Zero per-SemOp cost.** Adding a new SemOp to the transpiler is
  already guarded by `OpcodeMap` coverage + attribute registrations
  (`SemOpAttrs`). `supports(...)` gets it for free from the
  inverse map and the generated tables.
- **Zero LLVM-side cost.** The generator exists. We are only
  flipping a well-scoped preprocessor gate in a Salmon TU; no
  upstream patch, no schedule dependency.
- **Cross-generation robust.** gfx1250/1251/1300+ ship their
  predicate information in the same `.td` files Salmon builds
  against. The generated `.inc` is automatically up-to-date.
- **Fail-loud on gap.** The two failure modes we care about —
  (a) a SemOp with no MC opcodes and (b) a SemOp supported on
  neither native nor any decompose path — both produce loud
  refusals (§4.1's `report_fatal_error` contract and §5's
  `TranslationReport` refusal, respectively).

### 4.1 Proposed API on `ISAProfile`

```cpp
// transpiler/isa_profile.hpp
namespace transpiler {

class OpcodeMap;

struct ISAProfile {
  unsigned waveSize = 64;
  bool hasAGPR = false;
  bool hasMFMA = false;
  bool hasVOPD = false;
  bool hasScalarFP = false;
  bool hasWMMA12 = false;

  bool isWave32() const { return waveSize == 32; }

  // The raw feature bitset MC exposes. Retained so existing
  // AMDGPU::foo(STI)-style helpers keep working on a profile that
  // has outlived its MCSubtargetInfo.
  const llvm::FeatureBitset &featureBits() const { return featureBits_; }

  // Target-capability dispatch. Returns true iff the target ISA
  // natively implements the primitive denoted by `op`:
  //   ∃ mcOpc ∈ opcMap.opcodesFor(op) :
  //     llvm::AMDGPU_MC::isOpcodeAvailable(mcOpc, featureBits_)
  //
  // `opcMap` is the per-MCState opcode map already built by the
  // raiser (`raiser.cpp`). `ISAProfile` does not retain it, so the
  // query takes it as an argument; an overload that binds one at
  // construction time is provided below for callers that already
  // have a stable map.
  //
  // Refusal contract: if `opcMap.opcodesFor(op)` is empty, reports
  // fatal. An empty set means the coverage verifier
  // (verifySupportQueryCoverage, §6) missed a SemOp at startup,
  // which is a transpiler bug, not a user-input problem. We do
  // NOT silently return false — that would hide the bug and the
  // handler would take the decompose path for a SemOp we cannot
  // describe, producing wrong code.
  bool supports(SemOp op, const OpcodeMap &opcMap) const;

  static ISAProfile fromSubtarget(const llvm::MCSubtargetInfo &STI);

 private:
  ISAProfile() = default;
  llvm::FeatureBitset featureBits_;
};

} // namespace transpiler
```

**Convenience binding.** `raiser.cpp` stores one
`ISAProfile` per side (source / target) and one `OpcodeMap` per
`MCState`. To keep handler-call sites terse we pair them once:

```cpp
// transpiler/isa_profile_binder.hpp
struct BoundISAProfile {
  const ISAProfile &profile;
  const OpcodeMap &opcMap;

  bool supports(SemOp op) const { return profile.supports(op, opcMap); }
  unsigned waveSize()   const   { return profile.waveSize; }
  bool     hasMFMA()    const   { return profile.hasMFMA; }
  // ... delegates for every field ...
};
```

`RaiseContext::targetIsa` becomes a `BoundISAProfile` and
`if (ctx.targetIsa.hasWMMA12)` in the live WMMA branch stays
callable. New axes write `if (ctx.targetIsa.supports(sop))` — no
new `ISAProfile` field per axis.

**Data source recap.** `featureBits_` comes from
`MCSubtargetInfo::getFeatureBits()`. Per-SemOp required features
come from `llvm::AMDGPU_MC::computeRequiredFeatures(opc)`, which
reads the TableGen-emitted tables in `AMDGPUGenInstrInfo.inc`.

**Unregistered-SemOp failure mode.** Hard refusal via
`report_fatal_error`, not a default-true or default-false. A SemOp
that has no MC opcodes registered in `OpcodeMap` means we promoted
a source opcode into the SemOp stream without teaching the raiser
about it — exactly the class of coverage gap the `verify...Attrs`
startup checks exist to prevent. Reporting the gap at the first
`supports(...)` call puts the abort on the loading side of the
translation, before any IR is emitted.

## 5. Migration — generalise `buildObstructionReport` into a multi-axis `TranslationReport`

The wave-size axis has an existing pre-translation classifier
(`wave_size_obstruction.{hpp,cpp}`, invoked from `raiser.cpp:160`).
It walks the decoded instruction stream and produces an
`ObstructionReport`; if any site has `rewrite == None` or
`rewriteImplemented == false`, the classifier turns the report
into a `RaiseFailure` before any IR construction starts.

Every other axis today is implicit: the per-handler branch
(WMMA / TDM / split-barrier) discovers at IR-emission time that the
target lacks the capability and decomposes or refuses. For the
same reasons the wave-size axis benefits from pre-classification —
we prefer to fail a kernel loudly before we have half-emitted IR,
and we want one place to diagnose "this kernel exceeds our
translation capability" — we generalise the classifier to cover
capability coverage for every axis.

### 5.1 Classes and layout

Keep `wave_size_obstruction.{hpp,cpp}` as-is (the wave-size axis
has its own taxonomy that the principle-doc migration should not
disturb), and add a **new** adjacent module:

```
transpiler/translation_report.hpp      // public types
transpiler/translation_report.cpp      // builder
```

with types:

```cpp
enum class TranslationAxis : uint8_t {
  WaveSize = 0,      // delegates to buildObstructionReport
  Capability,        // per-SemOp supports(sop) + registered decompose
  // later: ExecFlow, Abi, ...
};

// One refusal reason per failing site. Axis-agnostic payload; each
// axis populates the fields it uses.
struct TranslationSite {
  TranslationAxis axis;
  const DecodedInst *inst = nullptr;   // where it was found
  SemOp sop = SemOp::Unknown;          // which SemOp triggered
  std::string detail;                  // human-readable
  // Axis-specific: for Capability, the missing feature bits
  llvm::FeatureBitset missingFeatures;
  // Axis-specific: for WaveSize, the ObstructionKind / RewriteId
  ObstructionKind waveKind = ObstructionKind::None;
  RewriteId waveRewrite = RewriteId::None;
  bool waveRewriteImplemented = false;
};

class TranslationReport {
 public:
  void addWaveSiteFrom(const ObstructionSite &site);
  void addCapabilitySite(const DecodedInst *inst, SemOp sop,
                         const llvm::FeatureBitset &missing);

  bool hasRefusal() const;           // any site forces a refusal
  RaiseFailure selectFailure() const; // picks the first refusal site
  std::string render(/*... srcIsa, tgtIsa, waveM, waveN ... */) const;

 private:
  llvm::SmallVector<TranslationSite> sites_;
};

TranslationReport buildTranslationReport(
    llvm::ArrayRef<DecodedInst> insts,
    const MCState &mc,
    const ISAProfile &src,
    const ISAProfile &tgt,
    const OpcodeMap &opcMap,
    const HandlerRegistry &handlers); // see §5.3
```

**Relationship to existing types.** `ObstructionSite` /
`ObstructionKind` / `RewriteId` / `ObstructionReport` stay intact
— they are the wave-size axis's taxonomy, and inlining them into
the generic axis type would bleach useful structure. The
`TranslationSite` is a superset shape; the wave-size builder
produces `ObstructionSite` as it does today, and
`addWaveSiteFrom` adapts. This is the "subclassing cleanly" route
from the original task description, implemented as composition
rather than inheritance (wave-size as a first-class axis, not a
subclass of a generic report), which matches the codebase's
`struct`-heavy style.

**Justification for keeping `ObstructionReport`.** Its predicates
(`hasUnrewritable`, `hasPendingRewrite`, `isOblivious`,
`firstUnrewritable`, `firstPending`) and render helpers rely on
the wave-size-specific enums; generalising the type would either
duplicate them per axis or weaken them to booleans. Keeping the
wave-size type intact and folding its sites into
`TranslationReport` loses no expressiveness at the principal gate
(`TranslationReport::hasRefusal()`) and preserves every existing
lit test that asserts on the wave-size trace format.

### 5.2 Inputs and outputs

- **Inputs:** `ArrayRef<DecodedInst>`, `MCState`, source
  `ISAProfile`, target `ISAProfile`, `OpcodeMap`, handler
  registry (see §5.3). Same structural shape as
  `buildObstructionReport`; we add the target-capability / handler
  inputs that the wave-size axis does not use.
- **Outputs:** one `TranslationReport` with zero or more sites.
  Same mapping from "sites" to "outcome" as the wave-size axis:
  empty = accept, any refusal-worthy site = refuse with the first
  failure's diagnostic.

### 5.3 Per-axis plug-in

Each axis registers a **site builder** with the classifier:

```cpp
struct AxisBuilder {
  void (*build)(const DecodedInst &,
                const ISAProfile &src, const ISAProfile &tgt,
                const OpcodeMap &opcMap,
                const HandlerRegistry &handlers,
                TranslationReport &report);
};
```

The wave-size axis provides one builder that internally calls
`buildObstructionReport` on the whole stream once (not per
instruction; its analysis is whole-kernel — co-occurrence of
`v_cmpx` and `v_mbcnt`). The capability axis provides a per-SemOp
builder:

```cpp
void buildCapabilitySite(const DecodedInst &inst,
                         const ISAProfile &src, const ISAProfile &tgt,
                         const OpcodeMap &opcMap,
                         const HandlerRegistry &handlers,
                         TranslationReport &report) {
  SemOp sop = opcMap.lookup(inst.opcode);
  if (tgt.supports(sop, opcMap))
    return;                                          // native path ok
  if (handlers.hasDecomposePath(sop, src, tgt))
    return;                                          // decompose ok
  // Neither: refuse loudly.
  llvm::FeatureBitset required =
      aggregateRequiredFeatures(sop, opcMap);         // from inverse map
  llvm::FeatureBitset missing  = required & ~tgt.featureBits();
  report.addCapabilitySite(&inst, sop, missing);
}
```

`HandlerRegistry::hasDecomposePath` is the axis-owned part: each
axis doc declares, at registration time, which SemOps it
decomposes (e.g. matrix axis declares
`V_WMMA_F32_16x16x32_F16` → Template A when `!tgt.hasWMMA12`).
The same registry drives the `verifyShapeCoverage` startup checks
mentioned in each axis doc §5.0.3, so the decompose-path table is
not a third parallel list.

### 5.4 Interaction with per-SemOp handler dispatch

**Decision: handlers keep branching.** The existing pattern in
`handle_valu_vop3p.cpp:157-170`

```cpp
if (ctx.targetIsa.hasWMMA12) { /* native */ }
else                         { /* decompose */ }
```

is correct and stays. It is generalised from `hasWMMA12` to
`ctx.targetIsa.supports(SemOp::V_WMMA_F32_16x16x32_F16)` (per §4.1
above), but the two-way branch inside the handler remains.

The reason is that the handler has to do something even on the
native branch (pick the right intrinsic, wire types, …) — it is
not merely a gate. The `TranslationReport` is therefore
**advisory + refusal-only**: it filters kernels that cannot be
translated at all (no native *and* no decompose) before any IR is
emitted. For kernels that pass the report, the handler still
selects native-vs-decompose using the same `supports(...)` call;
the report does not hand the handler a pre-computed decision
token.

This split keeps the classifier reusable across axes (its output
is the same for every axis — "refuse or proceed") without
requiring the handler to unpack an axis-specific decision bag, and
matches how the wave-size axis already works: the classifier
decides refusal, the per-handler code keeps its existing dispatch
logic.

### 5.5 Hook-up site

`raiser.cpp:160` currently calls `buildObstructionReport`. After
the migration it calls `buildTranslationReport` (which internally
delegates to the wave-size builder for backward compatibility). The
refusal branch widens from `report.hasUnrewritable() ||
report.hasPendingRewrite()` to `report.hasRefusal()`; the error
string still uses `RaiseFailure`. No other call-site changes.

## 6. Verification / coverage checks (startup)

To keep the option 2(b) path honest at the gap described in §3.3:

- `verifySupportQueryCoverage(MCII, opcMap)`: for every SemOp
  referenced in the handler registry (the union of
  `getHandlerSOP1Attrs` / `…VALU…Attrs` / per-axis decompose
  tables), assert `opcMap.opcodesFor(sop).empty() == false`.
- Golden-file test: one per gfx generation in the corpus (gfx942,
  gfx950, gfx1100, gfx1250) — assert `isOpcodeAvailable(pinned
  opcode, profile.featureBits())` for a hand-picked minimal
  expected-true / expected-false pair per target. Upstream
  `InstrInfoEmitter` signature churn fails the test at build
  time.
- Lit test: a synthetic kernel that emits a SemOp our handler
  registry does not decompose on the target. The
  `TranslationReport` should refuse with the
  `missingFeatures`-decorated diagnostic, not silently decompose
  or silently emit an invalid intrinsic.

## 7. Next-task breakdown

All LoC estimates include doc + test; they are upper bounds for
"do this in one sitting, review-ready".

### T1 — `OpcodeMap` inverse map

Add `DenseMap<SemOp, SmallVector<unsigned>>` to `OpcodeMap`;
populate it in the same pass as the forward map; expose
`opcodesFor(SemOp) -> ArrayRef<unsigned>`.

- Files: `opcode_map.hpp`, `opcode_map.cpp`, one unit test.
- LoC: ~60.

### T2 — Generated-feature-query TU

Add `transpiler/isa_profile_feature_query.cpp`:

```cpp
#define GET_AVAILABLE_OPCODE_CHECKER
#include "AMDGPUGenInstrInfo.inc"
```

…plus a thin wrapper header re-declaring the three public
functions (`computeRequiredFeatures`, `isOpcodeAvailable`) so the
rest of Salmon can call them without pulling in the `.inc`.

- Files: `isa_profile_feature_query.hpp` (~15 lines),
  `isa_profile_feature_query.cpp` (~5 lines), CMake wiring.
- LoC: ~40.

### T3 — `ISAProfile::supports` + `BoundISAProfile`

Implement the API sketched in §4.1. Convert `ctx.targetIsa` /
`ctx.sourceIsa` in `RaiseContext` from `ISAProfile` to
`BoundISAProfile`. Replace every existing `ctx.targetIsa.hasX`
field read with either a delegated accessor or a `supports(op)`
call if X is a per-SemOp query dressed as a generation check.

- Files: `isa_profile.{hpp,cpp}`, `isa_profile_binder.hpp`,
  `raiser.cpp`, lightly touched handler files.
- LoC: ~150 (most of it is handler call-site migration).
- Mechanical: the existing hand-picked fields
  (`hasWMMA12`, future `hasTDM`, …) stay for now; T3 only adds
  `supports(...)` alongside. Axis-by-axis removal of the hand-
  picked fields in favor of `supports(...)` is deferred to T6.

### T4 — `TranslationReport` scaffolding

New module `translation_report.{hpp,cpp}` with the types in §5.1,
the axis-builder plumbing in §5.3, and the raiser hook-up rewrite
described in §5.5. The wave-size axis builder is an adapter around
`buildObstructionReport`.

- Files: `translation_report.{hpp,cpp}`, `raiser.cpp`.
- LoC: ~200.

### T5 — Capability axis builder + handler registry

Implement `buildCapabilitySite` (§5.3), introduce
`HandlerRegistry::hasDecomposePath` keyed on
`(SemOp, srcIsa, tgtIsa)`, populate it from the existing per-axis
decompose tables the axis docs reference (matrix
`verifyMatrixShapeCoverage`, TDM §5.0.2 builder, sync §5.0.2
table, ABI §4.0.1 bindings).

- Files: `translation_report.cpp`, `handler_registry.{hpp,cpp}`,
  per-axis registrations touched.
- LoC: ~250.

### T6 — Axis migrations: remove hand-picked `ISAProfile` fields

Once T3–T5 are in, each axis replaces `ctx.targetIsa.hasX` reads
with `ctx.targetIsa.supports(SemOp::Y)`. The `ISAProfile` struct
shrinks back to only the fields that describe the target
**policy** rather than its capability (e.g. `waveSize`, because
waves size gates the whole SPE projection, not a specific
primitive). `hasWMMA12` / `hasMFMA` / future `hasTDM` / … are
removed.

- Files: one axis per PR — matrix first (it is the instance that
  landed; lowest-risk migration), then TDM, sync, ABI.
- LoC per axis: ~30–80.

### T7 — Verifiers + golden tests

`verifySupportQueryCoverage` + per-generation golden tests + lit
test for the capability-axis refusal diagnostic.

- Files: `verify.cpp` (or equivalent existing slot),
  `test/lit/capability_refusal.mir`, a handful of unit tests.
- LoC: ~120.

### Total

- Critical path (T1 → T3 → T4 → T5): ~650 LoC.
- Full migration through T6 + verifier hardening in T7: ~1000 LoC.

No axis extension can land on the new classifier until T5 is in;
individual handlers can start consuming `supports(...)` after T3.

## 8. Proof-of-concept sketch (option 2(b))

Minimal illustrative TU. No build plumbing shown; imports the same
set of headers `AMDGPUMCTargetDesc.cpp` already requires.

```cpp
// transpiler/isa_profile_feature_query.cpp
//
// Exposes the per-opcode feature query that InstrInfoEmitter
// generates for the AMDGPU target. We flip the GET_AVAILABLE_OPCODE_
// CHECKER gate on our side of the include, which causes the emitter
// to materialise isOpcodeAvailable + computeRequiredFeatures with
// inline linkage into this TU.
#include "AMDGPUSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/MC/MCSubtargetInfo.h"

namespace transpiler::featureq {
#define GET_AVAILABLE_OPCODE_CHECKER
#include "AMDGPUGenInstrInfo.inc"
} // namespace transpiler::featureq
```

```cpp
// transpiler/isa_profile_feature_query.hpp
#include "llvm/TargetParser/SubtargetFeature.h"
namespace transpiler::featureq {
bool isOpcodeAvailable(unsigned Opcode, const llvm::FeatureBitset &Features);
llvm::FeatureBitset computeRequiredFeatures(unsigned Opcode);
} // namespace transpiler::featureq
```

```cpp
// transpiler/isa_profile.cpp (extract)
bool ISAProfile::supports(SemOp op, const OpcodeMap &opcMap) const {
  llvm::ArrayRef<unsigned> opcodes = opcMap.opcodesFor(op);
  if (opcodes.empty())
    llvm::report_fatal_error(llvm::Twine(
        "ISAProfile::supports: SemOp ") + semOpName(op) +
        " has no MC opcodes registered; OpcodeMap coverage gap");
  for (unsigned opc : opcodes) {
    if (featureq::isOpcodeAvailable(opc, featureBits_))
      return true;
  }
  return false;
}
```

The matching TranslationReport builder fragment:

```cpp
void buildCapabilitySite(const DecodedInst &inst,
                         const ISAProfile &src, const ISAProfile &tgt,
                         const OpcodeMap &opcMap,
                         const HandlerRegistry &handlers,
                         TranslationReport &report) {
  SemOp sop = opcMap.lookup(inst.opcode);
  if (tgt.supports(sop, opcMap))
    return;
  if (handlers.hasDecomposePath(sop, src, tgt))
    return;
  llvm::FeatureBitset required;
  for (unsigned opc : opcMap.opcodesFor(sop))
    required |= featureq::computeRequiredFeatures(opc);
  llvm::FeatureBitset missing = required & ~tgt.featureBits();
  report.addCapabilitySite(&inst, sop, missing);
}
```

No LLVM-side changes; this is the full behavioural surface the
recommendation asks for.

## 9. Summary

| Option | LLVM data source reuse | Per-SemOp hand data | Cross-gen robust | LLVM-side change | Recommendation |
|---|---|---|---|---|---|
| 1 Hand table | Feature bit only | 1 field/row | weak | none | no |
| 2(a) TG backend ext | Full `Predicates` list | none | strong | **new backend / patch** | superseded by 2(b) |
| 2(b) Lift existing emitter output | Full `Predicates` list | none | strong | none | **yes** |
| 3 Feature-set intersection | Feature bits (manual) | FeatureBitset/row | medium | none | no |
| 4 Backend-error fallback | n/a | n/a | n/a | n/a | rejected (project rule) |

The recommendation is option 2(b): the data and the query both
already exist in LLVM; we have been paying the maintenance cost of
options 1/3 only because a preprocessor gate has been blocking the
free route. Flipping that gate in a Salmon-owned TU is all that is
needed to make the per-SemOp support check principled,
hand-data-free, and automatically up-to-date across AMDGPU
generations.

#ifndef HOTSWAP_TRANSPILER_ISA_PROFILE_HPP
#define HOTSWAP_TRANSPILER_ISA_PROFILE_HPP

#include "MCTargetDesc/AMDGPUMCTargetDesc.h" // AMDGPU::Feature* enum
#include "Utils/AMDGPUBaseInfo.h"            // AMDGPU::hasMAIInsts
#include "llvm/MC/MCSubtargetInfo.h"

namespace transpiler {

// Snapshot of the capability bits the raiser actually branches on. Every
// field is derived directly from the MC subtarget feature bits that TableGen
// emits, so adding a new AMDGPU generation does not require touching this
// struct; we just read the already-defined FeatureFoo bit.
//
// This is a pure value snapshot — the factory copies bits out of the
// MCSubtargetInfo and does not retain any reference to it. Callers must
// construct via `fromSubtarget`; there is intentionally no default ctor.
struct ISAProfile {
  unsigned waveSize = 64;
  bool hasAGPR = false;
  bool hasMFMA = false;
  bool hasVOPD = false;
  bool hasScalarFP = false;
  // True iff the subtarget exposes the gfx12-era WMMA instructions
  // (FeatureWMMA{128,256}bInsts). gfx11 WMMA is encoded via FeatureGFX11Insts
  // + VOP3P patterns and is not covered here; the only WMMA source we lift
  // today is gfx1250.
  bool hasWMMA12 = false;
  // True iff the subtarget exposes the gfx1250 TENSOR cnt unit
  // (FeatureGFX1250Insts gates the VIMAGE TENSOR pseudo-instructions
  // `tensor_load_to_lds_d{2,4}` and `tensor_store_from_lds_d{2,4}` —
  // see `isGFX125xOnly` in AMDGPU.td and the
  // `int_amdgcn_tensor_load_to_lds` /
  // `int_amdgcn_tensor_store_from_lds` intrinsics in
  // IntrinsicsAMDGPU.td:4213). The flag is consumed by `handleVIMAGE`
  // to discriminate between the same-target intrinsic-emit path and
  // the cross-target loud refusal: the gfx942 and earlier ISAs have
  // no equivalent hardware unit, so cross-target lifts must refuse.
  bool hasTensorOps = false;
  // gfx125 widens compute_pgm_rsrc2.USER_SGPR_COUNT from the older 5-bit
  // GFX6-GFX120 field to a 6-bit field. Keep this as an ABI property rather
  // than deriving it from a string at each use site.
  bool hasGfx125UserSgprCountField = false;
  // True iff the subtarget exposes the `v_cvt_{,pk_}f32_{fp8,bf8}` /
  // `v_cvt_{,pk_}f16_{fp8,bf8}` family — `FeatureFP8ConversionInsts` in
  // AMDGPU.td. Present on gfx9.4.0 (gfx942) and onward (gfx950, gfx1170,
  // gfx1200, gfx1250, …). The hardware reads one or two FP8 lanes out of
  // a packed VGPR and produces native f32 / f16 / v2f32 / v2f16 values.
  bool hasFp8ConversionInsts = false;
  // True iff the FP8 conversion hardware on this subtarget interprets its
  // 8-bit input lanes with the OCP semantics (`Float8E4M3FN` for FP8,
  // `Float8E5M2` for BF8) rather than the AMD-legacy FNUZ semantics
  // (`Float8E4M3FNUZ` / `Float8E5M2FNUZ`) used by gfx9.4.0 / gfx942.
  //
  // The distinction matters for **lifting** FP8 read-side conversions
  // (`V_CVT_{,PK_}F32_{FP8,BF8}`) to portable IR. Today the only target
  // generic LLVM (`APFloatBase::getArbitraryFPSemantics`,
  // `LegalizeDAG.cpp::CONVERT_FROM_ARBITRARY_FP`) handles is the OCP set;
  // FNUZ is not lowerable through `llvm.convert.from.arbitrary.fp` at
  // all. Lifting gfx942 reads must therefore stay on the
  // `llvm.amdgcn.cvt.f32.{fp8,bf8}` AMDGCN intrinsic, whose semantics
  // are target-defined and silently flip with the subtarget. Lifting
  // gfx950+ reads can use the portable IR shape, which the AMDGPU
  // backend custom-lowers back to the same hardware instructions on
  // capable targets (PR llvm/llvm-project#194144) and which the
  // generic SelectionDAG expansion handles as bit-twiddling on every
  // other target.
  //
  // Detection: `FeatureFP8ConversionInsts` minus the gfx9.4.0 footprint
  // (`FeatureFP8Insts && !FeatureGFX950Insts`). gfx9.4.0 is the only
  // FNUZ-FP8 generation in tree.
  bool hasOcpFp8 = false;

  bool isWave32() const { return waveSize == 32; }

  static ISAProfile fromSubtarget(const llvm::MCSubtargetInfo &STI) {
    ISAProfile p;
    p.waveSize = STI.hasFeature(llvm::AMDGPU::FeatureWavefrontSize32) ? 32 : 64;
    // AGPRs/MFMA share the mai-insts feature today; keep them as separate
    // fields so future divergence stays expressible without touching callers.
    p.hasMFMA = llvm::AMDGPU::hasMAIInsts(STI);
    p.hasAGPR = p.hasMFMA;
    p.hasVOPD = STI.hasFeature(llvm::AMDGPU::FeatureVOPDInsts);
    p.hasScalarFP = STI.hasFeature(llvm::AMDGPU::FeatureSALUFloatInsts);
    p.hasWMMA12 = STI.hasFeature(llvm::AMDGPU::FeatureWMMA128bInsts) ||
                  STI.hasFeature(llvm::AMDGPU::FeatureWMMA256bInsts);
    p.hasTensorOps = STI.hasFeature(llvm::AMDGPU::FeatureGFX1250Insts);
    p.hasGfx125UserSgprCountField = llvm::AMDGPU::isGFX1250Plus(STI);
    p.hasFp8ConversionInsts =
        STI.hasFeature(llvm::AMDGPU::FeatureFP8ConversionInsts);
    // gfx9.4.0 (gfx942) is the sole FNUZ-FP8 generation in tree:
    // `FeatureFP8Insts` is set on every gfx9.4.x but `FeatureGFX950Insts`
    // gates on / off as we cross gfx942 → gfx950, where the OCP FP8
    // formats took over. Anywhere else with the FP8 conversion family
    // (gfx1170, gfx1200, gfx1250, …) is OCP by default — those targets
    // never carry `FeatureFP8Insts` at all.
    const bool isGfx942Fnuz = STI.hasFeature(llvm::AMDGPU::FeatureFP8Insts) &&
                              !STI.hasFeature(llvm::AMDGPU::FeatureGFX950Insts);
    p.hasOcpFp8 = p.hasFp8ConversionInsts && !isGfx942Fnuz;
    return p;
  }

  // Test-only factory.  Constructs an `ISAProfile` with only the
  // `waveSize` dimension set (the other feature flags default to
  // `false`) so unit tests exercising wave-direction-gated code —
  // `WaveNativeProjection`'s ctor assertion, `emitLaneActiveBit`'s
  // source / target wave-width arithmetic, the
  // `providesFullWaveExecInvariant` contract —
  // don't have to stand up a full `MCSubtargetInfo` (which would
  // require pulling in the LLVM AMDGPU target init chain just to
  // read one bit).  Production code MUST use `fromSubtarget`:
  // hand-forging loses the cross-checks between feature flags
  // (e.g. `hasAGPR == hasMFMA`) that `fromSubtarget` derives from
  // the canonical subtarget feature definitions in LLVM's
  // AMDGPU.td.  The factory is named and scoped rather than a
  // public default ctor so `git grep forTesting` is the review
  // anchor, not `git grep 'ISAProfile()'` (which would also hit
  // the private default ctor declaration below and mask real
  // findings).
  static ISAProfile forTesting(unsigned waveSize) {
    ISAProfile p;
    p.waveSize = waveSize;
    return p;
  }

 private:
  ISAProfile() = default; // constructible only via fromSubtarget() /
                          // forTesting(), per the comments above.
};

} // namespace transpiler

#endif

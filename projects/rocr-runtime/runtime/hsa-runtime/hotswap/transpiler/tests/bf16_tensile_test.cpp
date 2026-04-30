// Raise+lower regression gate for the 8 main GEMM kernels in a Tensile
// gfx1250 BF16 small-tile tuning output, retargeted to gfx942.  CPU-only;
// no GPU dispatch.  Originally a capability probe — graduated to a hard
// gate once the Tensile UserArgs kernarg-pointer dispatcher stopped
// blocking the SMEM provenance check (kernarg loads now flow through the
// amdgcn_kernarg_segment_ptr ABI).

#include "test_common.hpp"

#include "../code_object_utils.hpp"
#include "../pipeline.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace {

constexpr const char *kBf16TensileCo =
    BF16_TENSILE_DATA_DIR "/TensileLibrary_gfx1250_BBS_BH_TDM_small.co";

// All 8 main GEMM kernels in the small-tile tuning set.  Picked from
// `llvm-readelf --syms` on the .co; the names are the canonical Tensile
// "long form" with every tunable spelled out.
const std::vector<std::string> &kernelLongNames() {
  static const std::vector<std::string> v = {
      "Cijk_Alik_Bljk_BBS_BH_UserArgs_MT32x32x32_MI16x16x1_SN_LDSB0_AFC0_AG0_"
      "AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_"
      "DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI1_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_"
      "ISA1250_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_"
      "LRVWn1_LWPMn1_MIAV1_MIWT1_1_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_"
      "NEPBS0_NLCA1_NLCB1_ONLL1_PGR0_PLR1_PKA0_SGROB0_SIA0_SS0_SPO0_SRVW0_"
      "SSO0_SVW8_SK0_SKFTR0_SKXCCM0_SGRO0_TDMI3_TIN0_TLDS1_TLDSMn1_ULSGRO0_"
      "USL1_UIOFGRO0_UPLRP0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_4_1",
      "Cijk_Alik_Bljk_BBS_BH_UserArgs_MT64x32x32_MI16x16x1_SN_LDSB0_AFC0_AG0_"
      "AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_"
      "DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI1_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_"
      "ISA1250_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_"
      "LRVWn1_LWPMn1_MIAV1_MIWT2_1_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_"
      "NEPBS0_NLCA1_NLCB1_ONLL1_PGR0_PLR1_PKA0_SGROB0_SIA0_SS0_SPO0_SRVW0_"
      "SSO0_SVW8_SK0_SKFTR0_SKXCCM0_SGRO0_TDMI3_TIN0_TLDS1_TLDSMn1_ULSGRO0_"
      "USL1_UIOFGRO0_UPLRP0_USFGRO0_VSn1_VWA2_VWB1_WSGRA0_WSGRB0_WS32_WG32_4_1",
      "Cijk_Alik_Bljk_BBS_BH_UserArgs_MT32x64x32_MI16x16x1_SN_LDSB0_AFC0_AG0_"
      "AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_"
      "DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI1_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_"
      "ISA1250_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_"
      "LRVWn1_LWPMn1_MIAV1_MIWT1_2_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_"
      "NEPBS0_NLCA1_NLCB1_ONLL1_PGR0_PLR1_PKA0_SGROB0_SIA0_SS0_SPO0_SRVW0_"
      "SSO0_SVW8_SK0_SKFTR0_SKXCCM0_SGRO0_TDMI3_TIN0_TLDS1_TLDSMn1_ULSGRO0_"
      "USL1_UIOFGRO0_UPLRP0_USFGRO0_VSn1_VWA1_VWB2_WSGRA0_WSGRB0_WS32_WG32_4_1",
      "Cijk_Alik_Bljk_BBS_BH_UserArgs_MT64x64x32_MI16x16x1_SN_LDSB0_AFC0_AG0_"
      "AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_"
      "DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI1_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_"
      "ISA1250_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_"
      "LRVWn1_LWPMn1_MIAV1_MIWT2_2_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_"
      "NEPBS0_NLCA1_NLCB1_ONLL1_PGR0_PLR1_PKA0_SGROB0_SIA0_SS0_SPO0_SRVW0_"
      "SSO0_SVW8_SK0_SKFTR0_SKXCCM0_SGRO0_TDMI3_TIN0_TLDS1_TLDSMn1_ULSGRO0_"
      "USL1_UIOFGRO0_UPLRP0_USFGRO0_VSn1_VWA2_VWB2_WSGRA0_WSGRB0_WS32_WG32_4_1",
      "Cijk_Alik_Bljk_BBS_BH_UserArgs_MT32x32x64_MI16x16x1_SN_LDSB0_AFC0_AG0_"
      "AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_"
      "DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI1_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_"
      "ISA1250_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_"
      "LRVWn1_LWPMn1_MIAV1_MIWT1_1_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_"
      "NEPBS0_NLCA1_NLCB1_ONLL1_PGR0_PLR1_PKA0_SGROB0_SIA0_SS0_SPO0_SRVW0_"
      "SSO0_SVW8_SK0_SKFTR0_SKXCCM0_SGRO0_TDMI3_TIN0_TLDS1_TLDSMn1_ULSGRO0_"
      "USL1_UIOFGRO0_UPLRP0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_4_1",
      "Cijk_Alik_Bljk_BBS_BH_UserArgs_MT64x32x64_MI16x16x1_SN_LDSB0_AFC0_AG0_"
      "AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_"
      "DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI1_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_"
      "ISA1250_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_"
      "LRVWn1_LWPMn1_MIAV1_MIWT2_1_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_"
      "NEPBS0_NLCA1_NLCB1_ONLL1_PGR0_PLR1_PKA0_SGROB0_SIA0_SS0_SPO0_SRVW0_"
      "SSO0_SVW8_SK0_SKFTR0_SKXCCM0_SGRO0_TDMI3_TIN0_TLDS1_TLDSMn1_ULSGRO0_"
      "USL1_UIOFGRO0_UPLRP0_USFGRO0_VSn1_VWA2_VWB1_WSGRA0_WSGRB0_WS32_WG32_4_1",
      "Cijk_Alik_Bljk_BBS_BH_UserArgs_MT32x64x64_MI16x16x1_SN_LDSB0_AFC0_AG0_"
      "AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_"
      "DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI1_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_"
      "ISA1250_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_"
      "LRVWn1_LWPMn1_MIAV1_MIWT1_2_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_"
      "NEPBS0_NLCA1_NLCB1_ONLL1_PGR0_PLR1_PKA0_SGROB0_SIA0_SS0_SPO0_SRVW0_"
      "SSO0_SVW8_SK0_SKFTR0_SKXCCM0_SGRO0_TDMI3_TIN0_TLDS1_TLDSMn1_ULSGRO0_"
      "USL1_UIOFGRO0_UPLRP0_USFGRO0_VSn1_VWA1_VWB2_WSGRA0_WSGRB0_WS32_WG32_4_1",
      "Cijk_Alik_Bljk_BBS_BH_UserArgs_MT64x64x64_MI16x16x1_SN_LDSB0_AFC0_AG0_"
      "AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_"
      "DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI1_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_"
      "ISA1250_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_"
      "LRVWn1_LWPMn1_MIAV1_MIWT2_2_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_"
      "NEPBS0_NLCA1_NLCB1_ONLL1_PGR0_PLR1_PKA0_SGROB0_SIA0_SS0_SPO0_SRVW0_"
      "SSO0_SVW8_SK0_SKFTR0_SKXCCM0_SGRO0_TDMI3_TIN0_TLDS1_TLDSMn1_ULSGRO0_"
      "USL1_UIOFGRO0_UPLRP0_USFGRO0_VSn1_VWA2_VWB2_WSGRA0_WSGRB0_WS32_WG32_4_1",
  };
  return v;
}

} // namespace

// Iterate every main GEMM kernel in the .co and raise+lower it.  Every
// kernel must raise cleanly with all instructions lifted; any regression
// fails the test.
TEST(Bf16Tensile, RaiseSmallTuning) {
  if (!fileExists(kBf16TensileCo))
    GTEST_SKIP() << "Tensile .co not found: " << kBf16TensileCo;

  auto data = transpiler::readFile(kBf16TensileCo);
  ASSERT_FALSE(data.empty()) << "Failed to read " << kBf16TensileCo;

  auto names = transpiler::listKernelNames(data);
  printf("Kernels in .co: %zu\n", names.size());

  int success = 0, fail = 0;
  for (auto &k : kernelLongNames()) {
    printf("---- %s\n", k.c_str());
    auto r = transpiler::runPipeline(data, "gfx1250", "gfx942", k);
    if (r.success) {
      success++;
      printf("  RAISE OK   lifted=%d/%d hsaco=%zu bytes\n",
             r.liftedCount, r.totalCount, r.hsaco.size());
      EXPECT_EQ(r.liftedCount, r.totalCount) << "partial lift in " << k;
      EXPECT_GT(r.hsaco.size(), 0u) << "empty HSACO for " << k;
    } else {
      fail++;
      printf("  RAISE FAIL mnemonic=%s format=%s reason=%s detail=%s\n",
             r.failMnemonic.c_str(), r.failFormat.c_str(),
             r.failReason.c_str(), r.failDetail.c_str());
      ADD_FAILURE() << k << ": " << r.failReason << " on " << r.failMnemonic
                    << " (" << r.failFormat << ")";
    }
  }
  printf("\nBf16Tensile.RaiseSmallTuning: %d ok / %d fail\n", success, fail);

  EXPECT_EQ(fail, 0);
  EXPECT_EQ(static_cast<size_t>(success), kernelLongNames().size());
}

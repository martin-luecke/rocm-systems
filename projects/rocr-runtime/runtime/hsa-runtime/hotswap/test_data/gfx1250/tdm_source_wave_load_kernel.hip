// Source for tdm_source_wave_load_gfx1250.hsaco — dispatchable gfx1250
// fixture that proves TDM descriptors are source-wave-local under
// wave32 -> wave64 lifting.
//
// Flow:
//
//   * A 64-thread workgroup is two source wave32s on gfx1250.
//   * Each source wave computes its scalar wave id from TTMP8 and selects
//     descriptor A/B in SGPRs before issuing one `tensor_load_to_lds`.
//     Descriptor A points at input buffer A and LDS byte 0; descriptor B
//     points at input buffer B and LDS byte 128.
//   * After a workgroup barrier, all 64 threads copy their LDS slot to
//     `out[threadIdx.x]`.
//
// A target-wave-global helper miscompiles the upper source wave after
// lifting to gfx942: lane 32 starts X at 32 and stripes by 64, so a
// 32-element tile performs no upper-half copy. The source-wave-local
// helper computes local lane 0 for target lane 32 and stripes by 32.
//
// Regen recipe (from repo root):
//
//   DD=projects/rocr-runtime/runtime/hsa-runtime/hotswap/test_data/gfx1250
//   hipcc --offload-arch=gfx1250 --genco \
//     "$DD/tdm_source_wave_load_kernel.hip" \
//     -o "$DD/tdm_source_wave_load_gfx1250.hsaco.bundle"
//   /opt/rocm-7.2.1/lib/llvm/bin/clang-offload-bundler --type=o \
//     --targets=hipv4-amdgcn-amd-amdhsa--gfx1250 \
//     --input="$DD/tdm_source_wave_load_gfx1250.hsaco.bundle" \
//     --output="$DD/tdm_source_wave_load_gfx1250.hsaco" --unbundle
//   rm -f "$DD/tdm_source_wave_load_gfx1250.hsaco.bundle"
//
//   python3 - "$DD/tdm_source_wave_load_gfx1250.hsaco" <<'PY'
//   import sys
//   p = sys.argv[1]
//   b = bytearray(open(p, "rb").read())
//   LOAD_TAG = bytes.fromhex("0100 71D0".replace(" ", ""))
//   ZERO_DW = bytes(4)
//   PATCH_DW = bytes.fromhex("0000 007C".replace(" ", ""))
//   off = 0
//   n = 0
//   while True:
//     off = b.find(LOAD_TAG + ZERO_DW, off)
//     if off < 0:
//       break
//     b[off + 4:off + 8] = PATCH_DW
//     off += 12
//     n += 1
//   assert n >= 1, f"expected tensor_load_to_lds op, patched {n}"
//   print(f"patched {n} tensor_load_to_lds op(s)")
//   open(p, "wb").write(b)
//   PY

#include <hip/hip_runtime.h>
#include <cstdint>

typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));

struct TDMDescriptor {
  uint32_t g0[4];
  uint32_t g1[8];
  uint32_t g2[4];
  uint32_t g3[4];
};

__device__ __forceinline__ v4i load_v4(const uint32_t *p) {
  v4i v;
  v[0] = p[0];
  v[1] = p[1];
  v[2] = p[2];
  v[3] = p[3];
  return v;
}

__device__ __forceinline__ v8i load_v8(const uint32_t *p) {
  v8i v;
  v[0] = p[0];
  v[1] = p[1];
  v[2] = p[2];
  v[3] = p[3];
  v[4] = p[4];
  v[5] = p[5];
  v[6] = p[6];
  v[7] = p[7];
  return v;
}

extern "C" __global__ __launch_bounds__(64) void
tdm_source_wave_load_kernel(const TDMDescriptor *__restrict__ lower_desc,
                            const TDMDescriptor *__restrict__ upper_desc,
                            uint32_t *__restrict__ out) {
  __shared__ uint32_t lds[64];
  int tid = threadIdx.x;

  lds[tid] = 0xdeadbeefu;
  __syncthreads();

  uint32_t wave_id = 0;
  asm volatile("s_bfe_u32 %0, ttmp8, 0x50019" : "=s"(wave_id));
  const TDMDescriptor *desc = (wave_id & 1u) ? upper_desc : lower_desc;

  __builtin_amdgcn_tensor_load_to_lds(
      load_v4(desc->g0), load_v8(desc->g1), load_v4(desc->g2),
      load_v4(desc->g3), 0);
  __builtin_amdgcn_s_wait_tensorcnt(0);

  __syncthreads();
  out[tid] = lds[tid];
}

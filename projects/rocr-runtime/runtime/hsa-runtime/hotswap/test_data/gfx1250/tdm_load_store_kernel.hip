// Source for tdm_load_store_gfx1250.hsaco — dispatchable gfx1250 fixture
// that exercises BOTH TDM directions back-to-back in a single kernel,
// sharing one LDS staging buffer. Companion to `tdm_load_kernel.hip`
// (load-only) and `tdm_store_kernel.hip` (store-only); use this one
// to pin the composition contract — the load walker and the store
// walker must agree on element order, strides, OOB gating, and LDS
// advancement, and the kernel must fence LDS cross-wave visibility
// between the two tensor ops.
//
// Flow:
//
//   1. All lanes poison `lds[tid] = 0xdeadbeef` and barrier. The
//      poison keeps the compiler from eliding the `__shared__`
//      allocation (without a user-visible reader/writer the DS block
//      would be dropped and the TDM intrinsics — which reach LDS
//      only through the D#'s `lds_byte_addr`, not an IR argument —
//      would target an unreserved region). It also makes a silently
//      skipped TDM lift surface as `0xdeadbeef` bytes in `out`
//      rather than whatever stale LDS happened to contain.
//   2. The kernel reads the four input D# groups from host memory
//      and issues `__builtin_amdgcn_tensor_load_to_lds(in_g0..in_g3,
//      cpol=0)`. On gfx1250 this maps to the `_d4` form (4 SReg
//      operands + immediate cpol). The lifted IR call resolves to
//      `salmon_tdm_load_to_lds` under the gfx1250→gfx942 TDM
//      emulation path — the walker stripes `buffer_load_b32` reads
//      from the descriptor's `global_addr` into LDS at
//      `g0[1]=lds_byte_addr`.
//   3. `s_wait_tensorcnt(0)` fences the tensor load, then
//      `__syncthreads()` publishes the LDS writes across the
//      workgroup. The sync is load-bearing: the test harness
//      dispatches 128 threads (2 waves on gfx942 wave64, 4 waves on
//      gfx1250 wave32). Each wave independently walks the SAME
//      descriptor — on the lifted gfx942 path the writes are
//      idempotent (same data to same LDS offsets), but the store
//      side still must not launch until every wave's load has
//      committed, or a per-wave store can race a sibling wave's
//      in-flight load against the same LDS slot. A missing barrier
//      here is the exact bug this fixture catches vs. the
//      single-direction coverage tests in `TdmDescriptorCoverage`.
//   4. The kernel reads the four output D# groups and issues
//      `__builtin_amdgcn_tensor_store_from_lds(out_g0..out_g3,
//      cpol=0)`. The walker stripes `buffer_store_b32` writes from
//      LDS (at `g0[1]`) to the output descriptor's `global_addr`.
//   5. Final `s_wait_tensorcnt(0)` fences the store before the
//      host's `hipDeviceSynchronize` sees the launch as complete.
//
// Kernel ABI: 8 scalar pointer kernargs, input descriptor groups
// first, output descriptor groups second. The test harness passes
// `&d_in_desc->g0`, `&d_in_desc->g1`, … so each argument is a
// pointer to the group bytes (`v4i` for 128-bit groups,
// `v8i` for the 256-bit group 1). Kernel symbol is
// `tdm_load_store_kernel` (visible via `nm -D
// tdm_load_store_gfx1250.hsaco`).
//
// Regen recipe (from repo root). Two phases:
//
//   Phase 1 — hipcc + unbundle (same as every other gfx1250 fixture
//   in this directory; see `README.md`):
//
//     DD=projects/rocr-runtime/runtime/hsa-runtime/hotswap/test_data/gfx1250
//     hipcc --offload-arch=gfx1250 --genco \
//       "$DD/tdm_load_store_kernel.hip" \
//       -o "$DD/tdm_load_store_gfx1250.hsaco.bundle"
//     /opt/rocm-7.2.1/lib/llvm/bin/clang-offload-bundler --type=o \
//       --targets=hipv4-amdgcn-amd-amdhsa--gfx1250 \
//       --input="$DD/tdm_load_store_gfx1250.hsaco.bundle" \
//       --output="$DD/tdm_load_store_gfx1250.hsaco" --unbundle
//     rm -f "$DD/tdm_load_store_gfx1250.hsaco.bundle"
//
//   Phase 2 — byte-patch the VIMAGE TENSOR NULL sentinel. ROCm
//   7.2.1's hipcc writes the sentinel into byte 4 (bits 39-32,
//   `vdata` slot) of the 12-byte encoding; salmon's linked LLVM 23
//   disassembler looks for it in byte 7 (bits 63-56, `vaddr4`
//   slot). Without the patch, salmon's MCDisassembler refuses the
//   middle dword as `v_illegal` at instruction+4, the raiser
//   reports `Unsupported instruction: v_illegal`, and the lift
//   aborts. The sibling `tdm_load_kernel.hsaco` and
//   `tdm_store_kernel.hsaco` were generated with the same patch —
//   this is the documented workaround from
//   `lit_tests/tensor_load_to_lds_d2/tensor_load_to_lds_d2.hip`
//   (same disassembler disagreement, different workaround there
//   because those lit fixtures use raw `.long` encodings instead
//   of the hipcc builtin).
//
//   The patch rewrites dword1 of each TENSOR op from
//   `0x00000000` to `0x7C000000` (adds the NULL sentinel in
//   byte 7). The dword0 tag `D0 71 00 01` (LOAD) or
//   `D0 71 40 01` (STORE) anchors the match — these 4-byte
//   signatures are unique to the VIMAGE TENSOR encoding on
//   gfx1250 and do not collide with any other instruction in
//   a hipcc-compiled kernel:
//
//     python3 - "$DD/tdm_load_store_gfx1250.hsaco" <<'PY'
//     import sys
//     p = sys.argv[1]
//     b = bytearray(open(p,"rb").read())
//     LOAD_TAG  = bytes.fromhex("0100 71D0".replace(" ",""))  # 0xD0710001 LE
//     STORE_TAG = bytes.fromhex("0140 71D0".replace(" ",""))  # 0xD0714001 LE
//     ZERO_DW   = bytes(4)
//     PATCH_DW  = bytes.fromhex("0000 007C".replace(" ",""))  # 0x7C000000 LE
//     def patch(tag, name):
//       off = 0; n = 0
//       while True:
//         off = b.find(tag + ZERO_DW, off)
//         if off < 0: break
//         b[off+4:off+8] = PATCH_DW
//         off += 12; n += 1
//       assert n >= 1, f"{name}: no hipcc TENSOR encoding to patch (already patched, or the build is wrong)"
//       print(f"patched {n} {name} op(s)")
//     patch(LOAD_TAG,  "tensor_load_to_lds")
//     patch(STORE_TAG, "tensor_store_from_lds")
//     open(p,"wb").write(b)
//     PY

#include <hip/hip_runtime.h>
#include <cstdint>

typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));

// 256 slots = max tile_total the harness uses (tdm_gpu_test.cpp
// allocates 256-element in/out buffers). The D#.lds_byte_addr is 0
// in both load and store descriptors so the load writes from byte 0
// and the store reads from byte 0; the `__shared__` array lives at
// offset 0 by default.
#define LDS_SLOTS 256

extern "C" __global__ __launch_bounds__(128) void
tdm_load_store_kernel(const v4i *__restrict__ in_g0,
                      const v8i *__restrict__ in_g1,
                      const v4i *__restrict__ in_g2,
                      const v4i *__restrict__ in_g3,
                      const v4i *__restrict__ out_g0,
                      const v8i *__restrict__ out_g1,
                      const v4i *__restrict__ out_g2,
                      const v4i *__restrict__ out_g3) {
  __shared__ uint32_t lds[LDS_SLOTS];
  int tid = threadIdx.x;

  // Poison LDS before the TDM load. See step (1) in the file header
  // for the allocation-forcing rationale; this also produces a
  // legible `0xdeadbeef` sentinel in `out` for a silently-skipped
  // TDM lift, distinguishable from the hardware-OOB zero-fill the
  // walker produces on bounds violation.
  //
  // The `volatile addrspace(3)*` cast mirrors the idiom in
  // `tdm_store_kernel.hip`: the TDM intrinsics are declared
  // `IntrInaccessibleMemOrArgMemOnly` in `IntrinsicsAMDGPU.td`, so
  // LLVM's alias analysis treats LDS as untouched by them. A plain
  // `lds[tid] = poison` store has no alias-visible reader (the
  // subsequent TDM store reads LDS via the hidden descriptor path,
  // invisible to AA) and would be DCE'd — costing us both the
  // sentinel and the LDS allocation. Routing the poison write
  // through a volatile addrspace(3) pointer preserves it and keeps
  // the DS block reserved.
  volatile __attribute__((address_space(3))) uint32_t *lds_v =
      (volatile __attribute__((address_space(3))) uint32_t *)lds;
  if (tid < LDS_SLOTS) lds_v[tid] = 0xdeadbeefu;
  __syncthreads();

  v4i ig0 = *in_g0;
  v8i ig1 = *in_g1;
  v4i ig2 = *in_g2;
  v4i ig3 = *in_g3;
  __builtin_amdgcn_tensor_load_to_lds(ig0, ig1, ig2, ig3, 0);
  __builtin_amdgcn_s_wait_tensorcnt(0);
  __syncthreads();

  v4i og0 = *out_g0;
  v8i og1 = *out_g1;
  v4i og2 = *out_g2;
  v4i og3 = *out_g3;
  __builtin_amdgcn_tensor_store_from_lds(og0, og1, og2, og3, 0);
  __builtin_amdgcn_s_wait_tensorcnt(0);
}

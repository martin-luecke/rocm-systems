; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=global_load_async_to_lds_kernel 2>&1 | %FileCheck %s --check-prefix=STDERR
;
; Lift refusal test for FLAT `global_load_async_to_lds_b{8,32,64,128}`.
; Pins the contractual cross-target loud-failure behaviour of
; transpiler/handle_flat.cpp under the `GLOBAL_LOAD_ASYNC_TO_LDS_B*`
; family.
;
; The gfx1250 asynccnt unit (FLATInstructions.td:VFLAT 0x60-0x62
; reals; FLAT_Global_Load_LDS_Pseudo<IsAsync=1>) has no equivalent
; on gfx942. The matching LLVM intrinsics
; `int_amdgcn_global_load_async_to_lds_b{8,32,64,128}`
; (IntrinsicsAMDGPU.td:3939-3946, all sharing the
; `AMDGPUAsyncGlobalLoadToLDS` signature on line 3904) are gated
; by `FeatureGFX1250Insts` and emit the dedicated VFLAT 0x60-0x62
; encodings — no codegen path lowers them on a non-gfx1250
; backend. The user-rules forbid silent fallbacks; a synthesised
; `global_load + ds_write` pair would alter the wave's
; memory-ordering and asynccnt observable state, which the
; gfx1250 producers (tensilelite f8 / bf16 GEMMs, triton
; block-pipelined matmul kernels) rely on for software
; pipelining. The principled lift on a non-gfx1250 target IS the
; loud refusal.
;
; We assert two things:
;
;   1. The raiser exits non-zero (`%not` inverts the exit code — the
;      test passes only when raise_cli actually failed).
;   2. The stderr diagnostic from raise_cli names the offending
;      mnemonic (the first async load encountered in the kernel
;      stream is the b32) and the encoding format (`FLAT`).
;      raise_cli's failure-line format is fixed
;      (raise_cli.cpp:213): `kernel '<name>' failed to raise:
;      <mnemonic> [<format>] @offset=0x<offset> :: <detail>`.
;
; The handler also emits an explicit `transpiler: FLAT: ...`
; line that names the architectural mismatch and the
; same-target intrinsic; pinning that line keeps the diagnostic
; text from drifting into something less actionable for users
; who read raise_cli's stderr directly.

; STDERR: transpiler: FLAT: global_load_async_to_lds_b32
; STDERR-SAME: gfx1250 asynccnt unit
; STDERR-SAME: amdgcn.global.load.async.to.lds.b{8,32,64,128}
; STDERR-SAME: FeatureGFX1250Insts

; STDERR: raise_cli: kernel 'global_load_async_to_lds_kernel' failed to raise:
; STDERR-SAME: global_load_async_to_lds_b32
; STDERR-SAME: [FLAT]

; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx1250 --emit-ir=global_load_async_to_lds_kernel 2>&1 | %FileCheck %s --check-prefix=IR
;
; Lift fixture for FLAT `global_load_async_to_lds_b{32,64,128}` —
; the same-target (gfx1250 → gfx1250) intrinsic-emit path. Pins
; the principled lift in transpiler/handle_flat.cpp under
; `SemOp::GLOBAL_LOAD_ASYNC_TO_LDS_B{8,32,64,128}` when
; `ctx.targetIsa.hasTensorOps` is true. Companion fixture to
; `global_load_async_to_lds.ll`, which pins the cross-target
; (gfx942) loud refusal.
;
; The FLATInstructions.td `FLAT_Global_Load_LDS_Pseudo<IsAsync=1>`
; multiclass yields three operand-shape variants — plain
; (vdst:VGPR_32, vaddr:VGPR_64, off:imm, cpol:imm) and SADDR
; (vdst:VGPR_32, saddr:SGPR_64, vaddr:VGPR_32, off:imm, cpol:imm)
; for each width. The fixture's HIP source uses the clang
; builtins `__builtin_amdgcn_global_load_async_to_lds_b{32,64,128}`
; which compile to the plain VFLAT 0x60-0x62 reals (vdst=v1,
; vaddr=v0, saddr=s[0:1]/s[6:7]/s[4:5] — the kernel-arg pointers
; landed in the SGPR base + per-lane VGPR offset shape, which the
; AMDGPU disassembler prints as the SADDR variant).
;
; The matching LLVM intrinsic family
; (IntrinsicsAMDGPU.td:3939-3946, all sharing `AMDGPUAsyncGlobalLoadToLDS`
; on line 3904) is
;
;   void llvm.amdgcn.global.load.async.to.lds.b{8,32,64,128}(
;       ptr addrspace(1) %gaddr,
;       ptr addrspace(3) %laddr,
;       i32 immarg      %offset,
;       i32 immarg      %cpol)
;
; The handler:
;   * casts `vdst` (a per-lane VGPR_32 carrying the LDS i32 base)
;     via `inttoptr i32 %vgpr to ptr addrspace(3)` (named
;     `lds_ptr*` in the emitted IR);
;   * decodes the global address via the shared FLAT `decodeFlatAddr`
;     helper (plain → vaddr-only, SADDR → saddr+vaddr) into a
;     `ptr addrspace(1)`;
;   * threads the FLAT `offset` immediate and the `cpol` immediate
;     through as the trailing `i32 immarg` pair (cpol = 0x800 here
;     because the assembler emits `scale_offset` for the
;     scaled-saddr path);
;   * wraps the call in `ctx.emitUnderExec` so wave-divergent EXEC
;     is honoured before the side-effecting DMA.
;
; We pin the per-width call shape and the LDS-pointer cast it
; consumes. Drift indicators:
;   * If a future LLVM rename swaps the intrinsic name (e.g. drops
;     the `async.` infix) the IR check fails immediately and
;     pinpoints the rename rather than letting a silently mis-named
;     intrinsic reach the backend.
;   * If the LDS-base operand is lowered as a `<n x i32>` vector
;     (or any non-`ptr addrspace(3)` shape), the matching
;     `inttoptr i32 ... to ptr addrspace(3)` line goes missing and
;     FileCheck reports the exact divergence.

; b32: per-lane LDS i32 base via inttoptr i32 → ptr addrspace(3),
; then the b32 async DMA call.
; IR: %lds_ptr{{[0-9]*}} = inttoptr i32 {{.*}} to ptr addrspace(3)
; IR: call void @llvm.amdgcn.global.load.async.to.lds.b32(
; IR-SAME: ptr addrspace(1)
; IR-SAME: ptr addrspace(3) %lds_ptr
; IR-SAME: i32 0
; IR-SAME: i32 {{-?[0-9]+}}

; b64: same shape, b64 intrinsic.
; IR: %lds_ptr{{[0-9]*}} = inttoptr i32 {{.*}} to ptr addrspace(3)
; IR: call void @llvm.amdgcn.global.load.async.to.lds.b64(
; IR-SAME: ptr addrspace(1)
; IR-SAME: ptr addrspace(3) %lds_ptr
; IR-SAME: i32 0
; IR-SAME: i32 {{-?[0-9]+}}

; b128: same shape, b128 intrinsic.
; IR: %lds_ptr{{[0-9]*}} = inttoptr i32 {{.*}} to ptr addrspace(3)
; IR: call void @llvm.amdgcn.global.load.async.to.lds.b128(
; IR-SAME: ptr addrspace(1)
; IR-SAME: ptr addrspace(3) %lds_ptr
; IR-SAME: i32 0
; IR-SAME: i32 {{-?[0-9]+}}

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	global_load_async_to_lds_kernel
	.p2align	8
	.type	global_load_async_to_lds_kernel,@function
global_load_async_to_lds_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b128 s[4:7], s[0:1], 0x0
	v_lshl_add_u32 v1, v0, 2, 0x600
	s_wait_xcnt 0x0
	s_load_b64 s[0:1], s[0:1], 0x10
	s_wait_kmcnt 0x0
	global_load_async_to_lds_b32 v1, v0, s[4:5] scale_offset
	v_lshl_add_u32 v1, v0, 3, 0x400
	global_load_async_to_lds_b64 v1, v0, s[6:7] scale_offset
	v_lshlrev_b32_e32 v1, 4, v0
	global_load_async_to_lds_b128 v1, v0, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel global_load_async_to_lds_kernel
		.amdhsa_group_segment_fixed_size 1792
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 1792
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .max_flat_workgroup_size: 1024
    .name:           global_load_async_to_lds_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         global_load_async_to_lds_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata

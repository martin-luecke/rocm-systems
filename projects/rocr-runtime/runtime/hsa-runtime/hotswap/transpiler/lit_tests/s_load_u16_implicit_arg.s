; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=s_load_u16_implicit_arg_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Implicit-args reroute, narrow scalar load variant.
;
; Companion to `s_load_b96_kernarg.s` (which exercises the dword
; family). A source kernel that issues `s_load_u16 sN, kernarg_pair,
; off` with `off >= implicitArgsBase` is reading 2 bytes of a hidden
; arg through the source-ABI flat layout. The lifted kernel must
; reach those bytes via `amdgcn_implicitarg_ptr` (rebased to
; `off - implicitArgsBase`) so the target backend resolves the
; address against the *target* hidden-arg block, which lives at a
; different layout.
;
; Setup: one 8-byte explicit `by_value` arg at offset 0
; → `implicitArgsBase` = align_up_8(8) = 8. A load at offset 16
; therefore lands inside the implicit-arg block and triggers the
; reroute. Kernarg pair is s[0:1].

; CHECK-LABEL: define amdgpu_kernel void @s_load_u16_implicit_arg_kernel(
; CHECK: %implicitarg_ptr = call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
; CHECK: %impl_gep = getelementptr inbounds i8, ptr addrspace(4) %implicitarg_ptr, i64 8
; CHECK: %smem_load_h = load i16, ptr addrspace(4) %impl_gep, align 2
; CHECK-NEXT: %smem_load_zext = zext i16 %smem_load_h to i32

; A regression that bypasses the helper would read through a raw
; inttoptr of `s[0:1]` (addrspace(1)) and target the wrong bytes at
; runtime. The U16 vs I16 mix-up would silently corrupt every
; high-bit-set value.
; CHECK-NOT: load i16, ptr addrspace(1) %{{.*}}, align 2
; CHECK-NOT: sext i16 %smem_load_h to i32

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	s_load_u16_implicit_arg_kernel
	.p2align	8
	.type	s_load_u16_implicit_arg_kernel,@function
s_load_u16_implicit_arg_kernel:
	s_load_u16 s2, s[0:1], 0x10
	s_wait_kmcnt 0x0
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel s_load_u16_implicit_arg_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 3
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
      - { .offset: 0, .size: 8, .value_kind: by_value }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name: s_load_u16_implicit_arg_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 3
    .symbol: s_load_u16_implicit_arg_kernel.kd
    .vgpr_count: 1
    .wavefront_size: 32
amdhsa.target: amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata

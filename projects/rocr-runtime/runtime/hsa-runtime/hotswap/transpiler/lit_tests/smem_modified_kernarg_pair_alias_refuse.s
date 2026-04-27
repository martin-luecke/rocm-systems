; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=smem_modified_kernarg_pair_alias_refuse_kernel 2>&1 \
; RUN:   | %FileCheck %s
;
; Negative companion to `smem_modified_kernarg_pair_base.s`.
; Copying the entry kernarg pair to another SGPR pair preserves
; Kernarg-derived provenance on the alias, so a later full overwrite of
; s[0:1] from that alias must mark s[0:1] as Unknown rather than
; NonKernarg (the alias is not an "independent non-kernarg source").
; The downstream SMEM load through s[0:1] therefore lifts via
; `addrspace(1)` rather than the kernarg-derived `addrspace(4)`
; cast — the lift no longer refuses, it just declines the SMEM hint.

; CHECK-LABEL: define amdgpu_kernel void @smem_modified_kernarg_pair_alias_refuse_kernel(
; CHECK-SAME: [4 x i8] %kargs

; The entry kernarg pair is seeded from `amdgcn_kernarg_segment_ptr`.
; CHECK: call ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()

; The post-overwrite SMEM load lands on `addrspace(1)`, not
; `addrspace(4)`. Routing it through the kernarg cast would let the
; AMDGPU backend assume invariance against the (mutated) base, which
; is exactly what the alias-then-overwrite shape invalidates.
; CHECK: %smem_load = load i32, ptr addrspace(1) %{{[^,]+}}, align 4
; CHECK-NOT: %smem_load = load i32, ptr addrspace(4)

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	smem_modified_kernarg_pair_alias_refuse_kernel
	.p2align	8
	.type	smem_modified_kernarg_pair_alias_refuse_kernel,@function
smem_modified_kernarg_pair_alias_refuse_kernel:
; %bb.0:
	s_mov_b64 s[12:13], s[0:1]
	s_mov_b64 s[6:7], 0
	s_add_nc_u64 s[0:1], s[12:13], s[6:7]
	s_load_b32 s8, s[0:1], 0x0
	s_wait_kmcnt 0x0
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel smem_modified_kernarg_pair_alias_refuse_kernel
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 4
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 14
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
      - { .offset: 0, .size: 4, .value_kind: by_value }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 4
    .max_flat_workgroup_size: 1024
    .name: smem_modified_kernarg_pair_alias_refuse_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 14
    .symbol: smem_modified_kernarg_pair_alias_refuse_kernel.kd
    .vgpr_count: 1
    .wavefront_size: 32
amdhsa.target: amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata

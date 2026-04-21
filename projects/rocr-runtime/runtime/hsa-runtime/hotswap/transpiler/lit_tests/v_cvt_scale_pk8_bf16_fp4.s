; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --emit-ir=v_cvt_scale_pk8_bf16_fp4_kernel 2>/dev/null | %FileCheck %s
;
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_cvt_scale_pk8_bf16_fp4_kernel 2>&1 | %FileCheck --check-prefix=CROSS %s
;
; Pins the gfx1250-only VOP3 packed-8 FP4 -> BF16 scaled convert
; (V_CVT_SCALE_PK8_BF16_FP4_e64, VOP3Instructions.td:1788).
;
; Same-target lift (RUN #1): the handler emits the LLVM intrinsic
; `@llvm.amdgcn.cvt.scale.pk8.bf16.fp4(i32, i32, i32 immarg)` with
; immarg = 0 (the corpus-observed scale_sel value) and a
; <8 x bfloat> result; the write-back path bit-casts the <8 x bfloat>
; to i128 / four dwords before handing it to `writeRegVec`.  No
; cross-target refusal fires on the same-target path.
;
; Cross-target lift (RUN #2): the handler refuses loudly because
; `int_amdgcn_cvt_scale_pk8_bf16_fp4` is gated behind `isGFX125xOnly`
; in IntrinsicsAMDGPU.td:686 and gfx942 has no MX-FP4 scaling unit.
; The refusal message pins the ISA / intrinsic / isGFX125xOnly
; anchor so a future TableGen rename forces a visible test update.

; CHECK-LABEL: define amdgpu_kernel void @v_cvt_scale_pk8_bf16_fp4_kernel(

; The handler's exact emit shape:
;   %cvt_scale_pk8_bf16_fp4 = call <8 x bfloat> @llvm.amdgcn.cvt.scale.pk8.bf16.fp4(i32 %src, i32 %scale, i32 0)
; Pinned with explicit return-width + element-type + immarg 0.
; CHECK: %cvt_scale_pk8_bf16_fp4 = call <8 x bfloat> @llvm.amdgcn.cvt.scale.pk8.bf16.fp4(i32 %{{.*}}, i32 %{{.*}}, i32 0)

; The <8 x bfloat> result must be collapsed to a 128-bit int before
; `writeRegVec` splits it into four consecutive dword VGPRs.
; CHECK: bitcast <8 x bfloat> %cvt_scale_pk8_bf16_fp4 to i128

; The intrinsic declaration must carry the gfx1250 immarg range
; annotation (`range(i32 0, 16)`) and the exact bf16-fp4 name.
; CHECK-DAG: declare <8 x bfloat> @llvm.amdgcn.cvt.scale.pk8.bf16.fp4(i32, i32, i32 immarg range(i32 0, 16))

; Negative pins: no cross-target refusal on the same-target path,
; and no accidental fallback through the unscaled `cvt_pk_f32_fp8`
; / `cvt_f32_fp8` paths that share similar mnemonics.
; CHECK-NOT: cross-target lift to gfx942
; CHECK-NOT: amdgcn.cvt.pk.f32
; CHECK-NOT: amdgcn.cvt.f32.fp8

; CROSS: v_cvt_scale_pk8_bf16_fp4 is a gfx1250-only VOP3
; CROSS-SAME: isGFX125xOnly
; CROSS-SAME: no corpus kernel exercises today

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_cvt_scale_pk8_bf16_fp4_kernel
	.p2align	8
	.type	v_cvt_scale_pk8_bf16_fp4_kernel,@function
v_cvt_scale_pk8_bf16_fp4_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b64 s[0:1], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_dual_mov_b32 v4, 0 :: v_dual_mov_b32 v0, s0
	s_add_co_i32 s2, s0, 4
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)
	v_cvt_scale_pk8_bf16_fp4 v[0:3], v0, s2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v1, 0xffff, v0
	v_lshl_or_b32 v0, v0, 16, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v1, v0 :: v_dual_mov_b32 v2, v0
	v_mov_b32_e32 v3, v0
	global_store_b128 v4, v[0:3], s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_cvt_scale_pk8_bf16_fp4_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 5
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
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           v_cvt_scale_pk8_bf16_fp4_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     3
    .symbol:         v_cvt_scale_pk8_bf16_fp4_kernel.kd
    .vgpr_count:     5
    .wavefront_size: 32
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata

# TDM Translation — VIMAGE `tensor_load_to_lds` / `tensor_store_from_lds`

> **Status (2026-04-24): cross-target emulation landed.**
>
> The VIMAGE TENSOR descriptor-driven family
> (`tensor_load_to_lds_d{2,4}`, `tensor_store_from_lds_d{2,4}`, plus
> the `S_WAIT_TENSORCNT` wait counter) is now supported end-to-end
> on gfx1250 → gfx942 cross-target lifts via a HIP-authored
> runtime that emulates the TDM walk over gfx942's MUBUF unit.
> Same-target gfx1250 → gfx1250 lifts continue to emit the native
> LLVM intrinsic directly. This doc is the design brief; for the
> sibling per-lane FLAT async DMA see
> [`async-copy-translation.md`](async-copy-translation.md).

---

## 1. Problem in one paragraph

gfx1250 introduced the Tensor Data Mover (TDM): a hardware unit
that walks a 4/5-D Tensor Descriptor (D#) to stripe a tile of
global memory into/out of LDS, synchronised via its own
`TENSORcnt` counter. The four ops
(`tensor_load_to_lds_d{2,4}` and `tensor_store_from_lds_d{2,4}`,
`MIMGInstructions.td:2049-2113`), their matching LLVM intrinsics
(`llvm.amdgcn.tensor.{load.to.lds,store.from.lds}`,
`IntrinsicsAMDGPU.td:4213`), and the `TENSORcnt` register are all
gated `isGFX125xOnly`. gfx942 has no TDM unit, no matching
intrinsic, and no direct analog in VMUBUF/VIMAGE — a cross-target
lift via the intrinsic fails at isel. The known producers today are
the upstream AMD Gluon TDM-pipelined GEMM examples and any future
compiler path that chooses VIMAGE TDM; the current captured GPT-OSS
MoE `_matmul_ogs` kernels use the sibling FLAT
`global_load_async_to_lds_b*` path documented in
`async-copy-translation.md`.

## 2. SemOp canonicalisation

Both `_d2` (up-to-2D) and `_d4` (up-to-4D) forms share a SemOp:

| Pseudo | SemOp |
|---|---|
| `tensor_load_to_lds_d2_gfx1250` | `TENSOR_LOAD_TO_LDS` |
| `tensor_load_to_lds_d4_gfx1250` | `TENSOR_LOAD_TO_LDS` |
| `tensor_store_from_lds_d2_gfx1250` | `TENSOR_STORE_FROM_LDS` |
| `tensor_store_from_lds_d4_gfx1250` | `TENSOR_STORE_FROM_LDS` |
| `s_wait_tensorcnt` | `S_WAIT_TENSORCNT` (no-op handler) |

The opcode-map collapse lives in `opcode_map.cpp` under
`// VIMAGE TENSOR (gfx1250 RDNA4 — VIMAGE 0xc4 / 0xc5)`. The
_d2_/_d4_ distinction is recovered inside the handler from
`op.nSrcs()` (4 vs 6) — the `_d2` encoding pins `vaddr2`/`vaddr3`
to the NULL SGPR sentinel which the disassembler does not surface
as operands, so `_d2` arrives with 4 logical sources and `_d4`
with 6.

## 3. Two-mode handler

All lift logic lives in `handle_vimage.cpp::handleVIMAGE`:

```
           ┌────────────────────────────────────┐
           │ marshalTDMArgs                     │
           │   read 4 D# groups from SGPRs into │
           │   <4 x i32> / <8 x i32> vectors;   │
           │   zero-fill groups 2/3 on _d2;     │
           │   zero-fill group 4 (reserved);    │
           │   pull cpol from the imm operand.  │
           └────────────┬───────────────────────┘
                        │
        ctx.targetIsa.hasTensorOps ?
                        │
         ┌──────────────┴──────────────┐
         ▼ YES (same-target)           ▼ NO (cross-target)
  emit call to                   emit call to
  llvm.amdgcn.tensor.{load.to,   salmon_tdm_{load_to_lds,
    store.from}.lds             store_from_lds}
  (IntrinsicsAMDGPU.td:4213)     (runtime/tdm.hip, link-merged)
```

### 3.1 Same-target (gfx1250 → gfx1250)

Principled lift. Build `Intrinsic::amdgcn_tensor_load_to_lds` /
`Intrinsic::amdgcn_tensor_store_from_lds` via
`Intrinsic::getOrInsertDeclaration` and call it with the six-tuple
`(grp0, grp1, grp2, grp3, grp4=zero, cpol)`. The intrinsic's
signature mirrors the hardware operand bank exactly; the backend
re-emits the VIMAGE TENSOR encoding at isel.

### 3.2 Cross-target (gfx1250 → gfx942 and earlier)

Emit a call to the HIP-authored device runtime in
`transpiler/runtime/tdm.hip`. Two C-linkage entry points:

```c
void salmon_tdm_load_to_lds  (v4i g0, v8i g1, v4i g2, v4i g3);
void salmon_tdm_store_from_lds(v4i g0, v8i g1, v4i g2, v4i g3);
```

Declared at IR-level by `tdm_runtime.{hpp,cpp}` and **link-merged
into the kernel module before `verifyModule`** — not loaded as a
device library at dispatch time. The merge happens in
`linkTDMRuntime`, called once per raise from `raiser.cpp`.
`tdm_runtime_blob.cpp` is generated at build time from
`tdm_runtime.gfx942.bc` (the compiled bitcode of `tdm.hip`) — if
the transpiler is built without hipcc the blob is empty,
`tdmRuntimeAvailable()` returns false, and `handleVIMAGE` falls
back to the pre-existing loud refusal (no silent miscompile).

Emulation strategy inside `tdm.hip` (see its header comment for
the authoritative spec):

- **X-axis OOB + multi-dim row OOB** fold onto `num_records=0`
  and delegate to MUBUF via
  `__builtin_amdgcn_raw_buffer_{load,store}_b{8,16,32,64}` against
  a per-row V# constructed from the D#.
- **`data_size` dispatch** (1/2/4/8 bytes) happens once per call
  in the entry point; the rest of the walk is templated on
  `<bool IsLoad, unsigned DS>` so per-element switching
  disappears.
- **Lane parallelism.** Lane L handles X = L, L+W, L+2W, …
  (W = `__builtin_amdgcn_wavefrontsize()`); X is unique per
  lane so the stripe is race-free on both the global and LDS
  sides.
- **EXEC gating.** The handler wraps each helper call in
  `RaiseContext::emitUnderExec`, so inactive modeled source lanes skip
  the entire descriptor walk even when the target projection keeps
  hardware EXEC widened between side-effect diamonds.
- **Atomic-barrier side effect** (descriptor group 1 field) is gated
  to lane 0.

Field offsets track the MI450 SPG Tensor DMA Resource Descriptor
tables (79–84).

The runtime helpers take only the four D# groups. They deliberately
drop the intrinsic's trailing `grp4_reserved` and `cpol` immediate:
group 4 is reserved by the intrinsic contract, and `cpol` is a
cache-policy immediate with no equivalent target encoding in the
MUBUF-based helper path. The descriptor-visible semantics, including
atomic-barrier updates, are encoded in the forwarded D# groups.

## 4. `S_WAIT_TENSORCNT`

Pre-registered as a SemOp with a no-op handler. On same-target
gfx1250 the intrinsic-emit path lets the backend re-emit the wait
naturally; on cross-target the synchronous helper call is already
ordered by LLVM's memory effects model (every TDM helper call is
an acquire+release memory barrier over LDS + global), so explicit
`S_WAIT_TENSORCNT` instructions between TDM ops collapse to
nothing meaningful after linking. See `async-copy-translation.md`
for the sibling `S_WAIT_ASYNCCNT` posture — same rationale.

## 5. Code map

| File | Role |
|---|---|
| `transpiler/handle_vimage.cpp` | Handler + arg marshal; two-mode dispatch (same-target intrinsic vs cross-target helper) |
| `transpiler/semop.{hpp,cpp}` | `TENSOR_LOAD_TO_LDS`, `TENSOR_STORE_FROM_LDS`, `S_WAIT_TENSORCNT` SemOp definitions |
| `transpiler/opcode_map.cpp` | _d2/_d4 MC-opcode → SemOp canonicalisation |
| `transpiler/runtime/tdm.hip` | HIP-authored emulation runtime (gfx942), two C-linkage entry points |
| `transpiler/tdm_runtime.{hpp,cpp}` | `declareTDMLoad` / `declareTDMStore`, `linkTDMRuntime`, `tdmRuntimeAvailable` |
| `transpiler/tdm_runtime_blob.cpp` | Generated: embeds `tdm_runtime.gfx942.bc` as a byte array |
| `transpiler/CMakeLists.txt` | Build rule that compiles `tdm.hip` → `.bc` → embedded blob (requires hipcc) |

## 6. Tests

| Test | Kind | What it pins |
|---|---|---|
| `lit_tests/tensor_load_to_lds.s` | Lit (no GPU) | Cross-target helper-call emission + same-target intrinsic emit for representative `_d2` load |
| `lit_tests/tensor_store_from_lds.s` | Lit (no GPU) | Cross-target helper-call emission for representative `_d2` store |
| `TdmRuntime.LinkerWiring` | gtest (no GPU) | Embedded bitcode parses and both helper bodies materialise |
| `TdmGpu.CrossTargetCorpus` | gtest (GPU) | Sweep `test_data/gfx1250`; lift every TDM-using HSACO; `hipModuleLoadData` on gfx942 |
| `TdmDescriptorCoverage.DispatchDenseContiguous/{Load,Store}_{1..5}D` | gtest (GPU) | Parameterised dispatch + byte-compare for each (direction × rank) cell — functional fence on the walker |
| `TdmGpu.LoadStoreRoundtrip5D` | gtest (GPU) | Load+store composition in one kernel, multi-wave dispatch forces workgroup LDS sync |

GPU test fixtures live in
`hotswap/test_data/gfx1250/tdm_{load,store,load_store}_kernel.hip`
with their compiled `.hsaco` siblings; each fixture's header
comment contains the full regen recipe.

## 7. Gotcha — VIMAGE TENSOR NULL sentinel byte

ROCm 7.2.1's hipcc and the salmon-linked LLVM 23 disassembler
disagree on where the NULL SGPR sentinel `0x7C` sits in the
12-byte encoding: hipcc writes it into byte 4 (`vdata` slot),
LLVM 23 looks for it in byte 7 (`vaddr4` slot). Without the
sentinel where LLVM 23 expects it, the MCDisassembler rejects the
middle dword as `v_illegal` at instruction+4 and the raiser
aborts with `Unsupported instruction: v_illegal`.

Two workarounds, one per fixture class:

- **Lit fixtures** (`lit_tests/tensor_*.s`) emit the 12 bytes by hand
  with `.long` directives in checked-in ASM-lit sources.
- **Dispatchable fixtures**
  (`test_data/gfx1250/tdm_{load,store,load_store}_kernel.hip`)
  use the hipcc builtin and then byte-patch the compiled `.hsaco`
  post-unbundle to rewrite each TENSOR op's dword1 from
  `0x00000000` → `0x7C000000`. The full python patch is baked
  into each `.hip` header under "Regen recipe".

The drift is purely in the disassembler's recognition of the
encoding; both byte layouts refer to the same operational
instruction on gfx1250 hardware, but only the byte-7 layout is
decoded by salmon's MC pipeline. Remove the workaround when the
bundled LLVM picks up the byte-4 layout upstream.

#include "code_object_utils.hpp"

#include "llvm/BinaryFormat/AMDGPUMetadataVerifier.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/BinaryFormat/MsgPackDocument.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstring>
#include <optional>

namespace transpiler {

namespace {

using OwnedElf = llvm::object::OwningBinary<llvm::object::ELF64LEObjectFile>;

inline uint32_t readU32(const uint8_t *p) {
  uint32_t v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}
inline uint16_t readU16(const uint8_t *p) {
  uint16_t v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

// Parse `bytes` as an AMDGPU ELF64LE code object. Failing to open the ELF
// is fatal: every downstream query needs a valid parsed object, so there
// is nothing a caller could meaningfully do with a failure.
OwnedElf openELF64LE(const std::vector<uint8_t> &bytes) {
  auto buf = llvm::MemoryBuffer::getMemBuffer(
      llvm::StringRef(reinterpret_cast<const char *>(bytes.data()),
                      bytes.size()),
      "", false);
  auto objOrErr = llvm::object::ObjectFile::createELFObjectFile(*buf);
  if (!objOrErr)
    llvm::report_fatal_error(
        llvm::Twine("transpiler: failed to parse ELF: ") +
        llvm::toString(objOrErr.takeError()));

  auto elf = llvm::unique_dyn_cast<llvm::object::ELF64LEObjectFile>(*objOrErr);
  if (!elf)
    llvm::report_fatal_error("transpiler: code object is not ELF64LE");

  return OwnedElf(std::move(elf), std::move(buf));
}

// Walk the ELF note sections looking for the AMDGPU HSA code-object-v3
// metadata blob, deserialize it via msgpack, and hand the `amdhsa.kernels`
// array to `accept`. Returns true iff `accept` fired for some note. Logs
// section/note iterator failures to `errs()` with a fixed prefix.
bool walkAmdgpuMetadata(
    const llvm::object::ELF64LEObjectFile &elf,
    llvm::function_ref<bool(llvm::msgpack::ArrayDocNode &)> accept) {
  constexpr llvm::StringLiteral kBanner = "transpiler: metadata: ";
  const auto &elfFile = elf.getELFFile();
  auto sectionsOrErr = elfFile.sections();
  if (!sectionsOrErr) {
    llvm::logAllUnhandledErrors(sectionsOrErr.takeError(), llvm::errs(),
                                kBanner);
    return false;
  }

  for (const auto &shdr : *sectionsOrErr) {
    if (shdr.sh_type != llvm::ELF::SHT_NOTE)
      continue;

    llvm::Error err = llvm::Error::success();
    for (auto note : elfFile.notes(shdr, err)) {
      if (note.getType() != llvm::ELF::NT_AMDGPU_METADATA ||
          note.getName() != "AMDGPU")
        continue;

      llvm::msgpack::Document doc;
      if (!doc.readFromBlob(note.getDescAsStringRef(4), /*Multi=*/false))
        continue;

      // Non-strict verify coerces untyped scalars to expected types, matching
      // what llvm-readobj does. On rejection we keep going with the raw doc
      // (preserving pre-refactor tolerance) but warn once so genuinely
      // malformed metadata is visible.
      if (!llvm::AMDGPU::HSAMD::V3::MetadataVerifier(/*Strict=*/false)
               .verify(doc.getRoot())) {
        llvm::errs() << "transpiler: warning: AMDGPU metadata failed "
                        "non-strict verification; continuing with raw doc\n";
      }

      auto &root = doc.getRoot();
      if (!root.isMap())
        continue;
      auto kernelsIt = root.getMap().find("amdhsa.kernels");
      if (kernelsIt == root.getMap().end() || !kernelsIt->second.isArray())
        continue;

      if (accept(kernelsIt->second.getArray())) {
        llvm::consumeError(std::move(err));
        return true;
      }
    }
    if (err) {
      llvm::logAllUnhandledErrors(std::move(err), llvm::errs(), kBanner);
      return false;
    }
  }
  return false;
}

// Locate an ELF section by name. Silently skips sections whose names cannot
// be decoded. Returns `std::nullopt` if no matching section exists.
std::optional<llvm::object::SectionRef>
findSectionByName(const llvm::object::ObjectFile &obj, llvm::StringRef name) {
  for (const auto &sec : obj.sections()) {
    auto nameOrErr = sec.getName();
    if (!nameOrErr) { llvm::consumeError(nameOrErr.takeError()); continue; }
    if (*nameOrErr == name)
      return sec;
  }
  return std::nullopt;
}

// Look up a key in a msgpack map; returns nullptr if absent.
llvm::msgpack::DocNode *findKey(llvm::msgpack::MapDocNode &map,
                                llvm::StringRef key) {
  auto it = map.find(key);
  return it != map.end() ? &it->second : nullptr;
}

// Coerce a msgpack Int/UInt node to int64_t. The AMDGPU metadata verifier
// accepts either (see `MetadataVerifier::verifyInteger`). Any other kind
// falls through to 0 — preserves the pre-refactor tolerant behaviour; a
// strict variant that asserts would be a separate behaviour change.
int64_t nodeInt(const llvm::msgpack::DocNode &n) {
  if (n.getKind() == llvm::msgpack::Type::UInt)
    return static_cast<int64_t>(n.getUInt());
  if (n.getKind() == llvm::msgpack::Type::Int)
    return n.getInt();
  return 0;
}

// Locate the `<kernelName>.kd` symbol and copy its 64 KD bytes into `out`.
// Returns true on success. The KD symbol is *always* in the .rodata section
// for amdhsa code objects (the AMDGPU asm printer emits it there); we map
// the symbol's virtual address back to a file-level byte offset within the
// section's contents and copy the canonical 64-byte structure. Any
// mismatch (missing symbol, wrong size, address not within .rodata) is
// reported and produces `false`.
//
// We deliberately key off the symbol rather than the MsgPack metadata: the
// MsgPack notes do not include kernarg_preload_length / preload_offset,
// and that information is essential for modelling the gfx1250 user-SGPR
// ABI in Phase 4 of the raiser.
bool readKernelDescriptorBytes(llvm::object::ObjectFile &obj,
                               const std::string &kernelName,
                               std::array<uint8_t, 64> &out) {
  std::string kdSymName = kernelName + ".kd";

  auto rodataSec = findSectionByName(obj, ".rodata");
  if (!rodataSec) {
    llvm::errs() << "transpiler: readKernelDescriptorBytes: no .rodata "
                    "section in code object\n";
    return false;
  }

  uint64_t rodataAddr = rodataSec->getAddress();
  uint64_t rodataSize = rodataSec->getSize();
  auto rodataContentsOrErr = rodataSec->getContents();
  if (!rodataContentsOrErr) {
    (void)llvm::toString(rodataContentsOrErr.takeError());
    return false;
  }
  auto rodataContents = *rodataContentsOrErr;

  for (const auto &sym : obj.symbols()) {
    auto nameOrErr = sym.getName();
    if (!nameOrErr) {
      (void)llvm::toString(nameOrErr.takeError());
      continue;
    }
    if (*nameOrErr != kdSymName)
      continue;

    auto addrOrErr = sym.getAddress();
    if (!addrOrErr) {
      (void)llvm::toString(addrOrErr.takeError());
      return false;
    }
    uint64_t symAddr = *addrOrErr;

    if (symAddr < rodataAddr || symAddr + 64 > rodataAddr + rodataSize) {
      llvm::errs() << "transpiler: readKernelDescriptorBytes: symbol '"
                   << kdSymName << "' at 0x" << llvm::utohexstr(symAddr)
                   << " is not contained within .rodata [0x"
                   << llvm::utohexstr(rodataAddr) << ", 0x"
                   << llvm::utohexstr(rodataAddr + rodataSize) << ")\n";
      return false;
    }

    uint64_t off = symAddr - rodataAddr;
    if (off + 64 > rodataContents.size()) {
      llvm::errs() << "transpiler: readKernelDescriptorBytes: symbol '"
                   << kdSymName << "' offset 0x" << llvm::utohexstr(off)
                   << " + 64 exceeds .rodata contents size 0x"
                   << llvm::utohexstr(rodataContents.size()) << "\n";
      return false;
    }

    std::memcpy(out.data(),
                reinterpret_cast<const uint8_t *>(rodataContents.data()) + off,
                64);
    return true;
  }

  llvm::errs() << "transpiler: readKernelDescriptorBytes: symbol '" << kdSymName
               << "' not found\n";
  return false;
}

// Parse the four KD register fields we care about into `meta`. Wraps
// readKernelDescriptorBytes so the call site stays compact and the byte-
// offset constants are co-located with their usage.
void populateKernelDescriptorFields(llvm::object::ObjectFile &obj,
                                    KernelMeta &meta) {
  std::array<uint8_t, 64> kdBytes;
  if (!readKernelDescriptorBytes(obj, meta.name, kdBytes)) {
    meta.hasKernelDescriptor = false;
    return;
  }

  using namespace llvm::amdhsa;
  meta.privateSegmentFixedSize =
      readU32(kdBytes.data() + PRIVATE_SEGMENT_FIXED_SIZE_OFFSET);
  meta.computePgmRsrc1 = readU32(kdBytes.data() + COMPUTE_PGM_RSRC1_OFFSET);
  meta.computePgmRsrc2 = readU32(kdBytes.data() + COMPUTE_PGM_RSRC2_OFFSET);
  meta.kernelCodeProperties =
      readU16(kdBytes.data() + KERNEL_CODE_PROPERTIES_OFFSET);
  meta.kernargPreload = readU16(kdBytes.data() + KERNARG_PRELOAD_OFFSET);
  meta.hasKernelDescriptor = true;
}

} // namespace

std::vector<uint8_t> readFile(const std::string &path) {
  auto bufOrErr = llvm::MemoryBuffer::getFile(path);
  if (!bufOrErr) {
    llvm::errs() << "transpiler: cannot read " << path << ": "
                 << bufOrErr.getError().message() << "\n";
    return {};
  }
  llvm::StringRef data = (*bufOrErr)->getBuffer();
  return std::vector<uint8_t>(data.bytes_begin(), data.bytes_end());
}

TextSection extractTextSection(const std::vector<uint8_t> &elfData) {
  TextSection result;
  auto owned = openELF64LE(elfData);
  const auto *elf = owned.getBinary();

  auto textSec = findSectionByName(*elf, ".text");
  if (!textSec) {
    llvm::errs() << "transpiler: .text section not found in ELF\n";
    return result;
  }
  auto contentsOrErr = textSec->getContents();
  if (!contentsOrErr) {
    llvm::errs() << "transpiler: failed to read .text contents: "
                 << llvm::toString(contentsOrErr.takeError()) << "\n";
    return result;
  }
  result.bytes.assign(contentsOrErr->begin(), contentsOrErr->end());
  result.offset = textSec->getAddress();
  result.size = textSec->getSize();
  result.valid = true;
  return result;
}

std::vector<std::string> listKernelNames(const std::vector<uint8_t> &elfData) {
  std::vector<std::string> names;
  auto owned = openELF64LE(elfData);
  const auto *elf = owned.getBinary();

  walkAmdgpuMetadata(*elf, [&](llvm::msgpack::ArrayDocNode &kernels) {
    for (auto &kNode : kernels) {
      if (!kNode.isMap()) continue;
      if (auto *n = findKey(kNode.getMap(), ".name"))
        names.push_back(n->toString());
    }
    return true; // stop after the first AMDGPU metadata note
  });
  return names;
}

KernelMeta extractKernelMeta(const std::vector<uint8_t> &elfData,
                             const std::string &kernelName) {
  KernelMeta meta;
  auto owned = openELF64LE(elfData);
  auto *elf = owned.getBinary();

  bool found = walkAmdgpuMetadata(
      *elf, [&](llvm::msgpack::ArrayDocNode &kernels) {
        for (auto &kNode : kernels) {
          if (!kNode.isMap()) continue;
          auto &kMap = kNode.getMap();

          auto *nameNode = findKey(kMap, ".name");
          if (!nameNode || nameNode->toString() != kernelName)
            continue;

          meta.name = kernelName;
          if (auto *n = findKey(kMap, ".kernarg_segment_size"))
            meta.kernargSegmentSize = nodeInt(*n);
          if (auto *n = findKey(kMap, ".group_segment_fixed_size"))
            meta.groupSegmentFixedSize = nodeInt(*n);
          if (auto *n = findKey(kMap, ".private_segment_fixed_size"))
            meta.privateSegmentFixedSize = nodeInt(*n);
          if (auto *n = findKey(kMap, ".max_flat_workgroup_size"))
            meta.maxFlatWorkgroupSize = nodeInt(*n);

          if (auto *argsNode = findKey(kMap, ".args");
              argsNode && argsNode->isArray()) {
            for (auto &argNode : argsNode->getArray()) {
              if (!argNode.isMap()) continue;
              auto &aMap = argNode.getMap();
              KernelArgMeta am;
              if (auto *n = findKey(aMap, ".name")) am.name = n->toString();
              if (auto *n = findKey(aMap, ".offset")) am.offset = nodeInt(*n);
              if (auto *n = findKey(aMap, ".size")) am.size = nodeInt(*n);
              if (auto *n = findKey(aMap, ".value_kind")) am.valueKind = n->toString();
              if (auto *n = findKey(aMap, ".address_space"))
                am.addressSpace = nodeInt(*n);
              meta.args.push_back(am);
            }
          }
          return true;
        }
        return false;
      });
  if (!found) {
    llvm::errs() << "transpiler: extractKernelMeta: kernel '" << kernelName
                 << "' not found in metadata\n";
    return meta;
  }

  // Parse the KD bytes from .rodata once we know the kernel name
  // matched. populateKernelDescriptorFields sets
  // meta.hasKernelDescriptor on success and emits a diagnostic on
  // failure; the caller (raiser / Phase-4 init) is responsible for
  // refusing the lift if the field is false rather than silently
  // assuming a hardcoded SGPR layout.
  populateKernelDescriptorFields(*elf, meta);
  return meta;
}

uint64_t findKernelSymbolOffset(const std::vector<uint8_t> &elfData,
                                const std::string &kernelName) {
  auto owned = openELF64LE(elfData);
  const auto *elf = owned.getBinary();

  auto textSec = findSectionByName(*elf, ".text");
  if (!textSec) {
    llvm::errs() << "transpiler: findKernelSymbolOffset: no .text section found\n";
    return 0;
  }
  const uint64_t textBase = textSec->getAddress();

  for (const auto &sym : elf->symbols()) {
    auto nameOrErr = sym.getName();
    if (!nameOrErr) { llvm::consumeError(nameOrErr.takeError()); continue; }
    if (*nameOrErr != kernelName) continue;
    auto addrOrErr = sym.getAddress();
    if (!addrOrErr) { llvm::consumeError(addrOrErr.takeError()); continue; }
    if (*addrOrErr < textBase) {
      llvm::errs() << "transpiler: findKernelSymbolOffset: symbol address 0x"
                   << llvm::utohexstr(*addrOrErr) << " < .text base 0x"
                   << llvm::utohexstr(textBase) << "\n";
      return 0;
    }
    return *addrOrErr - textBase;
  }

  llvm::errs() << "transpiler: findKernelSymbolOffset: symbol '" << kernelName
               << "' not found, defaulting to offset 0\n";
  return 0;
}

std::string detectIsaFromElf(const std::vector<uint8_t> &elfData) {
  // Read the EI_CLASS / e_machine / e_flags fields straight off the
  // ELF64 header rather than building a full ObjectFile — this is
  // called from raise_cli BEFORE we have a CO opened, and we want
  // zero overhead and zero diagnostics on the malformed-input path.
  // The header layout is fixed by the ELF spec (see ELF.h
  // `Elf64_Ehdr`); the magic check rejects every non-ELF input.
  if (elfData.size() < sizeof(llvm::ELF::Elf64_Ehdr))
    return {};
  const auto *eh =
      reinterpret_cast<const llvm::ELF::Elf64_Ehdr *>(elfData.data());
  if (eh->e_ident[llvm::ELF::EI_MAG0] != llvm::ELF::ElfMagic[0] ||
      eh->e_ident[llvm::ELF::EI_MAG1] != llvm::ELF::ElfMagic[1] ||
      eh->e_ident[llvm::ELF::EI_MAG2] != llvm::ELF::ElfMagic[2] ||
      eh->e_ident[llvm::ELF::EI_MAG3] != llvm::ELF::ElfMagic[3])
    return {};
  if (eh->e_ident[llvm::ELF::EI_CLASS] != llvm::ELF::ELFCLASS64)
    return {};
  if (eh->e_machine != llvm::ELF::EM_AMDGPU)
    return {};
  uint32_t mach = eh->e_flags & llvm::ELF::EF_AMDGPU_MACH;
  // ELF.h's AMDGPU_MACH_LIST X-macro pairs every mach value with
  // its canonical "gfxNNN[a-z]?" / R600 marketing string. The macro
  // covers both R600 and AMDGCN families; we filter R600 mach codes
  // (0x01..0x10) explicitly because the transpiler only targets
  // AMDGCN. EF_AMDGPU_MACH_NONE (0x00) likewise returns "" so a
  // generic / older code object falls through to the filename
  // heuristic in raise_cli.
  if (mach >= llvm::ELF::EF_AMDGPU_MACH_R600_FIRST &&
      mach <= llvm::ELF::EF_AMDGPU_MACH_R600_LAST)
    return {};
  if (mach == llvm::ELF::EF_AMDGPU_MACH_NONE)
    return {};
  switch (mach) {
#define HANDLE_AMDGCN(NUM, ENUM, STR)                                          \
  case NUM:                                                                    \
    return STR;
    AMDGPU_MACH_LIST(HANDLE_AMDGCN)
#undef HANDLE_AMDGCN
  default:
    return {};
  }
}

} // namespace transpiler

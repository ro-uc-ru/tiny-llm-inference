#pragma once

#include <cstddef>
#include <filesystem>
#include <span>

namespace tiny_llm {

// Owns a read-only memory mapping of a whole file: the file's bytes appear in memory
// without being copied, and each page is read from disk the first time it is touched.
//
// The file descriptor is closed as soon as the mapping exists (the kernel keeps its own
// reference to the file), so the only resource this class owns is the mapping itself.
// An empty file gives an empty MappedFile: mmap rejects a size of 0, so nothing is mapped.
class MappedFile {
  public:
    // open + fstat + mmap + close.
    // Throws if the file cannot be opened, inspected or mapped (an environment error).
    // `explicit`: a path never turns into a MappedFile without writing MappedFile.
    explicit MappedFile(const std::filesystem::path& path);

    // munmap if this object owns a mapping; does nothing if it is empty.
    // Aborts if munmap fails: that only happens with a corrupted bytes_, which is a bug.
    ~MappedFile();

    // A mapping has a single owner, copying is forbidden: two copies would munmap twice.
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    // Moving transfers the mapping and leaves `other` empty, so only one destructor unmaps.
    // noexcept: a move is two assignments and cannot fail (std::vector relies on it).
    MappedFile(MappedFile&& other) noexcept;
    MappedFile& operator=(MappedFile&& other) noexcept;

    // Read-only view of the mapped bytes. It points into the mapping, not into this object:
    // it stays valid if the MappedFile is moved, and dangles once the owner is destroyed.
    [[nodiscard]] std::span<const std::byte> bytes() const noexcept { return bytes_; }

  private:
    // munmap the mapping if there is one, and leave bytes_ empty. Shared by the destructor
    // and the move assignment, which must drop its own mapping before taking another.
    void release() noexcept;

    // Start and size of the mapping, which are exactly what munmap needs.
    // Empty means "owns nothing" (empty file or moved-from).
    std::span<const std::byte> bytes_;
};

} // namespace tiny_llm

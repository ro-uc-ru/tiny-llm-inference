#pragma once

#include <cstddef>
#include <filesystem>
#include <span>

namespace tiny_llm {

/**
 * @brief Owns a read-only memory mapping of a whole file.
 *
 * The file's bytes appear in memory without being copied, and each page is read from disk
 * the first time it is touched.
 *
 * The file descriptor is closed as soon as the mapping exists (the kernel keeps its own
 * reference to the file), so the only resource this class owns is the mapping itself.
 * An empty file gives an empty MappedFile: `mmap` rejects a size of 0, so nothing is mapped.
 */
class MappedFile {
  public:
    /**
     * @brief Maps the whole file: `open` + `fstat` + `mmap` + `close`.
     *
     * `explicit`: a path never turns into a MappedFile without writing MappedFile.
     *
     * @param path The file to map.
     * @throws LoadError If `open`, `fstat` or `mmap` fails. The message names the call and the
     *         errno text ("open: No such file or directory"), not the path: the caller adds it.
     */
    explicit MappedFile(const std::filesystem::path& path);

    /**
     * @brief Unmaps the file if this object owns a mapping; does nothing if it is empty.
     *
     * Checks that `munmap` succeeds: it only fails with a corrupted `bytes_`, which is a bug.
     */
    ~MappedFile();

    /**
     * A mapping has a single owner, so copying is forbidden: two copies would unmap twice.
     */
    MappedFile(const MappedFile&) = delete;
    /**
     * @copydoc MappedFile(const MappedFile&)
     */
    MappedFile& operator=(const MappedFile&) = delete;

    /**
     * @brief Takes the mapping of `other` and leaves `other` empty, so only one destructor
     *        unmaps it.
     *
     * `noexcept`: a move is two assignments and cannot fail (`std::vector` relies on it).
     *
     * @param other The source; empty afterwards.
     */
    MappedFile(MappedFile&& other) noexcept;

    /**
     * @brief Releases this object's own mapping, then takes the one of `other`.
     *
     * Self-assignment (`file = std::move(file)`) does nothing.
     *
     * @param other The source; empty afterwards.
     * @return This object.
     */
    MappedFile& operator=(MappedFile&& other) noexcept;

    /**
     * @brief Read-only view of the mapped bytes.
     *
     * It points into the mapping, not into this object: it stays valid if the MappedFile is
     * moved, and dangles once the owner is destroyed.
     *
     * @return The whole file, or an empty span for an empty or moved-from MappedFile.
     */
    [[nodiscard]] std::span<const std::byte> bytes() const noexcept { return bytes_; }

  private:
    /**
     * @brief Unmaps the mapping if there is one, and leaves `bytes_` empty.
     *
     * Shared by the destructor and the move assignment, which must drop its own mapping
     * before taking another.
     */
    void release() noexcept;

    /**
     * Start and size of the mapping, which are exactly what `munmap` needs.
     * Empty means "owns nothing" (empty file or moved-from).
     */
    std::span<const std::byte> bytes_;
};

} // namespace tiny_llm

#include <tiny_llm/check.hpp>
#include <tiny_llm/load_error.hpp>
#include <tiny_llm/mapped_file.hpp>

#include <cerrno>
#include <fcntl.h> // open
#include <format>
#include <sys/mman.h> // mmap, munmap
#include <sys/stat.h> // fstat
#include <system_error>
#include <unistd.h> // close
#include <utility>

namespace tiny_llm {
namespace {

/**
 * @brief Builds the LoadError of a failed system call: "open: No such file or directory".
 *
 * No path: MappedFile says what failed, the caller says with which file.
 *
 * @param call_name The system call that failed, such as "open".
 * @param error_number The errno it left, saved before any other call could overwrite it.
 * @return The error, ready to throw.
 */
LoadError system_call_error(const char* call_name, int error_number) {
    return LoadError(
        std::format("{}: {}", call_name, std::generic_category().message(error_number)));
}

} // namespace

/**
 * The constructor either finishes with a valid object or throws. If it throws, the
 * destructor never runs, so every exit after a successful open closes the descriptor
 * itself before leaving.
 */
MappedFile::MappedFile(const std::filesystem::path& path) {
    // Ask the kernel to open the file; the descriptor is our ticket for every later call.
    const int file_descriptor = ::open(path.c_str(), O_RDONLY);
    if (file_descriptor == -1) {
        throw system_call_error("open", errno);
    }

    // `struct` picks the type: POSIX has both a function and a type called stat.
    struct stat file_info{};
    if (::fstat(file_descriptor, &file_info) == -1) {
        const int saved_error = errno; // close may overwrite errno
        ::close(file_descriptor);
        throw system_call_error("fstat", saved_error);
    }

    // st_size is off_t (signed); a regular file never reports a negative size.
    const auto size = static_cast<std::size_t>(file_info.st_size);

    // mmap rejects a size of 0: an empty file gives a valid, empty MappedFile.
    if (size == 0) {
        ::close(file_descriptor);
        return;
    }

    // Read-only private mapping of the whole file; the kernel chooses the address.
    // Nothing is copied: each 16 KiB page is read from disk the first time it is touched.
    void* const address = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, file_descriptor, 0);
    if (address == MAP_FAILED) { // MAP_FAILED is (void*)-1, not nullptr
        const int saved_error = errno;
        ::close(file_descriptor);
        throw system_call_error("mmap", saved_error);
    }

    // The mapping keeps its own reference to the file, so the descriptor can go now.
    // Ignored on purpose: a failed close on a read-only file loses nothing.
    static_cast<void>(::close(file_descriptor));

    // From here on, this object owns the mapping.
    bytes_ = std::span{static_cast<const std::byte*>(address), size};
}

MappedFile::~MappedFile() {
    release();
}

void MappedFile::release() noexcept {
    // Empty file or moved-from object: there is no mapping to release.
    if (bytes_.empty()) {
        return;
    }
    // munmap takes a non-const void*; it does not write the bytes, it only needs the address.
    void* const address = const_cast<std::byte*>(bytes_.data());
    // munmap only fails for an address or size that is not a mapping: a bug in this class.
    // A destructor cannot throw, and ignoring it would hide the bug, so it is a check.
    const int munmap_result = ::munmap(address, bytes_.size());
    TINY_LLM_CHECK(munmap_result == 0, "munmap failed: bytes_ does not describe a mapping");
    bytes_ = {};
}

/**
 * Takes other's mapping and leaves other empty, so only one destructor will unmap it.
 * The member initializer list builds bytes_ directly with the taken value.
 */
MappedFile::MappedFile(MappedFile&& other) noexcept : bytes_{std::exchange(other.bytes_, {})} {}

/**
 * The target may already own a mapping: release it first, or nobody would ever unmap it.
 */
MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    // file = std::move(file): releasing first would lose our own mapping.
    if (this == &other) {
        return *this;
    }
    release();
    bytes_ = std::exchange(other.bytes_, {});
    return *this;
}

} // namespace tiny_llm

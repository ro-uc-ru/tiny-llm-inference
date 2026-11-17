#include <tiny_llm/config.hpp>
#include <tiny_llm/load_error.hpp>
#include <tiny_llm/mapped_file.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

/**
 * Each test gets its own files in the temporary directory, named after the test so that
 * tests running in parallel never share one. TearDown removes them even if the test fails.
 */
class MappedFileTest : public ::testing::Test {
  protected:
    void TearDown() override {
        for (const auto& path : temporary_paths_) {
            std::filesystem::remove(path);
        }
    }

    std::filesystem::path write_temporary_file(std::span<const std::byte> contents,
                                               const std::string& suffix = "") {
        const std::string test_name =
            ::testing::UnitTest::GetInstance()->current_test_info()->name();
        const auto path =
            std::filesystem::temp_directory_path() / ("tiny_llm_" + test_name + suffix);

        std::ofstream output(path, std::ios::binary);
        output.write(reinterpret_cast<const char*>(contents.data()),
                     static_cast<std::streamsize>(contents.size()));
        temporary_paths_.push_back(path);
        return path;
    }

  private:
    std::vector<std::filesystem::path> temporary_paths_;
};

constexpr std::array<std::byte, 5> small_contents{std::byte{0x74}, std::byte{0x69}, std::byte{0x6e},
                                                  std::byte{0x79}, std::byte{0x00}}; // "tiny\0"

constexpr std::array<std::byte, 3> other_contents{std::byte{0x6c}, std::byte{0x6c},
                                                  std::byte{0x6d}}; // "llm"

} // namespace

TEST_F(MappedFileTest, MapsTheWholeFile) {
    const auto path = write_temporary_file(small_contents);

    const tiny_llm::MappedFile file{path};

    EXPECT_EQ(file.bytes().size(), small_contents.size());
    EXPECT_TRUE(std::ranges::equal(file.bytes(), small_contents));
}

TEST_F(MappedFileTest, EmptyFileGivesEmptyBytes) {
    const auto path = write_temporary_file({});

    const tiny_llm::MappedFile file{path};

    EXPECT_TRUE(file.bytes().empty());
}

/**
 * A moved-from MappedFile is valid but empty: reading it right after the move is part of
 * this class's contract, not an accident.
 */
TEST_F(MappedFileTest, MoveConstructionTransfersTheMapping) {
    const auto path = write_temporary_file(small_contents);
    tiny_llm::MappedFile original{path};
    const std::byte* const mapped_address = original.bytes().data();

    const tiny_llm::MappedFile moved{std::move(original)};

    EXPECT_EQ(moved.bytes().data(), mapped_address); // same mapping, nothing copied
    EXPECT_EQ(moved.bytes().size(), small_contents.size());
    EXPECT_TRUE(original.bytes().empty());
}

TEST_F(MappedFileTest, MoveAssignmentTakesTheOtherMapping) {
    tiny_llm::MappedFile target{write_temporary_file(small_contents, "_target")};
    tiny_llm::MappedFile source{write_temporary_file(other_contents, "_source")};
    const std::byte* const source_address = source.bytes().data();

    target = std::move(source); // target must release its own mapping first

    EXPECT_EQ(target.bytes().data(), source_address);
    EXPECT_TRUE(std::ranges::equal(target.bytes(), other_contents));
    EXPECT_TRUE(source.bytes().empty());
}

/**
 * Self-move assignment must leave the mapping untouched. Without the `this == &other` guard,
 * release() would unmap it and then take the (now empty) bytes of `other`, which is the same
 * object: the file would end up silently empty. No sanitizer catches that, only this test.
 *
 * `file = std::move(file)` does not compile (-Wself-move with -Werror), so the move goes
 * through a second name for the same object, as `files[i] = std::move(files[j])` with
 * `i == j` would in real code.
 */
TEST_F(MappedFileTest, SelfMoveAssignmentKeepsTheMapping) {
    const auto path = write_temporary_file(small_contents);
    tiny_llm::MappedFile file{path};
    const std::byte* const mapped_address = file.bytes().data();
    tiny_llm::MappedFile& same_file = file;

    file = std::move(same_file);

    EXPECT_EQ(file.bytes().data(), mapped_address);
    EXPECT_EQ(file.bytes().size(), small_contents.size());
    EXPECT_TRUE(std::ranges::equal(file.bytes(), small_contents));
}

TEST(MappedFileErrors, MissingFileThrows) {
    EXPECT_THROW(tiny_llm::MappedFile{"/nonexistent/tiny_llm/model.bin"}, tiny_llm::LoadError);
}

/**
 * The message says what failed and why, but not where: the caller adds the path.
 */
TEST(MappedFileErrors, MissingFileMessageHasNoPath) {
    try {
        tiny_llm::MappedFile file{"/nonexistent/tiny_llm/model.bin"};
        FAIL() << "expected a LoadError";
    } catch (const tiny_llm::LoadError& error) {
        EXPECT_STREQ(error.what(), "open: No such file or directory");
    }
}

/**
 * The second load failure, end to end: a file that exists but is shorter than its header.
 * Mapping it works; parsing the header is what fails.
 */
TEST_F(MappedFileTest, FileShorterThanHeaderThrows) {
    const std::array<std::byte, 10> truncated_contents{};
    const tiny_llm::MappedFile file{write_temporary_file(truncated_contents)};

    EXPECT_THROW(static_cast<void>(tiny_llm::parse_config(file.bytes())), tiny_llm::LoadError);
}

/**
 * End to end with the real model. The model is not versioned, so the test is skipped
 * unless TINY_LLM_MODEL_PATH points to stories110M.bin.
 */
TEST(MappedFileModel, Stories110MHeaderParses) {
    const char* model_path = std::getenv("TINY_LLM_MODEL_PATH");
    if (model_path == nullptr) {
        GTEST_SKIP() << "TINY_LLM_MODEL_PATH is not set";
    }

    const tiny_llm::MappedFile file{model_path};
    EXPECT_EQ(file.bytes().size(), 438'381'596u);

    const tiny_llm::Config config = tiny_llm::parse_config(file.bytes());
    EXPECT_EQ(config.embedding_size, 768u);
    EXPECT_EQ(config.feed_forward_size, 2048u);
    EXPECT_EQ(config.layer_count, 12u);
    EXPECT_EQ(config.head_count, 12u);
    EXPECT_EQ(config.kv_head_count, 12u);
    EXPECT_EQ(config.vocab_size, 32000u);
    EXPECT_EQ(config.max_sequence_length, 1024u);
}

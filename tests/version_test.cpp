#include <tiny_llm/version.hpp>

#include <gtest/gtest.h>

/**
 * A trivial test whose only job is to prove the pipeline works:
 * CMake builds the library and the test binary, ctest finds and runs the test,
 * and the sanitizers are linked in without complaint.
 */
TEST(VersionTest, MatchesProjectVersion) {
    EXPECT_EQ(tiny_llm::version(), "0.1.0");
}

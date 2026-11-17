#include <tiny_llm/check.hpp>

#include <gtest/gtest.h>

namespace {

/**
 * Test suites whose name ends in DeathTest run before all the others: GoogleTest forks a
 * child process for each death test, and forking is safest before other tests start threads.
 *
 * EXPECT_DEATH(statement, regex) runs `statement` in the child and passes if the child dies
 * (any signal or non-zero exit) and its stderr matches `regex`. The parent process survives.
 */

TEST(CheckDeathTest, FailedCheckAbortsWithItsMessage) {
    const int head_count = 0;
    EXPECT_DEATH(TINY_LLM_CHECK(head_count > 0, "head count must be positive"),
                 "check failed: head_count > 0\n  message: head count must be positive");
}

TEST(CheckDeathTest, FailedCheckReportsFileAndLine) {
    EXPECT_DEATH(TINY_LLM_CHECK(1 + 1 == 3, "arithmetic is broken"), "check_test\\.cpp:[0-9]+: ");
}

/**
 * The message is only evaluated when the check fails: a passing check must not call it.
 */
TEST(CheckTest, PassingCheckDoesNotEvaluateTheMessage) {
    int message_evaluation_count = 0;
    const auto count_and_return_message = [&message_evaluation_count] {
        ++message_evaluation_count;
        return "never shown";
    };

    TINY_LLM_CHECK(2 + 2 == 4, count_and_return_message());

    EXPECT_EQ(message_evaluation_count, 0);
}

} // namespace

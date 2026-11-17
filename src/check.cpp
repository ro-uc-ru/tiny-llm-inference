#include <tiny_llm/check.hpp>

#include <cstdio>
#include <cstdlib>
#include <print>

namespace tiny_llm {

/**
 * [[noreturn]] lives on the declaration in check.hpp; the definition does not repeat it.
 */
void check_failed(std::string_view condition, std::string_view message,
                  std::source_location location) {
    // "file:line:" first, the format compilers use, so editors and CI logs link to the line.
    // stderr is unbuffered: the text is out before abort() kills the process.
    std::println(stderr, "{}:{}: check failed: {}\n  message: {}\n  in: {}", location.file_name(),
                 location.line(), condition, message, location.function_name());

    // No exception and no cleanup: a broken invariant means the state can no longer be trusted.
    // abort() raises SIGABRT (exit code 134 in the shell), which a debugger stops at with the
    // stack intact, and which EXPECT_DEATH detects in its child process.
    std::abort();
}

} // namespace tiny_llm

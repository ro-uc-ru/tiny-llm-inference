#pragma once

#include <stdexcept>
#include <string>

namespace tiny_llm {

/**
 * @brief An environment error while loading a file.
 *
 * The file is missing, cannot be mapped, is shorter than its header or holds an impossible
 * header. It is not a bug in this code, so it is thrown, and the caller decides what to do
 * (today, `main` prints it and exits with 1). Bugs (broken invariants) are not LoadErrors:
 * they abort with TINY_LLM_CHECK. See docs/error-handling.md.
 *
 * The message says what failed and, for system calls, the errno text
 * ("open: No such file or directory"). It does not include the path: the thrower may not
 * know it (parse_config() only sees bytes), so the caller that knows the path adds it.
 *
 * Inherits from `std::runtime_error`, which stores the message and returns it from `what()`,
 * so `catch (const std::exception&)` also sees it.
 */
class LoadError : public std::runtime_error {
  public:
    /**
     * @brief Builds the error with its message.
     * @param message What failed, without the path.
     *
     * `explicit`: a `std::string` never turns into a LoadError without writing LoadError.
     */
    explicit LoadError(const std::string& message) : std::runtime_error(message) {}
};

} // namespace tiny_llm

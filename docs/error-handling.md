# Error handling

The engine has two kinds of failure, and each has its own style. Mixing two styles without rules
is worse than either, so this page is the rule.

| Failure | Example | Style |
| --- | --- | --- |
| **Bug:** a broken invariant in our own code | A shape that does not match in a matmul, an index out of range, `munmap` rejecting our own mapping | `TINY_LLM_CHECK` prints and aborts |
| **Environment error:** the outside world is not as expected | A missing file, a file shorter than its header, a header with impossible values | `throw tiny_llm::LoadError`, caught once in `main` |

The split is the same as Rust's: `panic!` for bugs, `Result` for errors.

## Bugs: `TINY_LLM_CHECK`

```cpp
#include <tiny_llm/check.hpp>

TINY_LLM_CHECK(row_index < row_count, "row index out of range");
```

If the condition is false, it prints the location, the condition as written and the message to
stderr, then calls `std::abort()`:

```
src/tensor.cpp:42: check failed: row_index < row_count
  message: row index out of range
  in: float tiny_llm::Tensor::at(std::size_t, std::size_t) const
```

- **Always on**, also in release builds. Unlike `assert`, it does not disappear with `NDEBUG`:
  checking a shape costs nothing next to a matmul.
- **No cleanup.** Destructors do not run. Carrying on with corrupted state is worse than stopping,
  and `abort` leaves the stack exactly where it broke for the debugger.
- **The message is only evaluated on failure**, so it can be built with `std::format`.
- **Usable anywhere**, including destructors, `noexcept` functions and hot loops.
- **Tested with death tests:** `EXPECT_DEATH(statement, regex)` runs the statement in a child
  process and checks that it dies with the expected stderr.

## Environment errors: `LoadError`

```cpp
#include <tiny_llm/load_error.hpp>

throw tiny_llm::LoadError(std::format("a llama2.c checkpoint header needs {} bytes, got {}",
                                      header_size_in_bytes, bytes.size()));
```

- **One class**, derived from `std::runtime_error`. No hierarchy: no caller handles a missing file
  differently from a truncated one today.
- **The message says what failed and why**, with the `errno` text for system calls
  (`open: No such file or directory`).
- **The message does not include the path.** The thrower may not know it (`parse_config` only sees
  bytes). The caller that knows the path adds it.
- **Thrown only at the loading boundary**, never in the hot path.
- **Tested with** `EXPECT_THROW(statement, tiny_llm::LoadError)`.

## Where errors are caught

Only `main` (`apps/tiny_llm.cpp`) catches, and only `LoadError`:

```
$ tiny_llm /missing/model.bin
error loading /missing/model.bin: open: No such file or directory
$ echo $?
1
```

| Exit code | Meaning |
| --- | --- |
| 0 | Success |
| 1 | `LoadError`: the file could not be loaded |
| 2 | Usage error: wrong arguments |
| 134 | `SIGABRT`: a check failed (a bug) or an unexpected exception reached `std::terminate` |

Any other exception (`std::bad_alloc`, `std::format_error`…) is not caught on purpose: it is
unexpected, so it ends in `std::terminate` like a bug, instead of passing for an environment error.

## Documenting a function

Every public function says in its Doxygen comment what it throws (`@throws LoadError …`) and, in
the text, which invariants it checks.

## Not used

- **`std::expected<T, E>`** (C++23) puts the error in the return type, like Rust's `Result`. Using
  it alongside exceptions would mean a third style.
- **`std::optional<T>`** is for "maybe no value, and the reason does not matter". It will appear if
  something needs it.

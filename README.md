# tiny-llm-inference

A small LLM inference engine in modern C++ (C++23), written from scratch to understand where the time goes when a transformer generates a token on a CPU, and how to make it faster.

> **Status:** early. The repository holds only the project skeleton; there is nothing to build yet.

## Goals

- Load [llama2.c](https://github.com/karpathy/llama2.c) checkpoints (stories110M, TinyLlama 1.1B) with `mmap`, without copying the weights.
- An fp32 forward pass that produces the same tokens as llama2.c's `run.c` at temperature 0.
- An optimized matmul (loop order, tiling, NEON intrinsics, OpenMP), each version measured in GFLOP/s, tokens/s and percentage of the machine's roofline.
- Group-wise int8 quantization in the style of llama.cpp's Q8_0, compared against llama.cpp on speed, memory and output quality.

## Roadmap

| Milestone | What it adds |
| --- | --- |
| 1 · Foundations | `MappedFile` (RAII over `mmap`), `Tensor` (a non-owning view), error handling, CMake presets, tests and CI |
| 2 · fp32 engine | `Config`, `Weights`, `RunState`, the forward pass, sampler and tokenizer |
| 3 · Fast matmul | Roofline measurement and five matmul versions, from naive to NEON + OpenMP |
| 4 · Quantization | int8 group quantization and the comparison with llama.cpp |

## Design principles

- **Explicit ownership.** Every buffer has exactly one owner. The weights belong to the memory-mapped file; everything else looks at them through views.
- **Two error styles.** Broken invariants (bugs) abort through an always-on check; environment errors (a missing or truncated file) throw an exception that is caught once in `main`. Documented in `docs/error-handling.md` (to come).
- **No warning is harmless.** Strict warnings as errors, AddressSanitizer and UndefinedBehaviorSanitizer during development, and CI on every push.

## Layout

```text
include/tiny_llm/   public headers
src/                implementation
tests/              GoogleTest tests (they generate their own small files; no model weights needed)
docs/               design notes
.github/workflows/  CI
```

## Building

Coming soon: CMake with presets and Ninja. Developed on macOS with Apple Clang 21; CI on Linux with clang and gcc.

## Acknowledgements

The checkpoint format and the reference implementation come from Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c).

## License

[MIT](LICENSE).

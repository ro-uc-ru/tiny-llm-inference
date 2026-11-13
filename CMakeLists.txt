# tiny-llm-inference — build description.
#
# CMake thinks in *targets* (a library, an executable) and each target carries its own
# properties: sources, include directories, compile flags, libraries it links against.
# PUBLIC properties travel to whoever links the target; PRIVATE ones stay inside it.

# 3.28: FetchContent_Declare(... SYSTEM) needs 3.25, and the presets file uses schema 6.
cmake_minimum_required(VERSION 3.28)

project(tiny_llm
    VERSION 0.1.0
    DESCRIPTION "A small LLM inference engine in modern C++"
    LANGUAGES CXX)

# --- Language -----------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # fail instead of silently falling back to C++20
set(CMAKE_CXX_EXTENSIONS OFF)        # -std=c++23, not -std=gnu++23

# Writes build/<preset>/compile_commands.json: the exact command for every .cpp,
# which is what clangd reads to give the same diagnostics as the compiler.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Options (set from CMakePresets.json, or -D on the command line) ----------------------

option(ENGINE_SANITIZE "Build everything with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(ENGINE_BUILD_TESTS "Build the GoogleTest suite" ON)

# Sanitizers apply to *every* target, GoogleTest included: mixing instrumented and
# uninstrumented code that shares containers gives false positives.
if(ENGINE_SANITIZE)
    add_compile_options(
        -fsanitize=address,undefined
        -fno-sanitize-recover=undefined  # UBSan aborts on the first report, so ctest sees a failure
        -fno-omit-frame-pointer)         # readable stack traces in the reports
    add_link_options(-fsanitize=address,undefined)
endif()

# --- Warning policy -----------------------------------------------------------------------

# An INTERFACE library has no sources: it only carries properties. Linking it gives a
# target these flags, so the policy lives here once and applies only to *our* code
# (never to GoogleTest, which is not written for -Wold-style-cast).
add_library(tiny_llm_warnings INTERFACE)
target_compile_options(tiny_llm_warnings INTERFACE
    -Wall -Wextra -Wpedantic -Werror
    -Wshadow
    -Wconversion -Wsign-conversion
    -Wold-style-cast
    -Wnon-virtual-dtor
    -Woverloaded-virtual)

# --- The engine library -------------------------------------------------------------------

add_library(tiny_llm
    src/version.cpp)

# PUBLIC: our own sources and anyone who links tiny_llm can #include <tiny_llm/...>.
target_include_directories(tiny_llm PUBLIC include)

# PRIVATE: only version.cpp needs to know the number; it is not part of the API.
target_compile_definitions(tiny_llm PRIVATE TINY_LLM_VERSION="${PROJECT_VERSION}")

target_link_libraries(tiny_llm PRIVATE tiny_llm_warnings)

# --- Tests --------------------------------------------------------------------------------

if(ENGINE_BUILD_TESTS)
    enable_testing()

    # FetchContent downloads and builds GoogleTest as part of our build, pinned to a
    # release and checked against its hash. SYSTEM makes its headers -isystem, so its
    # code never trips our warnings.
    include(FetchContent)
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
        URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
        SYSTEM)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_executable(tiny_llm_tests
        tests/version_test.cpp)

    # gtest_main provides main(); the test files only hold TEST(...) blocks.
    target_link_libraries(tiny_llm_tests PRIVATE tiny_llm tiny_llm_warnings GTest::gtest_main)

    # Asks the test binary for its list of tests after building it, so ctest sees each
    # TEST(...) as a separate test instead of one big executable.
    include(GoogleTest)
    gtest_discover_tests(tiny_llm_tests)
endif()

cmake_minimum_required(VERSION 3.16)
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()
project(gpt2infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(GPT2_BUILD_TESTS "Build unit tests" ON)
option(GPT2_SANITIZERS "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

if(GPT2_SANITIZERS)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

add_compile_options(-ffp-contract=off)

set(GPT2_AVX2 0)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  set(GPT2_AVX2 1)
endif()

add_library(gpt2
  src/thread_pool.cpp
  src/gemm.cpp
  src/gemm_scalar.cpp
  src/ops.cpp
  src/unicode.cpp
  src/bpe.cpp
  src/tokenizer.cpp
  src/sampler.cpp
  src/model.cpp
  src/generate.cpp
)
if(GPT2_AVX2)
  target_sources(gpt2 PRIVATE src/gemm_avx2.cpp)
endif()

target_include_directories(gpt2 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_definitions(gpt2 PUBLIC GPT2_COMPILE_AVX2=${GPT2_AVX2})
target_compile_options(gpt2 PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Werror)
find_package(Threads REQUIRED)
target_link_libraries(gpt2 PUBLIC Threads::Threads)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set_source_files_properties(src/gemm_scalar.cpp PROPERTIES
    COMPILE_OPTIONS "-fno-vectorize;-fno-slp-vectorize")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set_source_files_properties(src/gemm_scalar.cpp PROPERTIES
    COMPILE_OPTIONS "-fno-tree-vectorize;-fno-tree-slp-vectorize")
endif()
if(GPT2_AVX2)
  set_source_files_properties(src/gemm_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
endif()

add_executable(gpt2_infer src/main.cpp)
target_link_libraries(gpt2_infer PRIVATE gpt2)
target_compile_options(gpt2_infer PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Werror)

if(GPT2_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  add_executable(gpt2_tests
    tests/test_gemm.cpp
    tests/test_ops.cpp
    tests/test_tokenizer.cpp
    tests/test_sampler.cpp
    tests/test_model.cpp
    tests/test_hf_logits.cpp
  )
  target_link_libraries(gpt2_tests PRIVATE gpt2 GTest::gtest_main)
  target_compile_definitions(gpt2_tests PRIVATE GPT2_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  target_compile_options(gpt2_tests PRIVATE -Wall -Wextra -Wpedantic -Wshadow -ffp-contract=off)
  include(GoogleTest)
  gtest_discover_tests(gpt2_tests)
endif()

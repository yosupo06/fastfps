cmake_policy(SET CMP0048 NEW)
project(FastFPS)

cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
FetchContent_Declare(
  googletest
  # Specify the commit you depend on and update it regularly.
  URL https://github.com/google/googletest/archive/f8d7d77c06936315286eb55f8de22cd23c188571.zip
)
FetchContent_MakeAvailable(googletest)
FetchContent_Declare(googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG main)
FetchContent_MakeAvailable(googlebenchmark)


add_compile_options(-Wall -Wextra -Wshadow -Wconversion -Wno-sign-conversion -Werror)
add_compile_options(-mavx2)

# sanitizer
add_compile_options("$<$<CONFIG:DEBUG>:-fsanitize=undefined,address;-fno-sanitize-recover=all>")
add_link_options("$<$<CONFIG:DEBUG>:-fsanitize=undefined,address>")

include_directories(../src)
include_directories(./util)

# unittest
add_executable(test
  unittest/fft_test.cpp
  unittest/modint_test.cpp
  unittest/simd_test.cpp
  unittest/math_test.cpp
  unittest/modint8_test.cpp
  unittest/modvec_test.cpp)
target_link_libraries(test gtest_main)
add_test(NAME test COMMAND test)

# benchmark
add_executable(fft_bench benchmark/fft_benchmark.cpp)
target_link_libraries(fft_bench benchmark::benchmark)

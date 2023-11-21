cmake_policy(SET CMP0048 NEW)
project(FastFPS)

cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -Wshadow -Wconversion -Wno-sign-conversion -Werror)
add_compile_options(-fsanitize=undefined,address -fno-sanitize-recover=all)
add_compile_options(-mavx2)
add_link_options(-fsanitize=undefined,address)

include(FetchContent)
FetchContent_Declare(
  googletest
  # Specify the commit you depend on and update it regularly.
  URL https://github.com/google/googletest/archive/f8d7d77c06936315286eb55f8de22cd23c188571.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include_directories(../../src)

add_executable(test fft_test.cpp modint_test.cpp simd_test.cpp math_test.cpp)
target_link_libraries(test gtest_main)
add_test(NAME test COMMAND test)

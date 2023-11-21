#include "simd.hpp"

#include <numeric>
#include <random>
#include <vector>
#include <array>

#include <gtest/gtest.h>

using namespace fastfps;

using u32 = uint32_t;
using u64 = uint64_t;

// blend
TEST(SimdTest, BlendU32x8) {
    u32x8 a = {1, 2, 3, 4, 5, 6, 7, 8};
    u32x8 b = {10, 20, 30, 40, 50, 60, 70, 80};

    auto expect = std::array<u32, 8>({1, 20, 30, 4, 5, 6, 7, 80});

    ASSERT_EQ(expect, (a.blend<0b10000110>(b)).to_array());
}

// add
TEST(SimdTest, AddU32x8) {
    u32x8 a = {1, 2, 3, 4, 5, 6, 7, 8};
    u32x8 b = {10, 20, 30, 40, 50, 60, 70, 80};

    auto expect = std::array<u32, 8>({11, 22, 33, 44, 55, 66, 77, 88});

    ASSERT_EQ(expect, (a + b).to_array());
}
TEST(SimdTest, AddU64x4) {
    u64x4 a = {1, 2, 3, 4};
    u64x4 b = {10, 20, 30, 40};

    auto expect = std::array<u64, 4>({11, 22, 33, 44});

    ASSERT_EQ(expect, (a + b).to_array());
}

// sub
TEST(SimdTest, SubU32x8) {
    u32x8 a = {10, 20, 30, 40, 50, 60, 70, 80};
    u32x8 b = {1, 2, 3, 4, 5, 6, 7, 8};

    auto expect = std::array<u32, 8>({9, 18, 27, 36, 45, 54, 63, 72});

    ASSERT_EQ(expect, (a - b).to_array());
}
TEST(SimdTest, SubU64x4) {
    u64x4 a = {10, 20, 30, 40};
    u64x4 b = {1, 2, 3, 4};

    auto expect = std::array<u64, 4>({9, 18, 27, 36});

    ASSERT_EQ(expect, (a - b).to_array());
}

// mul
TEST(SimdTest, Mul0U32x8) {
    u32x8 a = {10, 20, 30, 40, 50, 60, 70, 80};
    u32x8 b = {1, 2, 3, 4, 5, 6, 7, 8};

    auto expect = std::array<u64, 4>({10, 90, 250, 490});

    ASSERT_EQ(expect, mul0(a, b).to_array());
}
TEST(SimdTest, Mul1U32x8) {
    u32x8 a = {10, 20, 30, 40, 50, 60, 70, 80};
    u32x8 b = {1, 2, 3, 4, 5, 6, 7, 8};

    auto expect = std::array<u64, 4>({40, 160, 360, 640});

    ASSERT_EQ(expect, mul1(a, b).to_array());
}
TEST(SimdTest, MulSignU32x8) {
    ASSERT_EQ(u64(-1u), mul0(u32x8(-1u), u32x8(1u)).at(0));
}

// shift
TEST(SimdTest, RShiftU64x4) {
    u64x4 a = {2, 4, 6, 8};

    auto expect = std::array<u64, 4>({1, 2, 3, 4});

    ASSERT_EQ(expect, (a.rshift<1>()).to_array());
}

// at
TEST(SimdTest, AtU32x8) {
    u32x8 a = {1, 2, 3, 4, 5, 6, 7, 8};

    ASSERT_EQ(1, a.at(0));
    ASSERT_EQ(2, a.at(1));
    ASSERT_EQ(3, a.at(2));
    ASSERT_EQ(4, a.at(3));
    ASSERT_EQ(5, a.at(4));
    ASSERT_EQ(6, a.at(5));
    ASSERT_EQ(7, a.at(6));
    ASSERT_EQ(8, a.at(7));
}
TEST(SimdTest, AtU64x4) {
    u64x4 a = {1, 2, 3, 4};

    ASSERT_EQ(1L, a.at(0));
    ASSERT_EQ(2L, a.at(1));
    ASSERT_EQ(3L, a.at(2));
    ASSERT_EQ(4L, a.at(3));
}

// set
TEST(SimdTest, SetU32x8) {
    u32x8 a = {1, 2, 3, 4, 5, 6, 7, 8};

    a.set(0, 10);
    a.set(1, 20);
    a.set(2, 30);
    a.set(3, 40);
    a.set(4, 50);
    a.set(5, 60);
    a.set(6, 70);
    a.set(7, 80);

    auto expect = std::array<u32, 8>({10, 20, 30, 40, 50, 60, 70, 80});

    ASSERT_EQ(expect, a.to_array());
}
TEST(SimdTest, SetU64x4) {
    u64x4 a = {1, 2, 3, 4};

    a.set(0, 10);
    a.set(1, 20);
    a.set(2, 30);
    a.set(3, 40);

    auto expect = std::array<u64, 4>({10, 20, 30, 40});

    ASSERT_EQ(expect, a.to_array());
}

// to
TEST(SimdTest, U64x4ToU32x8) {
    u64x4 a = {(1ull << 32) | 2, (3ull << 32) | 4, (5ull << 32) | 6,
               (7ull << 32) | 8};

    auto expect = std::array<u32, 8>({2, 1, 4, 3, 6, 5, 8, 7});

    ASSERT_EQ(expect, a.to_u32x8().to_array());
}

TEST(SimdTest, MinU32x8) {
    u32x8 a = {1, 20, 30, 4, 50, 6, 7, 8};
    u32x8 b = {10, 2, 3, 40, 5, 60, 70, 80};

    auto expect = std::array<u32, 8>({1, 2, 3, 4, 5, 6, 7, 8});

    ASSERT_EQ(expect, min(a, b).to_array());
}
TEST(SimdTest, MaxU32x8) {
    u32x8 a = {1, 20, 30, 4, 50, 6, 7, 8};
    u32x8 b = {10, 2, 3, 40, 5, 60, 70, 80};

    auto expect = std::array<u32, 8>({10, 20, 30, 40, 50, 60, 70, 80});

    ASSERT_EQ(expect, max(a, b).to_array());
}

TEST(SimdTest, PermuteVar) {
    u32x8 a = {0, 1, 2, 3, 4, 5, 6, 7};
    u32x8 idx = {6, 6, 2, 7, 0, 1, 6, 7};

    auto expect = std::array<u32, 8>({6, 6, 2, 7, 0, 1, 6, 7});

    ASSERT_EQ(expect, a.permutevar(idx).to_array());
}

#include <array>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "simd.hpp"

using namespace fastfps;

using u32 = uint32_t;
using u64 = uint64_t;

// set1
TEST(SimdTest, Set1U32x8) {
    ASSERT_EQ(u32x8(123, 123, 123, 123, 123, 123, 123, 123), u32x8::set1(123));
}
TEST(SimdTest, Set1U64x4) {
    ASSERT_EQ(u64x4(234, 234, 234, 234), u64x4::set1(234));
}

// to_array
TEST(SimdTest, ToArrayU32x8) {
    u32x8 a = u32x8(1, 2, 3, 4, 5, 6, 7, 8);
    auto expect = std::array<u32, 8>({1, 2, 3, 4, 5, 6, 7, 8});
    ASSERT_EQ(expect, a.to_array());
}
TEST(SimdTest, ToArrayU64x4) {
    u64x4 a = u64x4(1, 2, 3, 4);
    auto expect = std::array<u64, 4>({1, 2, 3, 4});
    ASSERT_EQ(expect, a.to_array());
}

// blend
TEST(SimdTest, BlendU32x8) {
    u32x8 a = {1, 2, 3, 4, 5, 6, 7, 8};
    u32x8 b = {10, 20, 30, 40, 50, 60, 70, 80};

    ASSERT_EQ(u32x8(1, 20, 30, 4, 5, 6, 7, 80), a.blend<0b10000110>(b));
}

// equal
TEST(SimdTest, EqualU32x8) {
    u32x8 a = {1, 2, 3, 4, 5, 6, 7, 8};
    u32x8 b = {1, 2, 3, 4, 5, 6, 7, 8};
    u32x8 c = {1, 20, 3, 4, 5, 6, 7, 8};

    ASSERT_TRUE(a == b);
    ASSERT_FALSE(a == c);
}
TEST(SimdTest, EqualU64x4) {
    u64x4 a = {1, 2, 3, 4};
    u64x4 b = {1, 2, 3, 4};
    u64x4 c = {10, 2, 3, 4};

    ASSERT_TRUE(a == b);
    ASSERT_FALSE(a == c);
}

// add
TEST(SimdTest, AddU32x8) {
    u32x8 a = {1, 2, 3, 4, 5, 6, 7, 8};
    u32x8 b = {10, 20, 30, 40, 50, 60, 70, 80};

    ASSERT_EQ(u32x8(11, 22, 33, 44, 55, 66, 77, 88), a + b);
}
TEST(SimdTest, AddU64x4) {
    u64x4 a = {1, 2, 3, 4};
    u64x4 b = {10, 20, 30, 40};

    ASSERT_EQ(u64x4(11, 22, 33, 44), a + b);
}

// sub
TEST(SimdTest, SubU32x8) {
    u32x8 a = {10, 20, 30, 40, 50, 60, 70, 80};
    u32x8 b = {1, 2, 3, 4, 5, 6, 7, 8};

    ASSERT_EQ(u32x8(9, 18, 27, 36, 45, 54, 63, 72), a - b);
}
TEST(SimdTest, SubU64x4) {
    u64x4 a = {10, 20, 30, 40};
    u64x4 b = {1, 2, 3, 4};

    ASSERT_EQ(u64x4(9, 18, 27, 36), a - b);
}

// mul
TEST(SimdTest, MulU32x8) {
    u32x8 a = {10, 20, 30, 40, 50, 60, 70, 80};
    u32x8 b = {1, 2, 3, 4, 5, 6, 7, 8};

    ASSERT_EQ(u32x8(10, 40, 90, 160, 250, 360, 490, 640), a * b);
}
TEST(SimdTest, Mul0U32x8) {
    u32x8 a = {10, 20, 30, 40, 50, 60, 70, 80};
    u32x8 b = {1, 2, 3, 4, 5, 6, 7, 8};

    ASSERT_EQ(u64x4(10, 90, 250, 490), mul0(a, b));
}
TEST(SimdTest, Mul1U32x8) {
    u32x8 a = {10, 20, 30, 40, 50, 60, 70, 80};
    u32x8 b = {1, 2, 3, 4, 5, 6, 7, 8};

    ASSERT_EQ(u64x4(40, 160, 360, 640), mul1(a, b));
}
TEST(SimdTest, MulSignU32x8) {
    ASSERT_EQ(u64(-1u), mul0(u32x8::set1(-1u), u32x8::set1(1u)).at(0));
}

// shift
TEST(SimdTest, RShiftU64x4) {
    u64x4 a = {2, 4, 6, 8};

    ASSERT_EQ(u64x4(1, 2, 3, 4), a.rshift<1>());
}

// at
TEST(SimdTest, AtU32x8) {
    u32x8 a = {1, 2, 3, 4, 5, 6, 7, 8};

    ASSERT_EQ(1u, a.at(0));
    ASSERT_EQ(2u, a.at(1));
    ASSERT_EQ(3u, a.at(2));
    ASSERT_EQ(4u, a.at(3));
    ASSERT_EQ(5u, a.at(4));
    ASSERT_EQ(6u, a.at(5));
    ASSERT_EQ(7u, a.at(6));
    ASSERT_EQ(8u, a.at(7));
}
TEST(SimdTest, AtU64x4) {
    u64x4 a = {1, 2, 3, 4};

    ASSERT_EQ(1ull, a.at(0));
    ASSERT_EQ(2ull, a.at(1));
    ASSERT_EQ(3ull, a.at(2));
    ASSERT_EQ(4ull, a.at(3));
}

// set
TEST(SimdTest, SetU32x8) {
    u32x8 a = {1, 2, 3, 4, 5, 6, 7, 8};

    a.set(1, 20);
    a.set(3, 40);
    a.set(4, 50);
    a.set(7, 80);

    ASSERT_EQ(u32x8(1, 20, 3, 40, 50, 6, 7, 80), a);
}
TEST(SimdTest, SetU64x4) {
    u64x4 a = {1, 2, 3, 4};

    a.set(0, 10);
    a.set(2, 30);
    a.set(3, 40);

    ASSERT_EQ(u64x4(10, 2, 30, 40), a);
}

// to
TEST(SimdTest, U64x4ToU32x8) {
    u64x4 a = {(1ull << 32) | 2, (3ull << 32) | 4, (5ull << 32) | 6,
               (7ull << 32) | 8};

    ASSERT_EQ(u32x8(2, 1, 4, 3, 6, 5, 8, 7), a.to_u32x8());
}

TEST(SimdTest, MinU32x8) {
    u32x8 a = {1, 20, 30, 4, 50, 6, 7, 8};
    u32x8 b = {10, 2, 3, 40, 5, 60, 70, 80};

    ASSERT_EQ(u32x8(1, 2, 3, 4, 5, 6, 7, 8), min(a, b));
}
TEST(SimdTest, MaxU32x8) {
    u32x8 a = {1, 20, 30, 4, 50, 6, 7, 8};
    u32x8 b = {10, 2, 3, 40, 5, 60, 70, 80};

    ASSERT_EQ(u32x8(10, 20, 30, 40, 50, 60, 70, 80), max(a, b));
}

TEST(SimdTest, PermuteVar) {
    u32x8 a = {0, 10, 20, 30, 40, 50, 60, 70};
    u32x8 idx = {6, 6, 2, 7, 0, 1, 6, 7};

    ASSERT_EQ(u32x8(60, 60, 20, 70, 0, 10, 60, 70), a.permutevar(idx));
}

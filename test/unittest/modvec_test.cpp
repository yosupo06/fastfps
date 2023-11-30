#include <array>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "modint.hpp"
#include "modvec.hpp"

#include "random.hpp"

using namespace fastfps;

using i32 = int32_t;
using u32 = uint32_t;
using i64 = int64_t;
using u64 = uint64_t;

const u32 MOD = 998244353;
using modint = ModInt<MOD>;
using modvec = ModVec<MOD>;

TEST(ModVecTest, Val) {
    modvec a = modvec({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    ASSERT_EQ(std::vector<u32>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}), a.val());

    ASSERT_EQ(10u, a.val(10));
    ASSERT_EQ(0u, a.val(-1));
    ASSERT_EQ(0u, a.val(100));
}

TEST(ModVecTest, Add) {
    modvec a = modvec({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    modvec b = modvec({100, 200});
    ASSERT_EQ(modvec({100, 201, 2, 3, 4, 5, 6, 7, 8, 9, 10}), a + b);
    ASSERT_EQ(modvec({100, 201, 2, 3, 4, 5, 6, 7, 8, 9, 10}), b + a);
}

TEST(ModVecTest, Sub) {
    modvec a = modvec({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    modvec b = modvec({100, 200});
    ASSERT_EQ(modvec({-100, -199, 2, 3, 4, 5, 6, 7, 8, 9, 10}), a - b);
    ASSERT_EQ(modvec({100, 199, -2, -3, -4, -5, -6, -7, -8, -9, -10}), b - a);
}

TEST(ModVecTest, Mul) {
    {
        modvec a = modvec({1, 2, 3});
        modvec b = modvec({4, 5, 6});
        ASSERT_EQ(modvec({4, 13, 28, 27, 18}), a * b);
    }
    {
        modvec a = modvec({1, 1, 1, 1, 1, 1, 1, 1, 1});
        modvec b = modvec({-1, 1});
        ASSERT_EQ(modvec({-1, 0, 0, 0, 0, 0, 0, 0, 0, 1}), (a * b));
    }
}

TEST(ModVecTest, RShift) {
    {
        modvec a = modvec({1, 2, 3});
        ASSERT_EQ(modvec({0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3}), a << 8);
    }
    {
        modvec a = modvec({1, 2, 3});
        ASSERT_EQ(modvec({1, 2, 3}), a << 0);
    }
    {
        modvec a = modvec({1, 2, 3});
        ASSERT_EQ(modvec({0, 0, 0, 1, 2, 3}), a << 3);
    }
    ASSERT_EQ(modvec({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}), modvec({1}) << 10);
}

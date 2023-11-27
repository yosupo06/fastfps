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

TEST(ModVecTest, Constructor) {
    modvec a = modvec({1, 2, 3});
    ASSERT_EQ(std::vector<u32>({1, 2, 3}), a.val());
}

TEST(ModVecTest, Add) {
    modvec a = modvec({1, 2, 3});
    modvec b = modvec({4, 5});
    ASSERT_EQ(modvec({5, 7, 3}), a + b);
    ASSERT_EQ(modvec({5, 7, 3}), b + a);
}

TEST(ModVecTest, Sub) {
    modvec a = modvec({1, 2, 3});
    modvec b = modvec({4, 5});
    ASSERT_EQ(modvec({-3, -3, 3}), a - b);
    ASSERT_EQ(modvec({3, 3, -3}), b - a);
}

TEST(ModVecTest, Mul) {
    modvec a = modvec({1, 2, 3});
    modvec b = modvec({4, 5, 6});
    ASSERT_EQ(modvec({4, 13, 28, 27, 18}), a * b);
}

#include "modint.hpp"

#include "testutil/random.hpp"

#include <array>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>

using namespace fastfps;

using i32 = int32_t;
using u32 = uint32_t;
using i64 = int64_t;
using u64 = uint64_t;

const u32 MOD = 998244353;
using mintx8 = modintx8<MOD>;

TEST(ModIntTest, Inv2n32) {
    for (u32 i = 1; i < 100u; i += 2) {
        u32 j = inv_2n32(i);
        ASSERT_EQ((i * j), 1u);
    }
}

// to_array
TEST(ModIntTest, ToArray) {
    mintx8 a = u32x8(1, 2, 3, 4, 5, 6, 7, 8);
    auto expect = std::array<u32, 8>({1, 2, 3, 4, 5, 6, 7, 8});

    ASSERT_EQ(expect, a.to_array());
}

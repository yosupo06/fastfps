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

TEST(ModIntTest, Constructor) {
    mintx8 a(1, 2 + MOD, 3, 4, 5, 6 + MOD, 7, 8);
    mintx8 b(u32x8(1, 2, 3 + MOD, 4, 5, 6, 7, 8 + MOD));

    mintx8 expect(1, 2, 3, 4, 5, 6, 7, 8);

    ASSERT_EQ(expect, a);
    ASSERT_EQ(expect, b);
}
TEST(ModIntTest, ToArray) {
    mintx8 a(0, 0, 1, 1, 2, 2, 3, 3);

    std::array<u32, 8> expect({0, 0, 1, 1, 2, 2, 3, 3});

    ASSERT_EQ(expect, a.to_array());
}

TEST(ModIntTest, Add) {
    mintx8 a(1, 2, 3, 4, 5, 6, 7, 8 + 1000);
    mintx8 b(1, 2, 3, 4, 5, 6, 7, 8 + MOD - 1000);

    mintx8 expect(2, 4, 6, 8, 10, 12, 14, 16);

    ASSERT_EQ(expect, (a + b));
}

TEST(ModIntTest, Sub) {
    mintx8 a(11, 22, 33, 44, 55, 66, 77, 88);
    mintx8 b(1, 2, 3, 4, 5, 6, 7, 8);

    mintx8 expect(10, 20, 30, 40, 50, 60, 70, 80);

    ASSERT_EQ(expect, (a - b));
}

TEST(ModIntTest, Mul) {
    mintx8 a(1, 2, 3, 4, 5, 6, 7, 8);
    mintx8 b(10, 20, 30, 40, 50, 60, 70, 80);

    mintx8 expect(10, 40, 90, 160, 250, 360, 490, 640);

    ASSERT_EQ(expect, (a * b));
}

TEST(ModIntTest, Equal) {
    mintx8 a(1, 2, 3, 4, 5, 6, 7 + MOD, 8);
    mintx8 b(1, 2, 3 + MOD, 4, 5 + MOD, 6, 7, 8);
    mintx8 c(1, 2, 4, 3, 5, 6, 7, 8);

    ASSERT_TRUE(a == b);
    ASSERT_FALSE(a == c);
}

TEST(ModIntTest, Neg) {
    mintx8 a(0, 0, 1, 1, 2, 2, 3, 3);

    mintx8 expect(0, MOD - 0, MOD - 1, 1, MOD - 2, 2, MOD - 3, 3);

    ASSERT_EQ(expect.to_array(), a.neg<0b01010110>().to_array());
}

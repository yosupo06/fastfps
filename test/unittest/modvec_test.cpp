#include <array>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "fastfps/modint.hpp"
#include "fastfps/modvec.hpp"

using namespace fastfps;

using i32 = int32_t;
using u32 = uint32_t;
using i64 = int64_t;
using u64 = uint64_t;

const u32 MOD = 998244353;
using modint = ModInt<MOD>;
using modvec = ModVec<MOD>;

TEST(ModVecTest, Constructor) {
    ASSERT_EQ(modvec({0, 0, 0, 0}), modvec(4));
    ASSERT_EQ(modvec(std::vector<modint>({1})), modvec({1}));
}

TEST(ModVecTest, Val) {
    modvec a = modvec({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    ASSERT_EQ(std::vector<u32>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}),
              a.val());

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
    {
        modvec a = modvec({1, 2, 3, 4, 5});
        ASSERT_EQ(modvec({2, 4, 6, 8, 10}), a * 2);
        ASSERT_EQ(modvec({-2, -4, -6, -8, -10}), a * -2);
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
    ASSERT_EQ(modvec({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}).val(),
              (modvec({1}) << 10).val());
}

TEST(ModVecTest, Resize) {
    {
        modvec a = modvec({1, 2, 3});
        a.resize(6);
        ASSERT_EQ(modvec({1, 2, 3, 0, 0, 0}), a);
    }
    {
        modvec a = modvec({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        a.resize(2);
        ASSERT_EQ(modvec({1, 2}), a);
    }
}

TEST(ModVecTest, CopyTo) {
    for (int n = 0; n <= 24; n++) {
        std::vector<u32> a(n);
        for (int i = 0; i < n; i++) {
            a[i] = 100 + i;
        }
        const auto a2 = modvec(a);
        for (int start = 0; start <= n; start++) {
            for (int len = 0; start + len <= n; len++) {
                for (int m = len; m <= 24; m++) {
                    std::vector<u32> b(m);
                    for (int i = 0; i < m; i++) {
                        b[i] = 10000 + i;
                    }
                    for (int dst_start = 0; dst_start + len <= m; dst_start++) {
                        auto expect = b;
                        for (int i = 0; i < len; i++) {
                            expect[dst_start + i] = a[start + i];
                        }

                        auto b2 = modvec(b);
                        a2.copy_to(start, len, b2, dst_start);
                        auto actual = b2.val();

                        ASSERT_EQ(expect, actual);
                    }
                }
            }
        }
    }
}

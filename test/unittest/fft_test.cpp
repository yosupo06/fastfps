#include <array>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "fft.hpp"
#include "types.hpp"

#include "random.hpp"

using namespace fastfps;

const u32 MOD = 998244353;
using modint = ModInt<MOD>;
using modint8 = ModInt8<MOD>;

TEST(ModIntTest, FFTInfo) {
    const auto& info = fft_info<MOD>;
    // 998244353 = 2^23 * 119 + 1
    ASSERT_EQ(23, info.ord2);
    for (int i = 0; i < info.ord2; i++) {
        ASSERT_EQ(modint(1), info.w[i] * info.iw[i]);
    }
}

void naive_fft(std::vector<modint8>& a) {
    const auto& info = fft_info<MOD>;

    std::vector<modint> b;
    for (auto x : a) {
        for (auto y : x.val()) {
            b.push_back(y);
        }
    }

    int n = int(b.size());
    std::vector<modint> c(n);
    for (int i = 0; i < n; i++) {
        modint base = 1;
        for (int h = 0; h < info.ord2; h++) {
            if (i & (1 << h)) {
                base = base * info.w[h + 1];
            }
        }
        modint rot = 1;
        for (int j = 0; j < n; j++) {
            c[i] += b[j] * rot;
            rot = rot * base;
        }
    }

    std::vector<modint8> d;
    for (int i = 0; i < n; i += 8) {
        std::array<modint, 8> buf;
        for (int j = 0; j < 8; j++) {
            buf[j] = c[i + j];
        }
        d.push_back(modint8(buf));
    }
    a = d;
}

void naive_ifft(std::vector<modint8>& a) {
    const auto& info = fft_info<MOD>;

    std::vector<u32> b;
    for (auto x : a) {
        for (auto y : x.val()) {
            b.push_back(y);
        }
    }

    int n = int(b.size());
    std::vector<modint> c(n);
    for (int i = 0; i < n; i++) {
        modint base = 1;
        for (int h = 0; h < info.ord2; h++) {
            if (i & (1 << h)) {
                base = base * info.iw[h + 1];
            }
        }
        modint rot = 1;
        for (int j = 0; j < n; j++) {
            c[j] += b[i] * rot;
            rot = rot * base;
        }
    }

    std::vector<modint8> d;
    for (int i = 0; i < n; i += 8) {
        std::array<modint, 8> buf;
        for (int j = 0; j < 8; j++) {
            buf[j] = c[i + j];
        }
        d.push_back(modint8(buf));
    }
    a = d;
}

TEST(FFTTest, ButterflyStress) {
    for (int lg = 0; lg <= 7; lg++) {
        int n = 1 << lg;
        std::vector<modint8> expect(n);
        for (int i = 0; i < n; i++) {
            std::array<u32, 8> a;
            for (int j = 0; j < 8; j++) {
                a[j] = randint(0u, MOD - 1);
            }
            expect[i] = u32x8(a);
        }
        auto actual = expect;

        naive_fft(expect);
        fft(actual);

        ASSERT_EQ(expect, actual);
    }
}

TEST(FFTTest, InvButterflyStress) {
    for (int lg = 0; lg <= 7; lg++) {
        int n = 1 << lg;
        std::vector<modint8> expect(n);
        for (int i = 0; i < n; i++) {
            std::array<u32, 8> a;
            for (int j = 0; j < 8; j++) {
                a[j] = randint(0u, MOD - 1);
            }
            expect[i] = u32x8(a);
        }
        std::vector<modint8> actual = expect;

        naive_ifft(expect);
        ifft(actual);

        ASSERT_EQ(expect, actual);
    }
}

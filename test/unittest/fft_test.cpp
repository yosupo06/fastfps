#include <array>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "types.hpp"
#include "fft.hpp"

#include "testutil/random.hpp"

using namespace fastfps;

const u32 MOD = 998244353;
using mintx8 = modintx8<MOD>;

TEST(ModIntTest, FFTInfo) {
    FFTInfo<MOD> info;
    // 998244353 = 2^23 * 119 + 1
    ASSERT_EQ(23, info.ord2);
    for (int i = 0; i < info.ord2; i++) {
        ASSERT_EQ(1, u64(1) * info.w[i] * info.iw[i] % MOD);
    }
}

void naive_fft(std::vector<mintx8>& a) {
    static FFTInfo<MOD> info;

    std::vector<u32> b;
    for (auto x : a) {
        for (auto y : x.to_array()) {
            b.push_back(y);
        }
    }

    int n = int(b.size());
    std::vector<u32> c(n);
    for (int i = 0; i < n; i++) {
        u64 base = 1;
        for (int h = 0; h < info.ord2; h++) {
            if (i & (1 << h)) {
                base = base * info.w[h + 1] % MOD;
            }
        }
        u64 rot = 1;
        for (int j = 0; j < n; j++) {
            c[i] = (u32)((c[i] + 1ull * b[j] * rot) % MOD);
            rot = rot * base % MOD;
        }
    }

    std::vector<mintx8> d;
    for (int i = 0; i < n; i += 8) {
        std::array<u32, 8> buf;
        for (int j = 0; j < 8; j++) {
            buf[j] = c[i + j];
        }
        d.push_back(mintx8(buf));
    }
    a = d;
}

void naive_ifft(std::vector<mintx8>& a) {
    static FFTInfo<MOD> info;

    std::vector<u32> b;
    for (auto x : a) {
        for (auto y : x.to_array()) {
            b.push_back(y);
        }
    }

    int n = int(b.size());
    std::vector<u32> c(n);
    for (int i = 0; i < n; i++) {
        u64 base = 1;
        for (int h = 0; h < info.ord2; h++) {
            if (i & (1 << h)) {
                base = base * info.iw[h + 1] % MOD;
            }
        }
        u64 rot = 1;
        for (int j = 0; j < n; j++) {
            c[j] = (u32)((c[j] + 1ull * b[i] * rot) % MOD);
            rot = rot * base % MOD;
        }
    }

    std::vector<mintx8> d;
    for (int i = 0; i < n; i += 8) {
        std::array<u32, 8> buf;
        for (int j = 0; j < 8; j++) {
            buf[j] = c[i + j];
        }
        d.push_back(mintx8(buf));
    }
    a = d;
}

TEST(FFTTest, ButterflyStress) {
    for (int ph = 0; ph < 100; ph++) {
        int n = 1 << randint(0, 5);
        std::vector<mintx8> expect(n);
        for (int i = 0; i < n; i++) {
            std::array<u32, 8> a;
            for (int j = 0; j < 8; j++) {
                a[j] = randint(0u, MOD - 1);
            }
            expect[i] = a;
        }
        auto actual = expect;

        naive_fft(expect);
        fft(actual);

        ASSERT_EQ(expect, actual);
    }
}

TEST(FFTTest, InvButterflyStress) {
    for (int ph = 0; ph < 100; ph++) {
        int n = 1 << randint(0, 5);
        std::vector<mintx8> expect(n);
        for (int i = 0; i < n; i++) {
            std::array<u32, 8> a;
            for (int j = 0; j < 8; j++) {
                a[j] = randint(0u, MOD - 1);
            }
            expect[i] = a;
        }
        std::vector<mintx8> actual = expect;

        naive_ifft(expect);
        ifft(actual);

        ASSERT_EQ(expect, actual);
    }
}

#pragma once

#include <immintrin.h>
#include <array>
#include <bit>
#include <vector>

#include "math.hpp"
#include "modint.hpp"
#include "types.hpp"

namespace fastfps {

template <u32 MOD> struct FFTInfo {
    using mintx8 = modintx8<MOD>;
    static constexpr u32 g = primitive_root<MOD>;
    static constexpr int ord2 = std::countr_zero(MOD - 1);

    // w[i]^(2^i) == 1 : w[i] is the fft omega of n=2^i
    std::array<u32, ord2 + 1> w, iw;

    // TODO: optimize
    std::array<u32, std::max(0, ord2 - 1 + 1)> rate2, irate2;
    std::array<mintx8, std::max(0, ord2 - 1 + 1)> rate2x;

    std::array<u32, std::max(0, ord2 - 2 + 1)> rate3, irate3;

    std::array<u32, std::max(0, ord2 - 3 + 1)> rate4, irate4;
    std::array<mintx8, std::max(0, ord2 - 3 + 1)> rate4xi, irate4xi;

    FFTInfo() {
        w[ord2] = pow_mod_constexpr(g, (MOD - 1) >> ord2, MOD);
        iw[ord2] = pow_mod_constexpr(w[ord2], MOD - 2, MOD);
        for (int i = ord2 - 1; i >= 0; i--) {
            w[i] = u32(u64(1) * w[i + 1] * w[i + 1] % MOD);
            iw[i] = u32(u64(1) * iw[i + 1] * iw[i + 1] % MOD);
        }

        // TODO: refactor
        {
            u32 prod = 1, iprod = 1;
            for (int i = 0; i <= ord2 - 2; i++) {
                rate2[i] = w[i + 2] * prod;
                irate2[i] = iw[i + 2] * iprod;
                prod = u32(u64(1) * prod * iw[i + 2] % MOD);
                iprod = u32(u64(1) * iprod * w[i + 2] % MOD);
            }
            for (int i = 0; i <= ord2 - 2; i++) {
                rate2x[i] = mintx8(rate2[i]);
            }
        }
        {
            u32 prod = 1, iprod = 1;
            for (int i = 0; i <= ord2 - 3; i++) {
                rate3[i] = u32(u64(1) * w[i + 3] * prod % MOD);
                irate3[i] = u32(u64(1) * iw[i + 3] * iprod % MOD);
                prod = u32(u64(1) * prod * iw[i + 3] % MOD);
                iprod = u32(u64(1) * iprod * w[i + 3] % MOD);
            }
        }
        {
            u32 prod = 1, iprod = 1;
            for (int i = 0; i <= ord2 - 4; i++) {
                rate4[i] = u32(u64(1) * w[i + 4] * prod % MOD);
                irate4[i] = u32(u64(1) * iw[i + 4] * iprod % MOD);
                prod = u32(u64(1) * prod * iw[i + 4] % MOD);
                iprod = u32(u64(1) * iprod * w[i + 4] % MOD);
                std::array<u32, 8> buf, ibuf;
                buf[0] = ibuf[0] = 1;
                for (int j = 1; j < 8; j++) {
                    buf[j] = u32(u64(1) * buf[j - 1] * rate4[i] % MOD);
                    ibuf[j] = u32(u64(1) * ibuf[j - 1] * irate4[i] % MOD);
                }
                rate4xi[i] = mintx8(u32x8(buf));
                irate4xi[i] = mintx8(u32x8(ibuf));
            }
        }
    }
};

template <u32 MOD> void fft(std::vector<modintx8<MOD>>& a) {
    using mintx8 = modintx8<MOD>;

    static const FFTInfo<MOD> info;

    int n = int(a.size());
    int lg = std::countr_zero((u32)n);
    int h = lg;
    if (h % 2) {
        // 2-base
        int len = n / 2;
        for (int i = 0; i < len; i++) {
            auto l = a[0 * len + i];
            auto r = a[1 * len + i];
            a[0 * len + i] = l + r;
            a[1 * len + i] = l - r;
        }
        h--;
    }

    const u32 w4 = info.w[2], w8 = info.w[3];

    while (h >= 2) {
        // 4-base
        mintx8 rotx(1);

        mintx8 imagx = mintx8(w4);
        for (int start = 0; start < n; start += (1 << h)) {
            auto rot2x = rotx * rotx;
            auto rot3x = rot2x * rotx;
            int len = 1 << (h - 2);
            for (int i = 0; i < len; i++) {
                auto a0 = a[start + 0 * len + i];
                auto a1 = a[start + 1 * len + i] * rotx;
                auto a2 = a[start + 2 * len + i] * rot2x;
                auto a3 = a[start + 3 * len + i] * rot3x;
                a[start + 0 * len + i] = (a0 + a2) + (a1 + a3);
                a[start + 1 * len + i] = (a0 + a2) - (a1 + a3);
                a[start + 2 * len + i] = (a0 - a2) + (a1 - a3) * imagx;
                a[start + 3 * len + i] = (a0 - a2) - (a1 - a3) * imagx;
            }
            rotx *= mintx8(
                info.rate3[std::countr_zero(~(unsigned int)(start >> h))]);
        }
        h -= 2;
    }

    {
        // fft each element
        const mintx8 step1 =
            mintx8(1, 1, 1, 1, 1, w8, u32(1ull * w8 * w8 % MOD),
                   u32(1ull * w8 * w8 % MOD * w8 % MOD));
        const mintx8 step2 = mintx8(1, 1, 1, w4, 1, 1, 1, w4);
        mintx8 rotxi = mintx8(1);
        for (int i = 0; i < n; i++) {
            mintx8 v = a[i] * rotxi;
            v = (v.template neg<0b11110000>() + v.template shufflex4<0b01>()) *
                step1;
            v = (v.template neg<0b11001100>() +
                 v.template shuffle<0b01001110>()) *
                step2;
            v = (v.template neg<0b10101010>() +
                 v.template shuffle<0b10110001>());
            a[i] = v;
            rotxi *= info.rate4xi[std::countr_zero(~(unsigned int)(i))];
        }
    }
}

template <u32 MOD> void ifft(std::vector<modintx8<MOD>>& a) {
    using mintx8 = modintx8<MOD>;

    static const FFTInfo<MOD> info;

    int n = int(a.size());
    int lg = std::countr_zero((u32)n);

    const u32 iw4 = info.iw[2], iw8 = info.iw[3];

    {
        // 8-base
        const mintx8 istep1 =
            mintx8(1, 1, 1, 1, 1, iw8, u32(1ull * iw8 * iw8 % MOD),
                   u32(1ull * iw8 * iw8 % MOD * iw8 % MOD));
        const mintx8 istep2 = mintx8(1, 1, 1, iw4, 1, 1, 1, iw4);
        mintx8 irotxi = mintx8(1);
        for (int i = 0; i < n; i++) {
            mintx8 v = a[i];
            v = (v.template neg<0b10101010>() +
                 v.template shuffle<0b10110001>()) *
                istep2;
            v = (v.template neg<0b11001100>() +
                 v.template shuffle<0b01001110>()) *
                istep1;
            v = (v.template neg<0b11110000>() + v.template shufflex4<0b01>()) *
                irotxi;
            a[i] = v;
            irotxi *= info.irate4xi[std::countr_zero(~(unsigned int)(i))];
        }
    }

    int h = 0;
    while (h + 2 <= lg) {
        // 4-base
        h += 2;

        mintx8 irotx = mintx8(1);
        mintx8 iimagx = mintx8(iw4);
        for (int start = 0; start < n; start += (1 << h)) {
            auto irot2x = irotx * irotx;
            auto irot3x = irot2x * irotx;
            int len = 1 << (h - 2);
            for (int i = 0; i < len; i++) {
                auto a0 = a[start + 0 * len + i];
                auto a1 = a[start + 1 * len + i];
                auto a2 = a[start + 2 * len + i];
                auto a3 = a[start + 3 * len + i];

                auto a0a1 = a0 + a1;
                auto a0na1 = a0 - a1;
                auto a2a3 = a2 + a3;
                auto a2na3iimag = (a2 - a3) * iimagx;

                a[start + 0 * len + i] = a0a1 + a2a3;
                a[start + 1 * len + i] = (a0na1 + a2na3iimag) * irotx;
                a[start + 2 * len + i] = (a0a1 - a2a3) * irot2x;
                a[start + 3 * len + i] = (a0na1 - a2na3iimag) * irot3x;
            }
            irotx *= mintx8(
                info.irate3[std::countr_zero(~(unsigned int)(start >> h))]);
        }
    }

    if (h + 1 == lg) {
        // 2-base
        int len = n / 2;
        for (int i = 0; i < len; i++) {
            auto l = a[0 * len + i];
            auto r = a[1 * len + i];
            a[0 * len + i] = l + r;
            a[1 * len + i] = l - r;
        }
        h++;
    }
}

}  // namespace fastfps

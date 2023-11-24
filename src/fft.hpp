#pragma once

#include <immintrin.h>
#include <array>
#include <bit>
#include <concepts>
#include <ranges>
#include <vector>

#include "math.hpp"
#include "modint.hpp"
#include "modint8.hpp"
#include "types.hpp"

namespace fastfps {

template <u32 MOD> struct FFTInfo {
    using modint = ModInt<MOD>;
    using mintx8 = ModInt8<MOD>;
    
    static constexpr u32 g = primitive_root<MOD>;
    static constexpr int ord2 = std::countr_zero(MOD - 1);

    // w[i]^(2^i) == 1 : w[i] is the fft omega of n=2^i    
    std::array<u32, ord2 + 1> w, iw;

    static constexpr std::array<modint, ord2 + 1> w_mod = []() {
        std::array<modint, ord2 + 1> v;
        v[ord2] = modint(g).pow((MOD - 1) >> ord2);
        for (int i = ord2 - 1; i >= 0; i--) {
            v[i] = v[i + 1] * v[i + 1];
        }
        return v;
    }();
    static constexpr std::array<modint, ord2 + 1> iw_mod = []() {
        std::array<modint, ord2 + 1> v;
        v[ord2] = w_mod[ord2].inv();
        for (int i = ord2 - 1; i >= 0; i--) {
            v[i] = v[i + 1] * v[i + 1];
        }
        return v;
    }();

    static constexpr std::array<modint, ord2 + 1> rot3 = []() {
        std::array<modint, std::max(0, ord2 + 1)> v;
        for (int i = 3; i <= ord2; i++) {
            v[i] = w_mod[i];
            for (int j = 3; j < i; j++) {
                v[i] *= iw_mod[j];
            }
        }
        return v;
    }();
    // rot[i] * rot_shift(8) = rot[i + 8]
    modint rot_shift8(u32 i) const { return rot3[std::countr_one(i >> 3) + 3]; }

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
                rate2[i] = u32(u64(1) * w[i + 2] * prod % MOD);
                irate2[i] = u32(u64(1) * iw[i + 2] * iprod % MOD);
                prod = u32(u64(1) * prod * iw[i + 2] % MOD);
                iprod = u32(u64(1) * iprod * w[i + 2] % MOD);
            }
            for (int i = 0; i <= ord2 - 2; i++) {
                rate2x[i] = mintx8::set1(rate2[i]);
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
template <u32 MOD> FFTInfo<MOD> fft_info = FFTInfo<MOD>();

template <u32 MOD> ModInt8<MOD> fft_single(ModInt8<MOD> x) {
    //static const u32 w4 = fft_info<MOD>.w[2], w8 = fft_info<MOD>.w[3];
    static const ModInt<MOD> w4 = fft_info<MOD>.w_mod[2], w8 = fft_info<MOD>.w_mod[3];

    static const ModInt8<MOD> step1 =
        ModInt8<MOD>(1, 1, 1, 1, 1, w8.val(), (w8 * w8).val(),
                      (w8 * w8 * w8).val());
    static const ModInt8<MOD> step2 = ModInt8<MOD>(1, 1, 1, w4.val(), 1, 1, 1, w4.val());

    x = (x.template neg<0b11110000>() + x.template shufflex4<0b01>()) * step1;
    x = (x.template neg<0b11001100>() + x.template shuffle<0b01001110>()) *
        step2;
    x = (x.template neg<0b10101010>() + x.template shuffle<0b10110001>());
    return x;
}

template <std::ranges::random_access_range R>
    requires is_modint8<std::ranges::range_value_t<R>>::value
void fft(R&& a) {
    using mintx8 = std::ranges::range_value_t<R>;
    static constexpr u32 MOD = mintx8::mod();

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
    while (h >= 2) {
        // 4-base
        const mintx8 w2 = mintx8::set1(fft_info<MOD>.w[2]);

        mintx8 rotx = mintx8::set1(1);
        for (int start = 0; start < n; start += (1 << h)) {
            mintx8 rot2x = rotx * rotx;
            mintx8 rot3x = rot2x * rotx;

            int len = 1 << (h - 2);
            for (int i = 0; i < len; i++) {
                auto a0 = a[start + 0 * len + i];
                auto a1 = a[start + 1 * len + i] * rotx;
                auto a2 = a[start + 2 * len + i] * rot2x;
                auto a3 = a[start + 3 * len + i] * rot3x;

                a[start + 0 * len + i] = (a0 + a2) + (a1 + a3);
                a[start + 1 * len + i] = (a0 + a2) - (a1 + a3);
                a[start + 2 * len + i] = (a0 - a2) + (a1 - a3) * w2;
                a[start + 3 * len + i] = (a0 - a2) - (a1 - a3) * w2;
            }
            rotx *= mintx8::set1(info.rot_shift8(8 * (start >> h)));
        }
        h -= 2;
    }

    {
        // fft each element
        mintx8 rotxi = mintx8::set1(1);
        for (int i = 0; i < n; i++) {
            a[i] = fft_single(a[i] * rotxi);
            rotxi *= info.rate4xi[std::countr_one((unsigned int)(i))];
        }
    }
}

template <u32 MOD> void ifft(std::vector<ModInt8<MOD>>& a) {
    using mintx8 = ModInt8<MOD>;

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
        mintx8 irotxi = mintx8::set1(1);
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
            irotxi *= info.irate4xi[std::countr_one((unsigned int)(i))];
        }
    }

    int h = 0;
    while (h + 2 <= lg) {
        // 4-base
        h += 2;

        mintx8 irotx = mintx8::set1(1);
        mintx8 iimagx = mintx8::set1(iw4);
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
            irotx *= mintx8::set1(
                info.irate3[std::countr_one((unsigned int)(start >> h))]);
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

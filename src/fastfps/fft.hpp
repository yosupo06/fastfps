#pragma once

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
    using modint8 = ModInt8<MOD>;

    static constexpr u32 g = primitive_root<MOD>;
    static constexpr int ord2 = std::countr_zero(MOD - 1);

    // w[i]^(2^i) == 1 : w[i] is the fft omega of n=2^i
    static constexpr std::array<modint, ord2 + 1> w = []() {
        std::array<modint, ord2 + 1> v;
        v[ord2] = modint(g).pow((MOD - 1) >> ord2);
        for (int i = ord2 - 1; i >= 0; i--) {
            v[i] = v[i + 1] * v[i + 1];
        }
        return v;
    }();
    static constexpr std::array<modint, ord2 + 1> iw = []() {
        std::array<modint, ord2 + 1> v;
        v[ord2] = w[ord2].inv();
        for (int i = ord2 - 1; i >= 0; i--) {
            v[i] = v[i + 1] * v[i + 1];
        }
        return v;
    }();

    static constexpr std::array<modint, ord2 + 1> rot8 = []() {
        std::array<modint, std::max(0, ord2 + 1)> v;
        for (int i = 3; i <= ord2; i++) {
            v[i] = w[i];
            for (int j = 3; j < i; j++) {
                v[i] *= iw[j];
            }
        }
        return v;
    }();
    static constexpr std::array<modint, ord2 + 1> irot8 = []() {
        std::array<modint, std::max(0, ord2 + 1)> v;
        for (int i = 3; i <= ord2; i++) {
            v[i] = iw[i];
            for (int j = 3; j < i; j++) {
                v[i] *= w[j];
            }
        }
        return v;
    }();
    // rot[i] * rot_shift8(i) = rot[i + 8]
    modint rot_shift8(u32 i) const { return rot8[std::countr_one(i >> 3) + 3]; }
    modint irot_shift8(u32 i) const {
        return irot8[std::countr_one(i >> 3) + 3];
    }

    static constexpr std::array<modint, ord2 + 1> rot4 = []() {
        std::array<modint, std::max(0, ord2 + 1)> v;
        for (int i = 4; i <= ord2; i++) {
            v[i] = w[i];
            for (int j = 4; j < i; j++) {
                v[i] *= iw[j];
            }
        }
        return v;
    }();
    static constexpr std::array<modint, ord2 + 1> irot4 = []() {
        std::array<modint, std::max(0, ord2 + 1)> v;
        for (int i = 4; i <= ord2; i++) {
            v[i] = iw[i];
            for (int j = 4; j < i; j++) {
                v[i] *= w[j];
            }
        }
        return v;
    }();
    std::array<modint8, ord2 + 1> rot16i = []() {
        std::array<modint8, std::max(0, ord2 + 1)> v;
        for (int i = 4; i <= ord2; i++) {
            std::array<modint, 8> buf;
            buf[0] = 1;
            for (int j = 1; j < 8; j++) {
                buf[j] = buf[j - 1] * rot4[i];
            }
            v[i] = modint8(buf);
        }
        return v;
    }();
    std::array<modint8, ord2 + 1> irot16i = []() {
        std::array<modint8, std::max(0, ord2 + 1)> v;
        for (int i = 4; i <= ord2; i++) {
            std::array<modint, 8> buf;
            buf[0] = 1;
            for (int j = 1; j < 8; j++) {
                buf[j] = buf[j - 1] * irot4[i];
            }
            v[i] = modint8(buf);
        }
        return v;
    }();
    // rot[i * j] * rot_shift16i(i)[j] = rot[(i + 8) * j]
    modint8 rot_shift16i(u32 i) const {
        return rot16i[std::countr_one(i >> 4) + 4];
    }
    modint8 irot_shift16i(u32 i) const {
        return irot16i[std::countr_one(i >> 4) + 4];
    }
};
template <u32 MOD> const FFTInfo<MOD> fft_info = FFTInfo<MOD>();

template <u32 MOD> ModInt8<MOD> fft_single(ModInt8<MOD> x) {
    static const FFTInfo<MOD>& info = fft_info<MOD>;
    static const ModInt<MOD> w4 = info.w[2], w8 = info.w[3];
    static const auto step4 = ModInt8<MOD>(1, 1, 1, w4, 1, 1, 1, w4);
    static const auto step8 =
        ModInt8<MOD>(1, 1, 1, 1, 1, w8, w8 * w8, w8 * w8 * w8);

    x = (blend<0b11110000>(x, -x) + x.permutevar({4, 5, 6, 7, 0, 1, 2, 3})) *
        step8;
    x = (blend<0b11001100>(x, -x) + x.permutevar({2, 3, 0, 1, 6, 7, 4, 5})) *
        step4;
    x = (blend<0b10101010>(x, -x) + x.permutevar({1, 0, 3, 2, 5, 4, 7, 6}));
    return x;
}

template <u32 MOD> ModInt8<MOD> ifft_single(ModInt8<MOD> x) {
    static const FFTInfo<MOD>& info = fft_info<MOD>;
    static const ModInt<MOD> iw4 = info.iw[2], iw8 = info.iw[3];
    static const auto step4 = ModInt8<MOD>(1, 1, 1, iw4, 1, 1, 1, iw4);
    static const auto step8 =
        ModInt8<MOD>(1, 1, 1, 1, 1, iw8, iw8 * iw8, iw8 * iw8 * iw8);

    x = (blend<0b10101010>(x, -x) + x.permutevar({1, 0, 3, 2, 5, 4, 7, 6})) *
        step4;
    x = (blend<0b11001100>(x, -x) + x.permutevar({2, 3, 0, 1, 6, 7, 4, 5})) *
        step8;
    x = (blend<0b11110000>(x, -x) + x.permutevar({4, 5, 6, 7, 0, 1, 2, 3}));

    return x;
}

template <std::ranges::random_access_range R>
    requires is_modint8<std::ranges::range_value_t<R>>::value
void fft(R&& a) {
    using modint8 = std::ranges::range_value_t<R>;
    static constexpr u32 MOD = modint8::mod();

    static const FFTInfo<MOD>& info = fft_info<MOD>;

    const int n = int(a.size());
    const int lg = std::countr_zero((u32)n);

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
        static const modint8 w2 = modint8::set1(info.w[2]);

        modint8 rotx = modint8::set1(1);
        for (int start = 0; start < n; start += (1 << h)) {
            const modint8 rot2x = rotx * rotx;
            const modint8 rot3x = rot2x * rotx;

            int len = 1 << (h - 2);
            for (int i = 0; i < len; i++) {
                auto a0 = a[start + 0 * len + i];
                auto a1 = a[start + 1 * len + i] * rotx;
                auto a2 = a[start + 2 * len + i] * rot2x;
                auto a3 = a[start + 3 * len + i] * rot3x;

                auto x = (a1 - a3) * w2;
                a[start + 0 * len + i] = (a0 + a2) + (a1 + a3);
                a[start + 1 * len + i] = (a0 + a2) - (a1 + a3);
                a[start + 2 * len + i] = (a0 - a2) + x;
                a[start + 3 * len + i] = (a0 - a2) - x;
            }
            rotx *= modint8::set1(info.rot_shift8(8 * (start >> h)));
        }
        h -= 2;
    }

    {
        // fft each element
        modint8 rotxi = modint8::set1(1);
        for (int i = 0; i < n; i++) {
            a[i] = fft_single(a[i] * rotxi);
            rotxi *= info.rot_shift16i(16 * i);
        }
    }
}

template <std::ranges::random_access_range R>
    requires is_modint8<std::ranges::range_value_t<R>>::value
void ifft(R&& a) {
    using modint8 = std::ranges::range_value_t<R>;
    static constexpr u32 MOD = modint8::mod();

    static const FFTInfo<MOD>& info = fft_info<MOD>;

    const int n = int(a.size());
    const int lg = std::countr_zero((u32)n);

    {
        // 8-base
        modint8 irotxi = modint8::set1(1);
        for (int i = 0; i < n; i++) {
            a[i] = ifft_single(a[i]) * irotxi;
            irotxi *= info.irot_shift16i(16 * i);
        }
    }

    int h = 0;
    while (h + 2 <= lg) {
        h += 2;

        // 4-base
        static const modint8 w2 = modint8::set1(info.iw[2]);

        modint8 rotx = modint8::set1(1);
        for (int start = 0; start < n; start += (1 << h)) {
            const auto rot2x = rotx * rotx;
            const auto rot3x = rot2x * rotx;
            int len = 1 << (h - 2);
            for (int i = 0; i < len; i++) {
                auto a0 = a[start + 0 * len + i];
                auto a1 = a[start + 1 * len + i];
                auto a2 = a[start + 2 * len + i];
                auto a3 = a[start + 3 * len + i];

                auto x0 = a0 + a1;
                auto x1 = a0 - a1;
                auto x2 = a2 + a3;
                auto x3 = (a2 - a3) * w2;

                a[start + 0 * len + i] = x0 + x2;
                a[start + 1 * len + i] = (x1 + x3) * rotx;
                a[start + 2 * len + i] = (x0 - x2) * rot2x;
                a[start + 3 * len + i] = (x1 - x3) * rot3x;
            }
            rotx *= modint8::set1(info.irot_shift8(8 * (start >> h)));
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

#pragma once

#include <immintrin.h>
#include <array>
#include <cassert>
#include <span>

#include "math.hpp"
#include "modint.hpp"
#include "types.hpp"

namespace fastfps {

template <u32 MOD> struct ModInt8 {
    using modint = ModInt<MOD>;
    using m256i_u = __m256i_u;

    static_assert(MOD % 2 && MOD <= (1U << 30) - 1,
                  "mod must be odd and at most 2^30 - 1");

    static constexpr u32 mod() { return MOD; }

    ModInt8() : x(_mm256_setzero_si256()) {}
    ModInt8(std::span<const i32, 8> _x)
        : x(mul(_mm256_sub_epi32(_mm256_loadu_si256((m256i_u*)_x.data()),
                                 INT_MIN_X),
                _mm256_set1_epi32(B2))) {
        static const ModInt8 OFFSET = set1(u32(1) << 31);
        *this -= OFFSET;
    }
    ModInt8(std::span<const u32, 8> _x)
        : x(mul(_mm256_loadu_si256((m256i_u*)_x.data()),
                _mm256_set1_epi32(B2))) {}
    ModInt8(std::span<const modint, 8> _x)
        : x(_mm256_loadu_si256((m256i_u*)_x.data())) {
            static_assert(sizeof(modint) == 4);
        }
    explicit ModInt8(modint x0,
                     modint x1,
                     modint x2,
                     modint x3,
                     modint x4,
                     modint x5,
                     modint x6,
                     modint x7)
        : x(_mm256_set_epi32(x7.internal_val(),
                             x6.internal_val(),
                             x5.internal_val(),
                             x4.internal_val(),
                             x3.internal_val(),
                             x2.internal_val(),
                             x1.internal_val(),
                             x0.internal_val())) {}

    static ModInt8 set1(modint x) {
        ModInt8 v;
        v.x = _mm256_set1_epi32(x.internal_val());
        return v;
    }

    std::array<u32, 8> val() const {
        auto a = mul(x, _mm256_set1_epi32(1));
        alignas(32) std::array<u32, 8> b;
        _mm256_storeu_si256((__m256i_u*)b.data(), min(a, _mm256_sub_epi32(a, MOD_X)));
        return b;
    }

    ModInt8& operator+=(const ModInt8& rhs) {
        x = _mm256_add_epi32(x, rhs.x);
        x = min(x, _mm256_sub_epi32(x, MOD2_X));
        return *this;
    }
    friend ModInt8 operator+(const ModInt8& lhs, const ModInt8& rhs) {
        return ModInt8(lhs) += rhs;
    }

    ModInt8& operator-=(const ModInt8& rhs) {
        x = _mm256_sub_epi32(x, rhs.x);
        x = min(x, _mm256_add_epi32(x, MOD2_X));
        return *this;
    }
    friend ModInt8 operator-(const ModInt8& lhs, const ModInt8& rhs) {
        return ModInt8(lhs) -= rhs;
    }

    ModInt8& operator*=(const ModInt8& rhs) {
        x = mul(x, rhs.x);
        return *this;
    }
    friend ModInt8 operator*(const ModInt8& lhs, const ModInt8& rhs) {
        return ModInt8(lhs) *= rhs;
    }

    ModInt8 operator-() const { return ModInt8() - *this; }

    friend bool operator==(const ModInt8& lhs, const ModInt8& rhs) {
        auto lx = lhs.x, rx = rhs.x;
        lx = min(lx, _mm256_sub_epi32(lx, MOD_X));
        rx = min(rx, _mm256_sub_epi32(rx, MOD_X));
        auto z = _mm256_xor_si256(lx, rx);
        return _mm256_testz_si256(z, z);
    }

    // a.permutevar(idx)[i] = a[idx[i] % 8]
    ModInt8 permutevar(const std::array<u32, 8>& idx) const {
        return permutevar(_mm256_loadu_si256((m256i_u*)idx.data()));
    }

    // a[i] <- a[(middle + i) % 8]
    ModInt8 rotate(u32 middle) const {
        static const m256i_u base = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
        return permutevar(_mm256_add_epi32(base, _mm256_set1_epi32(middle)));
    }

    template <uint8_t MASK>
    friend ModInt8 blend(const ModInt8& lhs, const ModInt8& rhs) {
        ModInt8 v;
        v.x = _mm256_blend_epi32(lhs.x, rhs.x, MASK);
        return v;
    }

    friend ModInt8 blendvar(const ModInt8& lhs,
                            const ModInt8& rhs,
                            const std::array<u32, 8>& idx) {
        ModInt8 v;
        v.x = _mm256_blendv_epi8(
            rhs.x, lhs.x,
            _mm256_cmpeq_epi32(_mm256_loadu_si256((m256i_u*)idx.data()),
                               _mm256_setzero_si256()));
        return v;
    }

  private:
    m256i_u x;

    inline static const m256i_u MOD_X = _mm256_set1_epi32(MOD);
    inline static const m256i_u MOD2_X = _mm256_set1_epi32(2 * MOD);
    inline static const m256i_u N_INV_X = _mm256_set1_epi32(-inv_u32(MOD));
    inline static const m256i_u INT_MIN_X = _mm256_set1_epi32(u32(1) << 31);

    static constexpr u32 B2 = pow_mod_constexpr(2, 64, MOD);
    inline static const m256i_u B2_X = _mm256_set1_epi32(B2);

    // Input: l * r <= 2^32 * MOD
    // Output: l * r >>= 2^32
    static m256i_u mul(const m256i_u& l, const m256i_u& r) {
        auto x0 = mul_even(l, r);
        auto x1 = mul_even(_mm256_shuffle_epi32(l, 0xf5),
                           _mm256_shuffle_epi32(r, 0xf5));
        x0 += mul_even(mul_even(x0, N_INV_X), MOD_X);
        x1 += mul_even(mul_even(x1, N_INV_X), MOD_X);

        x0 = _mm256_srli_epi64(x0, 32);
        return _mm256_blend_epi32(x0, x1, 0b10101010);
    }
    // (lr[0], lr[2], lr[4], lr[6])
    static m256i_u mul_even(const m256i_u& l, const m256i_u& r) {
        return _mm256_mul_epu32(l, r);
    }

    static m256i_u min(const m256i_u& l, const m256i_u& r) {
        return _mm256_min_epu32(l, r);
    }
    static m256i_u max(const m256i_u& l, const m256i_u& r) {
        return _mm256_max_epu32(l, r);
    }

    // a.permutevar(idx)[i] = a[idx[i] % 8]
    ModInt8 permutevar(const m256i_u& idx) const {
        ModInt8 v;
        v.x = _mm256_permutevar8x32_epi32(x, idx);
        return v;
    }
};

template <typename T> struct is_modint8 : std::false_type {};
template <u32 MOD> struct is_modint8<ModInt8<MOD>> : std::true_type {};

}  // namespace fastfps

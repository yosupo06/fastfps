#pragma once

#include <array>

#include "simd.hpp"
#include "types.hpp"
#include "modint.hpp"

namespace fastfps {

template <u32 MOD> struct ModInt8 {
    static_assert(MOD % 2 && MOD <= (1U << 30) - 1,
                  "mod must be odd and at most 2^30 - 1");

    static constexpr u32 mod() { return MOD; }

    inline static const u32x8 MOD_X = u32x8::set1(MOD);

    static const u32x8 modx() { return u32x8::set1(MOD); }
    static const u32x8 n_inv() { return u32x8::set1(-inv_u32(MOD)); }

    static constexpr u32 inv = -inv_u32(MOD);
    
    static constexpr uint32_t B = ((1ull << 32)) % MOD;
    static constexpr uint32_t B2 = 1ull * B * B % MOD;

    ModInt8() {}
    ModInt8(u32x8 _x) : x(mul(_x, u32x8::set1(B2))) {}
//    ModInt8(ModInt<MOD> _x) : x(u32x8::set1(_x.internal_val())) {}
    ModInt8(u32 x0, u32 x1, u32 x2, u32 x3, u32 x4, u32 x5, u32 x6, u32 x7)
        : ModInt8(u32x8(x0, x1, x2, x3, x4, x5, x6, x7)) {}
    static ModInt8 set1(ModInt<MOD> x) {
        ModInt8 v;
        v.x = u32x8::set1(x.internal_val());
        return v;
    }

    std::array<u32, 8> to_array() const {
        auto a = mul(x, u32x8::set1(1)).to_array();
        // TODO: optimize
        std::array<u32, 8> b;
        for (int i = 0; i < 8; i++) {
            b[i] = a[i] % MOD;
        }
        return b;
    }

    ModInt8& operator+=(const ModInt8& rhs) {
        x += rhs.x;
        x = min(x, x - u32x8::set1(2 * MOD));
        return *this;
    }
    friend ModInt8 operator+(const ModInt8& lhs, const ModInt8& rhs) {
        return ModInt8(lhs) += rhs;
    }

    ModInt8& operator-=(const ModInt8& rhs) {
        x -= rhs.x;
        x = min(x, x + u32x8::set1(2 * MOD));
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

    template <int N> ModInt8 neg() const {
        ModInt8 v;
        v.x = x.blend<N>(u32x8::set1(2 * MOD) - x);
        return v;
    }

    template <int N> ModInt8 shuffle() const {
        // TODO: avoid to use intrincs directly
        ModInt8 v;
        v.x = _mm256_shuffle_epi32(x.as_m256i_u(), N);
        return v;
    }
    template <int N> ModInt8 shufflex4() const {
        // TODO: avoid to use intrincs directly
        ModInt8 v;
        v.x = _mm256_permute2x128_si256(x.as_m256i_u(), x.as_m256i_u(), N);
        return v;
    }

    friend bool operator==(const ModInt8& lhs, const ModInt8& rhs) {
        // TODO: optimize
        return lhs.to_array() == rhs.to_array();
    }

    // a.permutevar(idx)[i] = a[idx[i] % 8]
    ModInt8 permutevar(u32x8 idx) const {
        ModInt8 v;
        v.x = x.permutevar(idx);
        return v;
    }

  private:
    u32x8 x;

    // Input: l * r <= 2^32 * MOD
    // Output: l * r >>= 2^32
    static u32x8 mul(const u32x8& l, const u32x8& r) {
        auto x0 = mul0(l, r);
        auto x1 = mul1(l, r);
        x0 += mul0(mul0(x0.to_u32x8(), n_inv()).to_u32x8(), MOD_X);
        x1 += mul0(mul0(x1.to_u32x8(), n_inv()).to_u32x8(), MOD_X);
        return x1.to_u32x8().blend<0b01010101>(x0.rshift<32>().to_u32x8());
    }
};

template <typename T> struct is_modint8 : std::false_type {};
template <u32 MOD> struct is_modint8<ModInt8<MOD>> : std::true_type {};

}  // namespace fastfps

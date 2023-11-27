#pragma once

#include <array>

#include "modint.hpp"
#include "simd.hpp"
#include "types.hpp"

namespace fastfps {

template <u32 MOD> struct ModInt8 {
    using modint = ModInt<MOD>;

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
    ModInt8(modint x0,
            modint x1,
            modint x2,
            modint x3,
            modint x4,
            modint x5,
            modint x6,
            modint x7)
        : x(u32x8(x0.internal_val(),
                  x1.internal_val(),
                  x2.internal_val(),
                  x3.internal_val(),
                  x4.internal_val(),
                  x5.internal_val(),
                  x6.internal_val(),
                  x7.internal_val())) {}
    ModInt8(std::array<modint, 8> _x)
        : ModInt8(_x[0], _x[1], _x[2], _x[3], _x[4], _x[5], _x[6], _x[7]) {}

    static ModInt8 set1(modint x) {
        ModInt8 v;
        v.x = u32x8::set1(x.internal_val());
        return v;
    }

    std::array<u32, 8> val() const {
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

    ModInt8 operator-() const { return ModInt8() - *this; }

    friend bool operator==(const ModInt8& lhs, const ModInt8& rhs) {
        // TODO: optimize
        return lhs.val() == rhs.val();
    }

    // a.permutevar(idx)[i] = a[idx[i] % 8]
    ModInt8 permutevar(u32x8 idx) const {
        ModInt8 v;
        v.x = x.permutevar(idx);
        return v;
    }

    template <uint8_t MASK>
    friend ModInt8 blend(const ModInt8& lhs, const ModInt8& rhs) {
        ModInt8 v;
        v.x = blend<MASK>(lhs.x, rhs.x);
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
        return blend<0b01010101>(x1.to_u32x8(), x0.rshift<32>().to_u32x8());
    }
};

template <typename T> struct is_modint8 : std::false_type {};
template <u32 MOD> struct is_modint8<ModInt8<MOD>> : std::true_type {};

}  // namespace fastfps

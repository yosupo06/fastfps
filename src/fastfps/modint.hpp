#pragma once

#include <array>
#include <cassert>
#include <iostream>

#include "fastfps/types.hpp"

namespace fastfps {

// x * inv_u32(x) = 1 (mod 2^32)
constexpr u32 inv_u32(const u32 x) {
    assert(x % 2);
    u32 inv = 1;
    for (int i = 0; i < 5; i++) {
        inv *= 2u - inv * x;
    }
    return inv;
}

template <u32 MOD> struct ModInt {
    static_assert(MOD % 2 && MOD <= (1U << 30) - 1,
                  "mod must be odd and at most 2^30 - 1");

    static constexpr u32 mod() { return MOD; }

    constexpr ModInt() : x(0) {}
    constexpr explicit ModInt(u32 _x) : x(mulreduce(_x, B2)) {}

    constexpr ModInt(std::signed_integral auto _x)
        : ModInt((u32)(_x % (i32)MOD + MOD)) {}
    constexpr ModInt(std::unsigned_integral auto _x)
        : ModInt((u32)(_x % MOD)) {}

    constexpr u32 val() const {
        u32 y = mulreduce(x, 1);
        return y < MOD ? y : y - MOD;
    }
    constexpr u32 internal_val() const { return x; }

    constexpr ModInt& operator+=(const ModInt& rhs) {
        x += rhs.x;
        x = std::min(x, x - 2 * MOD);
        return *this;
    }
    constexpr friend ModInt operator+(const ModInt& lhs, const ModInt& rhs) {
        return ModInt(lhs) += rhs;
    }

    constexpr ModInt& operator-=(const ModInt& rhs) {
        x += 2 * MOD - rhs.x;
        x = std::min(x, x - 2 * MOD);
        return *this;
    }
    constexpr friend ModInt operator-(const ModInt& lhs, const ModInt& rhs) {
        return ModInt(lhs) -= rhs;
    }

    constexpr ModInt& operator*=(const ModInt& rhs) {
        x = mulreduce(x, rhs.x);
        return *this;
    }
    constexpr friend ModInt operator*(const ModInt& lhs, const ModInt& rhs) {
        return ModInt(lhs) *= rhs;
    }

    friend bool operator==(const ModInt& lhs, const ModInt& rhs) {
        auto lx = lhs.x;
        if (lx >= MOD) lx -= MOD;
        auto rx = rhs.x;
        if (rx >= MOD) rx -= MOD;
        return lx == rx;
    }

    constexpr ModInt pow(u64 n) const {
        ModInt v = *this, r = 1;
        while (n) {
            if (n & 1) r *= v;
            v *= v;
            n >>= 1;
        }
        return r;
    }
    constexpr ModInt inv() const {
        // TODO: for non-prime
        return pow(MOD - 2);
    }

    friend std::ostream& operator<<(std::ostream& os, const ModInt& v) {
        return os << v.val();
    }

  private:
    u32 x;

    static constexpr u32 B = ((u64(1) << 32)) % MOD;
    static constexpr u32 B2 = u64(1) * B * B % MOD;
    static constexpr u32 INV = -inv_u32(MOD);

    // Input: (l * r) must be no more than (2^32 * MOD)
    // Output: ((l * r) / 2^32) % MOD
    static constexpr u32 mulreduce(u32 l, u32 r) {
        u64 x = u64(1) * l * r;
        x += u64(u32(x) * INV) * MOD;
        return u32(x >> 32);
    }
};

template <typename T> struct is_modint : std::false_type {};
template <u32 MOD> struct is_modint<ModInt<MOD>> : std::true_type {};

}  // namespace fastfps

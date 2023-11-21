#pragma once

#include <array>

#include "simd.hpp"
#include "types.hpp"

namespace fastfps {

constexpr uint32_t inv_2n32(const uint32_t x) {
    uint32_t inv = 1;
    for (int i = 0; i < 5; i++) {
        inv *= 2u - inv * x;
    }
    return inv;
}

template <uint32_t MOD> struct modintx8 {
    static_assert(MOD % 2 && MOD <= (1U << 30) - 1,
                  "mod must be odd and at most 2^30 - 1");

    static const u32x8 mod() { return u32x8(MOD); }
    static const u32x8 n_inv() { return u32x8(-inv_2n32(MOD)); }

    static constexpr uint32_t B = ((1ull << 32)) % MOD;
    static constexpr uint32_t B2 = 1ull * B * B % MOD;

    modintx8() {}
    modintx8(u32x8 _x) : x(mul(_x, B2)) {}
    modintx8(std::array<u32, 8> _x) : x(mul(u32x8(_x), B2)) {}
    modintx8(u32 x0, u32 x1, u32 x2, u32 x3, u32 x4, u32 x5, u32 x6, u32 x7)
        : modintx8(u32x8(x0, x1, x2, x3, x4, x5, x6, x7)) {}

    std::array<u32, 8> to_array() const {
        auto a = mul(x, u32x8(1)).to_array();
        // TODO: optimize
        std::array<u32, 8> b;
        for (int i = 0; i < 8; i++) {
            b[i] = a[i] % MOD;
        }
        return b;
    }

    modintx8& operator+=(const modintx8& rhs) {
        x += rhs.x;
        x = min(x, x - u32x8(2 * MOD));
        return *this;
    }
    friend modintx8 operator+(const modintx8& lhs, const modintx8& rhs) {
        return modintx8(lhs) += rhs;
    }

    modintx8& operator-=(const modintx8& rhs) {
        x += u32x8(2 * MOD) - rhs.x;
        x = min(x, x - u32x8(2 * MOD));
        return *this;
    }
    friend modintx8 operator-(const modintx8& lhs, const modintx8& rhs) {
        return modintx8(lhs) -= rhs;
    }

    modintx8& operator*=(const modintx8& rhs) {
        x = mul(x, rhs.x);
        return *this;
    }
    friend modintx8 operator*(const modintx8& lhs, const modintx8& rhs) {
        return modintx8(lhs) *= rhs;
    }

    template <int N> modintx8 neg() const {
        modintx8 v;
        v.x = x.blend<N>(u32x8(2 * MOD) - x);
        return v;
    }


    template <int N> modintx8 shuffle() const {
        // TODO: avoid to use intrincs directly
        modintx8 v;
        v.x = _mm256_shuffle_epi32(x.as_m256i_u(), N);
        return v;
    }
    template <int N> modintx8 shufflex4() const {
        // TODO: avoid to use intrincs directly
        modintx8 v;
        v.x = _mm256_permute2x128_si256(x.as_m256i_u(), x.as_m256i_u(), N);
        return v;
    }

    friend bool operator==(const modintx8& lhs, const modintx8& rhs) {
        // TODO: optimize
        return lhs.to_array() == rhs.to_array();
    }

  private:
    u32x8 x;

    static u32x8 mul(u32x8 l, u32x8 r) {
        auto x0 = mul0(l, r);
        auto x1 = mul1(l, r);
        x0 += mul0(mul0(x0.to_u32x8(), n_inv()).to_u32x8(), u32x8(MOD));
        x1 += mul0(mul0(x1.to_u32x8(), n_inv()).to_u32x8(), u32x8(MOD));
        return x1.to_u32x8().blend<0b01010101>(x0.rshift<32>().to_u32x8());
    }
};

}  // namespace fastfps
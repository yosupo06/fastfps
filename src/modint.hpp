#pragma once

#include "simd.hpp"

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

    std::array<u32, 8> to_array() const {
        auto a = mul(x, u32x8(1)).to_array();
        // TODO: optimize
        std::array<u32, 8> b;
        for (int i = 0; i < 8; i++) {
            b[i] = a[i] % MOD;
        }
        return b;
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
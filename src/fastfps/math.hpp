#pragma once

#include "types.hpp"

namespace fastfps {

constexpr u32 pow_mod_constexpr(u64 x, u64 n, u32 m) {
    if (m == 1) return 0;
    x %= m;

    u64 r = 1;
    while (n) {
        if (n & 1) r = (r * x) % m;
        x = (x * x) % m;
        n >>= 1;
    }

    return (u32)(r);
}

constexpr u32 primitive_root_constexpr(u32 m) {
    if (m == 2) return 1;

    u32 divs[20] = {};
    int cnt = 0;
    u32 x = (m - 1) / 2;
    for (int i = 2; (u64)(i)*i <= x; i += 2) {
        if (x % i == 0) {
            divs[cnt++] = i;
            while (x % i == 0) {
                x /= i;
            }
        }
    }
    if (x > 1) {
        divs[cnt++] = x;
    }
    for (u32 g = 2;; g++) {
        bool ok = true;
        for (int i = 0; i < cnt; i++) {
            if (pow_mod_constexpr(g, (m - 1) / divs[i], m) == 1) {
                ok = false;
                break;
            }
        }
        if (ok) return g;
    }
}
template <u32 m> constexpr u32 primitive_root = primitive_root_constexpr(m);

}  // namespace fastfps

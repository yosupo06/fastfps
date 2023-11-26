#pragma once

#include <immintrin.h>
#include <array>
#include <cstdint>

#include "types.hpp"

namespace fastfps {

struct u64x4;

struct u32x8 {
    u32x8() : x(_mm256_setzero_si256()) {}
    u32x8(const __m256i_u& _x) : x(_x) {}
    u32x8(const std::array<u32, 8>& _x)
        : x(_mm256_loadu_si256((__m256i_u*)_x.data())) {}
    u32x8(u32 x0, u32 x1, u32 x2, u32 x3, u32 x4, u32 x5, u32 x6, u32 x7)
        : x(_mm256_set_epi32(x7, x6, x5, x4, x3, x2, x1, x0)) {}

    static u32x8 set1(u32 x) { return u32x8(_mm256_set1_epi32(x)); }

    template <uint8_t MASK>
    friend u32x8 blend(const u32x8& lhs, const u32x8& rhs) {
        return _mm256_blend_epi32(lhs.x, rhs.x, MASK);
    }

    __m256i_u as_m256i_u() const { return x; }
    std::array<uint32_t, 8> to_array() const {
        alignas(32) std::array<uint32_t, 8> b;
        _mm256_store_si256((__m256i_u*)b.data(), x);
        return b;
    }
    uint32_t at(int i) const {
        // TODO: optimize
        return to_array()[i];
    }
    void set(int i, uint32_t xi) {
        auto b = to_array();
        b[i] = xi;
        x = _mm256_load_si256((__m256i_u*)b.data());
    }

    u32x8& operator+=(const u32x8& rhs) {
        x = _mm256_add_epi32(x, rhs.x);
        return *this;
    }
    friend u32x8 operator+(const u32x8& lhs, const u32x8& rhs) {
        return u32x8(lhs) += rhs;
    }
    u32x8& operator-=(const u32x8& rhs) {
        x = _mm256_sub_epi32(x, rhs.x);
        return *this;
    }
    friend u32x8 operator-(const u32x8& lhs, const u32x8& rhs) {
        return u32x8(lhs) -= rhs;
    }
    u32x8& operator*=(const u32x8& rhs) {
        x = _mm256_mullo_epi32(x, rhs.x);
        return *this;
    }
    friend u32x8 operator*(const u32x8& lhs, const u32x8& rhs) {
        return u32x8(lhs) *= rhs;
    }

    u32x8 operator-() const {
        return u32x8() - *this;
    }

    friend bool operator==(const u32x8& lhs, const u32x8& rhs) {
        u32 bitmask = _mm256_movemask_epi8(_mm256_cmpeq_epi32(lhs.x, rhs.x));
        return (bitmask == -1u);
    }

    // a.permutevar(idx)[i] = a[idx[i] % 8]
    u32x8 permutevar(u32x8 idx) const {
        return _mm256_permutevar8x32_epi32(x, idx.as_m256i_u());
    }

  private:
    __m256i_u x;
};

struct u64x4 {
    u64x4() : x(_mm256_setzero_si256()) {}
    u64x4(const __m256i_u& _x) : x(_x) {}
    u64x4(const std::array<uint64_t, 4>& _x)
        : x(_mm256_loadu_si256((__m256i_u*)_x.data())) {}
    u64x4(uint64_t x0, uint64_t x1, uint64_t x2, uint64_t x3)
        : x(_mm256_set_epi64x(x3, x2, x1, x0)) {}

    static u64x4 set1(u64 x) { return u64x4(_mm256_set1_epi64x(x)); }

    u32x8 to_u32x8() const { return x; }
    std::array<uint64_t, 4> to_array() const {
        alignas(32) std::array<uint64_t, 4> b;
        _mm256_store_si256((__m256i*)b.data(), x);
        return b;
    }
    uint64_t at(int i) const {
        // TODO: optimize
        return to_array()[i];
    }
    void set(int i, uint64_t xi) {
        auto b = to_array();
        b[i] = xi;
        x = _mm256_load_si256((__m256i*)b.data());
    }

    u64x4& operator+=(const u64x4& rhs) {
        x = _mm256_add_epi64(x, rhs.x);
        return *this;
    }
    friend u64x4 operator+(const u64x4& lhs, const u64x4& rhs) {
        return u64x4(lhs) += rhs;
    }
    u64x4& operator-=(const u64x4& rhs) {
        x = _mm256_sub_epi64(x, rhs.x);
        return *this;
    }
    friend u64x4 operator-(const u64x4& lhs, const u64x4& rhs) {
        return u64x4(lhs) -= rhs;
    }

    friend bool operator==(const u64x4& lhs, const u64x4& rhs) {
        u32 bitmask = _mm256_movemask_epi8(_mm256_cmpeq_epi32(lhs.x, rhs.x));
        return (bitmask == -1u);
    }

    template <int N> u64x4 rshift() const { return _mm256_srli_epi64(x, N); }

  private:
    __m256i_u x;
};

// (x0, x2, x4, x6)
inline u64x4 mul0(const u32x8& l, const u32x8& r) {
    return _mm256_mul_epu32(l.as_m256i_u(), r.as_m256i_u());
}
// (x1, x3, x5, x7)
inline u64x4 mul1(const u32x8& l, const u32x8& r) {
    return _mm256_mul_epu32(_mm256_shuffle_epi32(l.as_m256i_u(), 0xf5),
                            _mm256_shuffle_epi32(r.as_m256i_u(), 0xf5));
}

inline u32x8 min(const u32x8& l, const u32x8& r) {
    return _mm256_min_epu32(l.as_m256i_u(), r.as_m256i_u());
}

inline u32x8 max(const u32x8& l, const u32x8& r) {
    return _mm256_max_epu32(l.as_m256i_u(), r.as_m256i_u());
}

}  // namespace fastfps

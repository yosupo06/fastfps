#pragma once

#include <algorithm>
#include <random>
#include <vector>

#include "fft.hpp"
#include "modint.hpp"
#include "modint8.hpp"

namespace fastfps {

template <int MOD> struct ModVec {
    using modint = ModInt<MOD>;
    using modint8 = ModInt8<MOD>;

  public:
    ModVec() : n(0), v() {}

    ModVec(const std::vector<modint>& _v) : n(std::ssize(_v)), v(vsize(n)) {
        for (int i = 0; i < vsize(n); i++) {
            std::array<modint, 8> buf;
            for (int j = 0; j < 8 && (i * 8 + j) < n; j++) {
                buf[j] = _v[i * 8 + j];
            }
            v[i] = buf;
        }
    }

    std::vector<u32> val() const {
        std::vector<u32> _v(n);
        for (int i = 0; i < std::ssize(v); i++) {
            std::array<u32, 8> buf = v[i].val();
            for (int j = 0; j < 8 && (i * 8 + j) < n; j++) {
                _v[j] = buf[i * 8 + j];
            }
        }
        return _v;
    }

    ModVec& operator+=(const ModVec& rhs) {
        n = std::max(n, rhs.n);
        if (size(v) < size(rhs.v)) {
            v.resize(size(rhs.v));
        }
        for (int i = 0; i < ssize(v); i++) {
            v[i] += rhs.v[i];
        }
        return *this;
    }
    friend ModVec operator+(const ModVec& lhs, const ModVec& rhs) {
        return ModVec(lhs) += rhs;
    }

    ModVec& operator-=(const ModVec& rhs) {
        n = std::max(n, rhs.n);
        if (size(v) < size(rhs.v)) {
            v.resize(size(rhs.v));
        }
        for (int i = 0; i < ssize(v); i++) {
            v[i] -= rhs.v[i];
        }
        return *this;
    }
    friend ModVec operator-(const ModVec& lhs, const ModVec& rhs) {
        return ModVec(lhs) -= rhs;
    }

    friend bool operator==(const ModVec& lhs, const ModVec& rhs) {
        return lhs.n == rhs.n && lhs.v == rhs.v;
    }

    ModVec& operator*=(const ModVec& rhs) {
        if (n == 0 || rhs.n == 0) {
            n = 0;
            v.clear();
            return *this;
        }
        n += rhs.n - 1;
        int n2 = 1 << std::bit_ceil((size_t)n);
        auto rv = rhs.v;
        v.resize(n2);
        rv.resize(n2);

        fft(v);
        fft(rv);
        for (int i = 0; i < n2; i++) {
            v[i] *= rv[i];
        }
        ifft(v);

        v.resize(vsize(n));

        modint8 inv = modint8::set1(modint(8 * n2).inv());
        for (int i = 0; i < n; i++) {
            v[i] *= inv;
        }
        return *this;
    }
    friend ModVec operator*(const ModVec& lhs, const ModVec& rhs) {
        return ModVec(lhs) *= rhs;
    }

  private:
    ssize_t n;
    std::vector<modint8> v;

    static ssize_t vsize(ssize_t n) { return (n + 7) / 8; }
};

}  // namespace fastfps

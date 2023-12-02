#pragma once

#include <algorithm>
#include <random>
#include <vector>

#include "fastfps/fft.hpp"
#include "fastfps/modint.hpp"
#include "fastfps/modint8.hpp"

namespace fastfps {

template <int MOD> struct ModVec {
    using modint = ModInt<MOD>;
    using modint8 = ModInt8<MOD>;

  public:
    ModVec() : n(0), v() {}
    explicit ModVec(ssize_t _n) : n(_n), v(vsize(_n)) {}
    ModVec(std::initializer_list<u32> li) : n(ssize(li)), v(vsize(n)) {
        auto it = li.begin();
        for (int i = 0; i < ssize(v); i++) {
            std::array<u32, 8> buf = {};
            for (int j = 0; j < 8 && (i * 8 + j) < n; j++) {
                buf[j] = *it;
                it++;
            }
            v[i] = modint8(buf);
        }
    }
    ModVec(std::initializer_list<i32> li) : n(ssize(li)), v(vsize(n)) {
        auto it = li.begin();
        for (int i = 0; i < ssize(v); i++) {
            std::array<i32, 8> buf = {};
            for (int j = 0; j < 8 && (i * 8 + j) < n; j++) {
                buf[j] = *it;
                it++;
            }
            v[i] = modint8(buf);
        }
    }

    ModVec(const std::vector<modint>& _v) : n(std::ssize(_v)), v(vsize(n)) {
        for (int i = 0; i < std::ssize(v); i++) {
            std::array<modint, 8> buf{};
            for (int j = 0; j < 8 && (i * 8 + j) < n; j++) {
                buf[j] = _v[i * 8 + j];
            }
            v[i] = modint8(buf);
        }
    }
    ModVec(const std::vector<u32>& _v) : n(std::ssize(_v)), v(vsize(n)) {
        for (int i = 0; i < std::ssize(v); i++) {
            std::array<u32, 8> buf{};
            for (int j = 0; j < 8 && (i * 8 + j) < n; j++) {
                buf[j] = _v[i * 8 + j];
            }
            v[i] = modint8(buf);
        }
    }

    size_t size() const { return n; }

    std::vector<u32> val() const {
        std::vector<u32> _v(n);
        for (int i = 0; i < std::ssize(v); i++) {
            std::array<u32, 8> buf = v[i].val();
            for (int j = 0; j < 8 && (i * 8 + j) < n; j++) {
                _v[i * 8 + j] = buf[j];
            }
        }
        return _v;
    }
    u32 val(ssize_t index) const {
        if (index < 0 || n <= index) return 0;
        // todo: optimzie
        return v[index / 8].val()[index % 8];
    }

    void resize(ssize_t sz) {
        n = sz;
        v.resize(vsize(n));
        clear_last();
        return;
    }

    ModVec& operator+=(const ModVec& rhs) {
        n = std::max(n, rhs.n);
        if (std::size(v) < std::size(rhs.v)) {
            v.resize(std::size(rhs.v));
        }
        for (int i = 0; i < std::ssize(rhs.v); i++) {
            v[i] += rhs.v[i];
        }
        return *this;
    }
    friend ModVec operator+(const ModVec& lhs, const ModVec& rhs) {
        return ModVec(lhs) += rhs;
    }

    ModVec& operator-=(const ModVec& rhs) {
        n = std::max(n, rhs.n);
        if (std::size(v) < std::size(rhs.v)) {
            v.resize(std::size(rhs.v));
        }
        for (int i = 0; i < std::ssize(rhs.v); i++) {
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
        auto rv = rhs.v;

        n += rhs.n - 1;

        ssize_t v_up = (ssize_t)std::bit_ceil((size_t)vsize(n));
        v.resize(v_up);
        rv.resize(v_up);
        fft(v);
        fft(rv);
        for (int i = 0; i < v_up; i++) {
            v[i] *= rv[i];
        }
        ifft(v);

        v.resize(vsize(n));

        modint8 inv = modint8::set1(modint(8 * v_up).inv());
        for (auto& x : v) x *= inv;
        return *this;
    }
    friend ModVec operator*(const ModVec& lhs, const ModVec& rhs) {
        return ModVec(lhs) *= rhs;
    }

    ModVec& operator*=(const modint& rhs) {
        modint8 r = modint8::set1(rhs);
        for (auto& x : v) x *= r;
        return *this;
    }
    friend ModVec operator*(const ModVec& lhs, const modint& rhs) {
        return ModVec(lhs) *= rhs;
    }
    friend ModVec operator*(const modint& lhs, const ModVec& rhs) {
        return ModVec(rhs) *= lhs;
    }

    // dst[dst_start .. dst_start + len) = this[start .. start + len)
    void copy_to(ssize_t start,
                 ssize_t len,
                 ModVec& dst,
                 ssize_t dst_start) const {
        // TODO: be able to self move
        assert(0 <= start && start + len <= n);
        assert(0 <= dst_start && dst_start + len <= dst.n);
        if (len == 0) return;

        auto succ = [&](ssize_t len2) {
            start += len2;
            dst_start += len2;
            len -= len2;
        };
        if (start % 8 == dst_start % 8) {
            {
                ssize_t len2 = std::min(len, 8 - dst_start % 8);
                const auto mask = [&]() {
                    std::array<u32, 8> b;
                    for (int i = 0; i < 8; i++) {
                        b[i] = (dst_start % 8 <= i && i < dst_start % 8 + len2);
                    }
                    return b;
                }();
                dst.v[dst_start / 8] =
                    blendvar(dst.v[dst_start / 8], v[start / 8], mask);
                succ(len2);
            }
            if (len == 0) return;
            assert(start % 8 == 0 && dst_start % 8 == 0);
            while (len >= 8) {
                // TODO: use std::copy
                dst.v[dst_start / 8] = v[start / 8];
                succ(8);
            }
            if (len == 0) return;
            {
                ssize_t len2 = len;
                const auto mask = [&]() {
                    std::array<u32, 8> b;
                    for (int i = 0; i < 8; i++) {
                        b[i] = (i < len2);
                    }
                    return b;
                }();
                dst.v[dst_start / 8] =
                    blendvar(dst.v[dst_start / 8], v[start / 8], mask);
                succ(len2);
            }
        } else {
            const ssize_t shift = (start + 8 - dst_start % 8) % 8;
            const auto blend_mask = [&]() {
                std::array<u32, 8> b;
                for (int i = 0; i < 8; i++) {
                    b[i] = (8 - shift <= i);
                }
                return b;
            }();
            {
                ssize_t len2 = std::min(len, 8 - dst_start % 8);
                const auto mask = [&]() {
                    std::array<u32, 8> b;
                    for (int i = 0; i < 8; i++) {
                        b[i] = (dst_start % 8 <= i && i < dst_start % 8 + len2);
                    }
                    return b;
                }();
                auto x = v[start / 8].rotate((u32)shift);
                if (len2 > 8 - start % 8) {
                    x = blendvar(x, v[start / 8 + 1].rotate((u32)(shift)),
                                 blend_mask);
                }
                dst.v[dst_start / 8] = blendvar(dst.v[dst_start / 8], x, mask);
                succ(len2);
            }
            if (len == 0) return;

            while (len >= 8) {
                modint8 l = v[start / 8], r = v[start / 8 + 1];
                dst.v[dst_start / 8] = blendvar(
                    l.rotate(u32(shift)), r.rotate(u32(shift)), blend_mask);
                succ(8);
            }
            if (len == 0) return;
            {
                ssize_t len2 = len;
                const auto mask = [&]() {
                    std::array<u32, 8> b = {};
                    for (int i = 0; i < 8; i++) {
                        b[i] = (i < len2);
                    }
                    return b;
                }();
                auto x = v[start / 8].rotate((u32)shift);
                if (len2 > 8 - start % 8) {
                    x = blendvar(x, v[start / 8 + 1].rotate((u32)(shift)),
                                 blend_mask);
                }
                dst.v[dst_start / 8] = blendvar(dst.v[dst_start / 8], x, mask);
                succ(len2);
            }
        }
    }

    ModVec& operator<<=(ssize_t s) {
        n += s;
        if (s % 8 == 0) {
            v.insert(v.begin(), s / 8, modint8());
        } else {
            v.resize(vsize(n));

            const auto mask = [&]() {
                std::array<u32, 8> b;
                for (int i = 0; i < 8; i++) {
                    b[i] = ((s % 8) <= i);
                }
                return b;
            }();
            for (auto i = std::ssize(v) - 1; i >= s / 8 + 1; i--) {
                modint8 l = v[i - 1 - s / 8], r = v[i - s / 8];
                v[i] = blendvar(l.rotate(u32(s % 8)), r.rotate(u32(8 - s % 8)),
                                mask);
            }
            v[s / 8] = blendvar(modint8(), v[0].rotate(u32(8 - s % 8)), mask);
            std::ranges::fill_n(v.begin(), s / 8, modint8());
        }
        return *this;
    }
    friend ModVec operator<<(const ModVec& lhs, ssize_t s) {
        return ModVec(lhs) <<= s;
    }

    ModVec inv(int m) const {
        // TODO: Optimize
        assert(val(0) == 1);
        ModVec res = ModVec({1});
        for (ssize_t i = 1; i < m; i *= 2) {
            ModVec pre(2 * i);
            copy_to(0, std::min(n, 2 * i), pre, 0);
            res = (res * 2 - res * res * pre);
            res.resize(2 * i);
        }
        res.resize(m);
        return res;
    }

    ModVec substr(ssize_t st, ssize_t len) const {
        assert(0 <= st && 0 <= len && st + len <= n);

        ModVec b(len);
        copy_to(st, len, b, 0);
        return b;
    }

    // sum a[i] * b[i]
    friend modint dot(const ModVec& lhs, const ModVec& rhs) {
        modint8 sum;
        for (int i = 0; i < std::min(std::ssize(lhs.v), std::ssize(rhs.v));
             i++) {
            sum += lhs.v[i] * rhs.v[i];
        }
        auto a = sum.val();
        // TODO: optimize
        modint ans;
        for (int i = 0; i < 8; i++) {
            ans += a[i];
        }
        return ans;
    }

    void reverse() {
        auto prev_n = n;
        n = std::ssize(v) * 8;
        std::reverse(v.begin(), v.end());
        for (modint8& x : v) {
            x = x.permutevar({7, 6, 5, 4, 3, 2, 1, 0});
        }

        ModVec tmp(prev_n);
        copy_to(n - prev_n, prev_n, tmp, 0);
        *this = tmp;
    }

    ModVec berlekamp_massey() const {
        ModVec b({-1}), c({-1});
        modint y = 1;
        for (int ed = 1; ed <= n; ed++) {
            int l = int(c.size()), m = int(b.size());
            b.resize(m + 1);
            m++;

            modint x = dot(c, substr(ed - l, l));
            if (x == 0) continue;
            modint freq = x * y.inv();
            if (l < m) {
                // use b
                auto prev_c = c;

                ModVec tmp(m);
                c.copy_to(0, l, tmp, m - l);
                c = tmp;
                c -= freq * b;

                b = prev_c;
                y = x;
            } else {
                // use c

                ModVec tmp(l);
                b.copy_to(0, m, tmp, l - m);

                c -= freq * tmp;
            }
        }
        c.reverse();
        return c;
    }

    friend std::ostream& operator<<(std::ostream& os, const ModVec& r) {
        auto r2 = r.val();

        os << "[";
        for (int i = 0; i < std::ssize(r2); i++) {
            if (i) os << ", ";
            os << r2[i];
        }
        return os << "]";
    }

  private:
    ssize_t n;
    std::vector<modint8> v;

    static ssize_t vsize(ssize_t n) { return (n + 7) / 8; }

    void clear_last() {
        if (n % 8 == 0) return;
        v.back() = blendvar(v.back(), modint8(), [&]() {
            std::array<u32, 8> b;
            for (int i = 0; i < 8; i++) {
                b[i] = ((n % 8) <= i);
            }
            return b;
        }());
    }
};

}  // namespace fastfps

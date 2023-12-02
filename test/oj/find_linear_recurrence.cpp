// verification-helper: PROBLEM https://judge.yosupo.jp/problem/find_linear_recurrence
#include <cstdio>
#include <vector>
#include <iostream>

#include "fastfps/modint.hpp"
#include "fastfps/modvec.hpp"

using namespace std;
using namespace fastfps;

const int MOD = 998244353;
using mint = ModInt<MOD>;
using mvec = ModVec<MOD>;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    cin >> n;

    std::vector<mint> a(n);
    for (int i = 0; i < n; i++) {
        int x;
        sc.read(x);
        a[i] = x;
    }

    auto c = modvec(a).berlekamp_massey().val();

    int m = c.size();

    pr.writeln(m - 1);
    for (int i = 0; i < m - 1; i++) {
        pr.write(c[i + 1].val());
        pr.write(' ');
    }
    pr.writeln();
}
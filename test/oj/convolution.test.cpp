// verification-helper: PROBLEM https://judge.yosupo.jp/problem/convolution_mod
#include <cstdio>
#include <vector>
#include <array>
#include <iostream>

#include "modint.hpp"
#include "modvec.hpp"

using namespace std;
using namespace fastfps;

const int MOD = 998244353;
using mint = ModInt<MOD>;
using mvec = ModVec<MOD>;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    cin >> n >> m;

    std::vector<mint> a(n);
    for (int i = 0; i < n; i++) {
        int x;
        cin >> x;
        a[i] = x;
    }
    std::vector<mint> b(m);
    for (int i = 0; i < m; i++) {
        int x;
        cin >> x;
        b[i] = x;
    }

    mvec a2(a), b2(b);

    auto c = (a2 * b2).val();
    for (auto x : c) {
        cout << x << " ";
    }
    cout << endl;
}

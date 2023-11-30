// verification-helper: PROBLEM https://judge.yosupo.jp/problem/inv_of_formal_power_series
#include <array>
#include <cstdio>
#include <iostream>
#include <vector>

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

    int n;
    cin >> n;

    std::vector<mint> a(n);
    for (int i = 0; i < n; i++) {
        int x;
        cin >> x;
        a[i] = x;
    }

    mvec f(a);

    mint a0 = a[0];

    auto c = ((f * a0.inv()).inv(n) * a0.inv()).val();
    for (auto x : c) {
        cout << x << " ";
    }
    cout << endl;
}

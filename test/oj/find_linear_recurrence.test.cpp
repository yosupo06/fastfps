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
        cin >> x;
        a[i] = x;
    }

    auto c = mvec(a).berlekamp_massey().val();

    auto m = ssize(c);

    cout << m - 1 << endl;
    for (int i = 0; i < m - 1; i++) {
        if (i) cout << " ";
        cout << c[i + 1];
    }
    cout << endl;
}

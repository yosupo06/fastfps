#include <array>
#include <iostream>
#include <vector>

#include <benchmark/benchmark.h>

#include "fft.hpp"
#include "modint.hpp"
#include "types.hpp"

using namespace fastfps;
const u32 MOD = 998244353;
using modint8 = ModInt8<MOD>;

void BM_fft(benchmark::State& state) {
    std::vector<modint8> a(state.range(0));
    for (int i = 0; i < state.range(0); i++) {
        std::array<u32, 8> b;
        for (int j = 0; j < 8; j++) {
            b[j] = i * 8 + j + 1234;
        }
        a[i] = b;
    }
    for (auto _ : state) {
        fft(a);
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_fft)->RangeMultiplier(2)->Range(1, 1 << 20);

void BM_ifft(benchmark::State& state) {
    std::vector<modint8> a(state.range(0));
    for (int i = 0; i < state.range(0); i++) {
        std::array<u32, 8> b;
        for (int j = 0; j < 8; j++) {
            b[j] = i * 8 + j + 1234;
        }
        a[i] = b;
    }
    for (auto _ : state) {
        ifft(a);
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_ifft)->RangeMultiplier(2)->Range(1, 1 << 20);

BENCHMARK_MAIN();

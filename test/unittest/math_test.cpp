#include "math.hpp"

#include <array>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>

using namespace fastfps;

TEST(MathTest, PrimitiveRoot) {
    ASSERT_EQ(3, primitive_root<998244353>);
}

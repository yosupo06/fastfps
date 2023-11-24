#include <array>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "math.hpp"

using namespace fastfps;

TEST(MathTest, PrimitiveRoot) {
    ASSERT_EQ(3, primitive_root<998244353>);
}

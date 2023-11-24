
#include <random>

inline std::mt19937 global_mt19937;

// random choise from [a, b]
template <class T>
inline T randint(T a, T b) {
    return std::uniform_int_distribution<T>(a, b)(global_mt19937);
}

inline bool randbool() {
    return randint(0, 1) == 0;
}

// random choice 2 disjoint elements from [lower, upper]
template <class T>
inline std::pair<T, T> randpair(T lower, T upper) {
    assert(upper - lower >= 1);
    T a, b;
    do {
        a = randint(lower, upper);
        b = randint(lower, upper);
    } while (a == b);
    if (a > b) std::swap(a, b);
    return {a, b};
}

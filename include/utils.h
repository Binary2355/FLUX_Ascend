#ifndef SRC_UTILS_H_
#define SRC_UTILS_H_

#include <stdlib.h>

template<typename T1, typename T2>
inline T1 Ceil(const T1& x, const T2& y)
{
    if (y == 0) {
        return 0;
    }
    return (x + y - 1) / y;
}

template<typename T1, typename T2>
inline T1 AlignUp(const T1& x, const T2& y)
{
    if (y == 0) {
        return 0;
    }
    return ((x + y - 1) / y) * y;
}

template<typename T1, typename T2>
inline T1 Tail(const T1& x, const T2& y)
{
    if (x == 0 || y == 0) {
        return 0;
    }
    return (x - 1) % y + 1;
}
#endif // SRC_UTILS_H_
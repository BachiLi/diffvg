#pragma once

#include "diffvg.h"

template <typename T>
DEVICE
inline bool solve_quadratic(T a, T b, T c, T *t0, T *t1) {
    // From https://github.com/mmp/pbrt-v3/blob/master/src/core/pbrt.h#L419
    T discrim = square(b) - 4 * a * c;
    if (discrim < 0) {
        return false;
    }
    T root_discrim = sqrt(discrim);

    T q;
    if (b < 0) {
        q = -0.5f * (b - root_discrim);
    } else {
        q = -0.5f * (b + root_discrim);
    }
    *t0 = q / a;
    *t1 = c / q;
    if (*t0 > *t1) {
        swap_(*t0, *t1);
    }
    return true;
}

template <typename T>
DEVICE
inline int solve_cubic(T a, T b, T c, T d, T t[3]) {
    if (fabs(a) < 1e-6f) {
        if (solve_quadratic(b, c, d, &t[0], &t[1])) {
            return 2;
        } else {
            return 0;
        }
    }
    // normalize cubic equation
    b /= a;
    c /= a;
    d /= a;
    T Q = (b * b - 3 * c) / 9.f;
    T R = (2 * b * b * b - 9 * b * c + 27 * d) / 54.f;
    if (R * R < Q * Q * Q) {
        // 3 real roots
        T theta = acos(R / sqrt(Q * Q * Q));
        t[0] = -2.f * sqrt(Q) * cos(theta / 3.f) - b / 3.f;
        t[1] = -2.f * sqrt(Q) * cos((theta + 2.f * T(M_PI)) / 3.f) - b / 3.f;
        t[2] = -2.f * sqrt(Q) * cos((theta - 2.f * T(M_PI)) / 3.f) - b / 3.f;
        return 3;
    } else {
        T A = R > 0 ? -pow(R + sqrt(R * R - Q * Q * Q), T(1./3.)):
                           pow(-R + sqrt(R * R - Q * Q * Q), T(1./3.));
        T B = fabs(A) > 1e-6f ? Q / A : T(0);
        t[0] = (A + B) - b / T(3);
        return 1;
    }
}

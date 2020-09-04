#pragma once

#include "diffvg.h"
#include "vector.h"
#include "matrix.h"

// https://stackoverflow.com/questions/39274472/error-function-atomicadddouble-double-has-already-been-defined
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
static inline DEVICE double atomicAdd(double *address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    if (val == 0.0)
        return __longlong_as_double(old);
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +__longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

#ifndef WIN32
    template <typename T0, typename T1>
    DEVICE
    inline T0 atomic_add_(T0 &target, T1 source) {
    #ifdef __CUDA_ARCH__
        return atomicAdd(&target, (T0)source);
    #else
        T0 old_val;
        T0 new_val;
        do {
            old_val = target;
            new_val = old_val + source;
        } while (!__atomic_compare_exchange(&target, &old_val, &new_val, true,
            std::memory_order::memory_order_seq_cst,
            std::memory_order::memory_order_seq_cst));
        return old_val;
    #endif
    }

    DEVICE
    inline
    float atomic_add(float &target, float source) {
        return atomic_add_(target, source);
    }
    DEVICE
    inline
    double atomic_add(double &target, double source) {
        return atomic_add_(target, source);
    }
#else
	float win_atomic_add(float &target, float source);
	double win_atomic_add(double &target, double source);
    DEVICE
    static float atomic_add(float &target, float source) {
    #ifdef __CUDA_ARCH__
        return atomicAdd(&target, source);
    #else
		return win_atomic_add(target, source);
    #endif
    }
    DEVICE
    static double atomic_add(double &target, double source) {
    #ifdef __CUDA_ARCH__
        return atomicAdd(&target, (double)source);
    #else
		return win_atomic_add(target, source);
    #endif
    }
#endif

template <typename T0, typename T1>
DEVICE
inline T0 atomic_add(T0 *target, T1 source) {
    return atomic_add(*target, (T0)source);
}

template <typename T0, typename T1>
DEVICE
inline TVector2<T0> atomic_add(TVector2<T0> &target, const TVector2<T1> &source) {
    atomic_add(target[0], source[0]);
    atomic_add(target[1], source[1]);
    return target;
}

template <typename T0, typename T1>
DEVICE
inline void atomic_add(T0 *target, const TVector2<T1> &source) {
    atomic_add(target[0], (T0)source[0]);
    atomic_add(target[1], (T0)source[1]);
}

template <typename T0, typename T1>
DEVICE
inline TVector3<T0> atomic_add(TVector3<T0> &target, const TVector3<T1> &source) {
    atomic_add(target[0], source[0]);
    atomic_add(target[1], source[1]);
    atomic_add(target[2], source[2]);
    return target;
}

template <typename T0, typename T1>
DEVICE
inline void atomic_add(T0 *target, const TVector3<T1> &source) {
    atomic_add(target[0], (T0)source[0]);
    atomic_add(target[1], (T0)source[1]);
    atomic_add(target[2], (T0)source[2]);
}

template <typename T0, typename T1>
DEVICE
inline TVector4<T0> atomic_add(TVector4<T0> &target, const TVector4<T1> &source) {
    atomic_add(target[0], source[0]);
    atomic_add(target[1], source[1]);
    atomic_add(target[2], source[2]);
    atomic_add(target[3], source[3]);
    return target;
}

template <typename T0, typename T1>
DEVICE
inline void atomic_add(T0 *target, const TVector4<T1> &source) {
    atomic_add(target[0], (T0)source[0]);
    atomic_add(target[1], (T0)source[1]);
    atomic_add(target[2], (T0)source[2]);
    atomic_add(target[3], (T0)source[3]);
}

template <typename T0, typename T1>
DEVICE
inline void atomic_add(T0 *target, const TMatrix3x3<T1> &source) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            atomic_add(target[3 * i + j], (T0)source(i, j));
        }
    }
}


#pragma once

#include "diffvg.h"
#include "vector.h"
#include "ptr.h"

enum class ColorType {
    Constant,
    LinearGradient,
    RadialGradient
};

struct Constant {
    Vector4f color;

    ptr<void> get_ptr() {
        return ptr<void>(this);
    }
};

struct LinearGradient {
    LinearGradient(const Vector2f &begin,
                   const Vector2f &end,
                   int num_stops,
                   ptr<float> stop_offsets,
                   ptr<float> stop_colors)
        : begin(begin), end(end), num_stops(num_stops),
          stop_offsets(stop_offsets.get()), stop_colors(stop_colors.get()) {}

    ptr<void> get_ptr() {
        return ptr<void>(this);
    }

    void copy_to(ptr<float> stop_offset,
                 ptr<float> stop_colors) const;

    Vector2f begin, end;
    int num_stops;
    float *stop_offsets;
    float *stop_colors; // rgba
};

struct RadialGradient {
    RadialGradient(const Vector2f &center,
                   const Vector2f &radius,
                   int num_stops,
                   ptr<float> stop_offsets,
                   ptr<float> stop_colors)
        : center(center), radius(radius), num_stops(num_stops),
          stop_offsets(stop_offsets.get()), stop_colors(stop_colors.get()) {}

    ptr<void> get_ptr() {
        return ptr<void>(this);
    }

    void copy_to(ptr<float> stop_offset,
                 ptr<float> stop_colors) const;

    Vector2f center, radius;
    int num_stops;
    float *stop_offsets;
    float *stop_colors; // rgba
};

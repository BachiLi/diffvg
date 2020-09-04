#pragma once

#include "diffvg.h"
#include "atomic.h"

enum class FilterType {
    Box,
    Tent,
    RadialParabolic, // 4/3(1 - (d/r))
    Hann // https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows
};

struct Filter {
    FilterType type;
    float radius;
};

struct DFilter {
    float radius;
};

DEVICE
inline
float compute_filter_weight(const Filter &filter,
                            float dx,
                            float dy) {
    if (fabs(dx) > filter.radius || fabs(dy) > filter.radius) {
        return 0;
    }
    if (filter.type == FilterType::Box) {
        return 1.f / square(2 * filter.radius);
    } else if (filter.type == FilterType::Tent) {
        return (filter.radius - fabs(dx)) * (filter.radius - fabs(dy)) /
               square(square(filter.radius));
    } else if (filter.type == FilterType::RadialParabolic) {
        return (4.f / 3.f) * (1 - square(dx / filter.radius)) *
               (4.f / 3.f) * (1 - square(dy / filter.radius));
    } else {
        assert(filter.type == FilterType::Hann);
        // normalize dx, dy to [0, 1]
        auto ndx = (dx / (2*filter.radius)) + 0.5f;
        auto ndy = (dy / (2*filter.radius)) + 0.5f;
        // the normalization factor is R^2
        return 0.5f * (1.f - cos(float(2 * M_PI) * ndx)) *
               0.5f * (1.f - cos(float(2 * M_PI) * ndy)) /
               square(filter.radius);
    }
}

DEVICE
inline
void d_compute_filter_weight(const Filter &filter,
                             float dx,
                             float dy,
                             float d_return,
                             DFilter *d_filter) {
    if (filter.type == FilterType::Box) {
        // return 1.f / square(2 * filter.radius);
        atomic_add(d_filter->radius,
            d_return * (-2) * 2 * filter.radius / cubic(2 * filter.radius));
    } else if (filter.type == FilterType::Tent) {
        // return (filer.radius - fabs(dx)) * (filer.radius - fabs(dy)) /
        //        square(square(filter.radius));
        auto fx = filter.radius - fabs(dx);
        auto fy = filter.radius - fabs(dy);
        auto norm = 1 / square(filter.radius);
        auto d_fx = d_return * fy * norm;
        auto d_fy = d_return * fx * norm;
        auto d_norm = d_return * fx * fy;
        atomic_add(d_filter->radius,
            d_fx + d_fy + (-4) * d_norm / pow(filter.radius, 5));
    } else if (filter.type == FilterType::RadialParabolic) {
        // return (4.f / 3.f) * (1 - square(dx / filter.radius)) *
        //        (4.f / 3.f) * (1 - square(dy / filter.radius));
        // auto d_square_x = d_return * (-4.f / 3.f);
        // auto d_square_y = d_return * (-4.f / 3.f);
        auto r3 = filter.radius * filter.radius * filter.radius;
        auto d_radius = -(2 * square(dx) + 2 * square(dy)) / r3;
        atomic_add(d_filter->radius, d_radius);
    } else {
        assert(filter.type == FilterType::Hann);
        // // normalize dx, dy to [0, 1]
        // auto ndx = (dx / (2*filter.radius)) + 0.5f;
        // auto ndy = (dy / (2*filter.radius)) + 0.5f;
        // // the normalization factor is R^2
        // return 0.5f * (1.f - cos(float(2 * M_PI) * ndx)) *
        //        0.5f * (1.f - cos(float(2 * M_PI) * ndy)) /
        //        square(filter.radius);

        // normalize dx, dy to [0, 1]
        auto ndx = (dx / (2*filter.radius)) + 0.5f;
        auto ndy = (dy / (2*filter.radius)) + 0.5f;
        auto fx = 0.5f * (1.f - cos(float(2*M_PI) * ndx));
        auto fy = 0.5f * (1.f - cos(float(2*M_PI) * ndy));
        auto norm = 1 / square(filter.radius);
        auto d_fx = d_return * fy * norm;
        auto d_fy = d_return * fx * norm;
        auto d_norm = d_return * fx * fy;
        auto d_ndx = d_fx * 0.5f * sin(float(2*M_PI) * ndx) * float(2*M_PI);
        auto d_ndy = d_fy * 0.5f * sin(float(2*M_PI) * ndy) * float(2*M_PI);
        atomic_add(d_filter->radius,
            d_ndx * (-2*dx / square(2*filter.radius)) +
            d_ndy * (-2*dy / square(2*filter.radius)) +
            (-2) * d_norm / cubic(filter.radius));
    }
}

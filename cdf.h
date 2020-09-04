#pragma once

#include "diffvg.h"

DEVICE int sample(const float *cdf, int num_entries, float u, float *updated_u = nullptr) {
    // Binary search the cdf
    auto lb = 0;
    auto len = num_entries - 1 - lb;
    while (len > 0) {
        auto half_len = len / 2;
        auto mid = lb + half_len;
        assert(mid >= 0 && mid < num_entries);
        if (u < cdf[mid]) {
            len = half_len;
        } else {
            lb = mid + 1;
            len = len - half_len - 1;
        }
    }
    lb = clamp(lb, 0, num_entries - 1);
    if (updated_u != nullptr) {
    	if (lb > 0) {
    		*updated_u = (u - cdf[lb - 1]) / (cdf[lb] - cdf[lb - 1]);
    	} else {
    		*updated_u = u / cdf[lb];
    	}
    }
    return lb;
}

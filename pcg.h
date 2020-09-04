#pragma once

#include "diffvg.h"

// http://www.pcg-random.org/download.html
struct pcg32_state {
    uint64_t state;
    uint64_t inc;
};

DEVICE inline uint32_t next_pcg32(pcg32_state *rng) {
    uint64_t oldstate = rng->state;
    // Advance internal state
    rng->state = oldstate * 6364136223846793005ULL + (rng->inc|1);
    // Calculate output function (XSH RR), uses old state for max ILP
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

// https://github.com/wjakob/pcg32/blob/master/pcg32.h
DEVICE inline float next_pcg32_float(pcg32_state *rng) {
    union {
        uint32_t u;
        float f;
    } x;
    x.u = (next_pcg32(rng) >> 9) | 0x3f800000u;
    return x.f - 1.0f;
}

// Initialize each pixel with a PCG rng with a different stream
DEVICE inline pcg32_state init_pcg32(int idx, uint64_t seed) {
    pcg32_state state;
    state.state = 0U;
    state.inc = (((uint64_t)idx + 1) << 1u) | 1u;
    next_pcg32(&state);
    state.state += (0x853c49e6748fea9bULL + seed);
    next_pcg32(&state);
    return state;
}

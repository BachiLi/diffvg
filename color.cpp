#include "color.h"

void LinearGradient::copy_to(ptr<float> stop_offsets,
                             ptr<float> stop_colors) const {
    float *o = stop_offsets.get();
    float *c = stop_colors.get();
    for (int i = 0; i < num_stops; i++) {
        o[i] = this->stop_offsets[i];
    }
    for (int i = 0; i < 4 * num_stops; i++) {
        c[i] = this->stop_colors[i];
    }
}

void RadialGradient::copy_to(ptr<float> stop_offsets,
                             ptr<float> stop_colors) const {
    float *o = stop_offsets.get();
    float *c = stop_colors.get();
    for (int i = 0; i < num_stops; i++) {
        o[i] = this->stop_offsets[i];
    }
    for (int i = 0; i < 4 * num_stops; i++) {
        c[i] = this->stop_colors[i];
    }
}

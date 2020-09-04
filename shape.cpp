#include "shape.h"

void Path::copy_to(ptr<float> points, ptr<float> thickness) const {
    float *p = points.get();
    for (int i = 0; i < 2 * num_points; i++) {
        p[i] = this->points[i];
    }
    if (this->thickness != nullptr) {
        float *t = thickness.get();
        for (int i = 0; i < num_points; i++) {
            t[i] = this->thickness[i];
        }
    }
}

void ShapeGroup::copy_to(ptr<float> shape_to_canvas) const {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            shape_to_canvas.get()[i * 3 + j] = this->shape_to_canvas(i, j);
        }
    }
}

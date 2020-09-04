#pragma once

#include "diffvg.h"
#include "cuda_utils.h"
#include "vector.h"
#include "matrix.h"

struct AABB {
    DEVICE
    inline AABB(const Vector2f &p_min = Vector2f{infinity<float>(), infinity<float>()},
                const Vector2f &p_max = Vector2f{-infinity<float>(), -infinity<float>()})
        : p_min(p_min), p_max(p_max) {}
    Vector2f p_min, p_max;
};

DEVICE
inline
AABB merge(const AABB &box, const Vector2f &p) {
    return AABB{Vector2f{min(p.x, box.p_min.x), min(p.y, box.p_min.y)},
                Vector2f{max(p.x, box.p_max.x), max(p.y, box.p_max.y)}};
}

DEVICE
inline
AABB merge(const AABB &box0, const AABB &box1) {
    return AABB{Vector2f{min(box0.p_min.x, box1.p_min.x), min(box0.p_min.y, box1.p_min.y)},
                Vector2f{max(box0.p_max.x, box1.p_max.x), max(box0.p_max.y, box1.p_max.y)}};
}

DEVICE
inline
bool inside(const AABB &box, const Vector2f &p) {
    return p.x >= box.p_min.x && p.x <= box.p_max.x &&
           p.y >= box.p_min.y && p.y <= box.p_max.y;
}

DEVICE
inline
bool inside(const AABB &box, const Vector2f &p, float radius) {
    return p.x >= box.p_min.x - radius && p.x <= box.p_max.x + radius &&
           p.y >= box.p_min.y - radius && p.y <= box.p_max.y + radius;
}

DEVICE
inline
AABB enlarge(const AABB &box, float width) {
    return AABB{Vector2f{box.p_min.x - width, box.p_min.y - width},
                Vector2f{box.p_max.x + width, box.p_max.y + width}};
}

DEVICE
inline
AABB transform(const Matrix3x3f &xform, const AABB &box) {
    auto ret = AABB();
    ret = merge(ret, xform_pt(xform, Vector2f{box.p_min.x, box.p_min.y}));
    ret = merge(ret, xform_pt(xform, Vector2f{box.p_min.x, box.p_max.y}));
    ret = merge(ret, xform_pt(xform, Vector2f{box.p_max.x, box.p_min.y}));
    ret = merge(ret, xform_pt(xform, Vector2f{box.p_max.x, box.p_max.y}));
    return ret;
}

DEVICE
inline
bool within_distance(const AABB &box, const Vector2f &pt, float r) {
    return pt.x >= box.p_min.x - r && pt.x <= box.p_max.x + r &&
           pt.y >= box.p_min.y - r && pt.y <= box.p_max.y + r;
}

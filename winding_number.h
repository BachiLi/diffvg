#pragma once

#include "diffvg.h"
#include "scene.h"
#include "shape.h"
#include "solve.h"
#include "vector.h"

DEVICE
int compute_winding_number(const Circle &circle, const Vector2f &pt) {
    const auto &c = circle.center;
    auto r = circle.radius;
    // inside the circle: return 1, outside the circle: return 0
    if (distance_squared(c, pt) < r * r) {
        return 1;
    } else {
        return 0;
    }
}

DEVICE
int compute_winding_number(const Ellipse &ellipse, const Vector2f &pt) {
    const auto &c = ellipse.center;
    const auto &r = ellipse.radius;
    // inside the ellipse: return 1, outside the ellipse: return 0
    if (square(c.x - pt.x) / square(r.x) + square(c.y - pt.y) / square(r.y) < 1) {
        return 1;
    } else {
        return 0;
    }
}

DEVICE
bool intersect(const AABB &box, const Vector2f &pt) {
    if (pt.y < box.p_min.y || pt.y > box.p_max.y) {
        return false;
    }
    if (pt.x > box.p_max.x) {
        return false;
    }
    return true;
}

DEVICE
int compute_winding_number(const Path &path, const BVHNode *bvh_nodes, const Vector2f &pt) {
    // Shoot a horizontal ray from pt to right, intersect with all curves of the path,
    // count intersection
    auto num_segments = path.num_base_points;
    constexpr auto max_bvh_size = 128;
    int bvh_stack[max_bvh_size];
    auto stack_size = 0;
    auto winding_number = 0;
    bvh_stack[stack_size++] = 2 * num_segments - 2;
    while (stack_size > 0) {
        const BVHNode &node = bvh_nodes[bvh_stack[--stack_size]];
        if (node.child1 < 0) {
            // leaf
            auto base_point_id = node.child0;
            auto point_id = - node.child1 - 1;
            assert(base_point_id < num_segments);
            assert(point_id < path.num_points);
            if (path.num_control_points[base_point_id] == 0) {
                // Straight line
                auto i0 = point_id;
                auto i1 = (point_id + 1) % path.num_points;
                auto p0 = Vector2f{path.points[2 * i0], path.points[2 * i0 + 1]};
                auto p1 = Vector2f{path.points[2 * i1], path.points[2 * i1 + 1]};
                // intersect p0 + t * (p1 - p0) with pt + t' * (1, 0)
                // solve:
                // pt.x + t' = v0.x + t * (v1.x - v0.x)
                // pt.y      = v0.y + t * (v1.y - v0.y)
                if (p1.y != p0.y) {
                    auto t = (pt.y - p0.y) / (p1.y - p0.y);
                    if (t >= 0 && t <= 1) {
                        auto tp = p0.x - pt.x + t * (p1.x - p0.x);
                        if (tp >= 0) {
                            if (p1.y - p0.y > 0) {
                                winding_number += 1;
                            } else {
                                winding_number -= 1;
                            }
                        }
                    }
                }
            } else if (path.num_control_points[base_point_id] == 1) {
                // Quadratic Bezier curve
                auto i0 = point_id;
                auto i1 = point_id + 1;
                auto i2 = (point_id + 2) % path.num_points;
                auto p0 = Vector2f{path.points[2 * i0], path.points[2 * i0 + 1]};
                auto p1 = Vector2f{path.points[2 * i1], path.points[2 * i1 + 1]};
                auto p2 = Vector2f{path.points[2 * i2], path.points[2 * i2 + 1]};
                // The curve is (1-t)^2p0 + 2(1-t)tp1 + t^2p2
                // = (p0-2p1+p2)t^2+(-2p0+2p1)t+p0
                // intersect with pt + t' * (1 0)
                // solve
                // pt.y = (p0-2p1+p2)t^2+(-2p0+2p1)t+p0
                float t[2];
                if (solve_quadratic(p0.y-2*p1.y+p2.y,
                                    -2*p0.y+2*p1.y,
                                    p0.y-pt.y,
                                    &t[0], &t[1])) {
                    for (int j = 0; j < 2; j++) {
                        if (t[j] >= 0 && t[j] <= 1) {
                            auto tp = (p0.x-2*p1.x+p2.x)*t[j]*t[j] +
                                      (-2*p0.x+2*p1.x)*t[j] +
                                      p0.x-pt.x;
                            if (tp >= 0) {
                                if (2*(p0.y-2*p1.y+p2.y)*t[j]+(-2*p0.y+2*p1.y) > 0) {
                                    winding_number += 1;
                                } else {
                                    winding_number -= 1;
                                }
                            }
                        }
                    }
                }
            } else if (path.num_control_points[base_point_id] == 2) {
                // Cubic Bezier curve
                auto i0 = point_id;
                auto i1 = point_id + 1;
                auto i2 = point_id + 2;
                auto i3 = (point_id + 3) % path.num_points;
                auto p0 = Vector2f{path.points[2 * i0], path.points[2 * i0 + 1]};
                auto p1 = Vector2f{path.points[2 * i1], path.points[2 * i1 + 1]};
                auto p2 = Vector2f{path.points[2 * i2], path.points[2 * i2 + 1]};
                auto p3 = Vector2f{path.points[2 * i3], path.points[2 * i3 + 1]};
                // The curve is (1 - t)^3 p0 + 3 * (1 - t)^2 t p1 + 3 * (1 - t) t^2 p2 + t^3 p3
                // = (-p0+3p1-3p2+p3) t^3 + (3p0-6p1+3p2) t^2 + (-3p0+3p1) t + p0
                // intersect with pt + t' * (1 0)
                // solve:
                // pt.y = (-p0+3p1-3p2+p3) t^3 + (3p0-6p1+3p2) t^2 + (-3p0+3p1) t + p0
                double t[3];
                int num_sol = solve_cubic(double(-p0.y+3*p1.y-3*p2.y+p3.y),
                                          double(3*p0.y-6*p1.y+3*p2.y),
                                          double(-3*p0.y+3*p1.y),
                                          double(p0.y-pt.y),
                                          t);
                for (int j = 0; j < num_sol; j++) {
                    if (t[j] >= 0 && t[j] <= 1) {
                        // t' = (-p0+3p1-3p2+p3) t^3 + (3p0-6p1+3p2) t^2 + (-3p0+3p1) t + p0 - pt.x
                        auto tp = (-p0.x+3*p1.x-3*p2.x+p3.x)*t[j]*t[j]*t[j]+
                                  (3*p0.x-6*p1.x+3*p2.x)*t[j]*t[j]+
                                  (-3*p0.x+3*p1.x)*t[j]+
                                  p0.x-pt.x;
                        if (tp > 0) {
                            if (3*(-p0.y+3*p1.y-3*p2.y+p3.y)*t[j]*t[j]+
                                2*(3*p0.y-6*p1.y+3*p2.y)*t[j]+
                                (-3*p0.y+3*p1.y) > 0) {
                                winding_number += 1;
                            } else {
                                winding_number -= 1;
                            }
                        }
                    }
                }
            } else {
                assert(false);
            }
        } else {
            assert(node.child0 >= 0 && node.child1 >= 0);
            const AABB &b0 = bvh_nodes[node.child0].box;
            if (intersect(b0, pt)) {
                bvh_stack[stack_size++] = node.child0;
            }
            const AABB &b1 = bvh_nodes[node.child1].box;
            if (intersect(b1, pt)) {
                bvh_stack[stack_size++] = node.child1;
            }
            assert(stack_size <= max_bvh_size);
        }
    }
    return winding_number;
}

DEVICE
int compute_winding_number(const Rect &rect, const Vector2f &pt) {
    const auto &p_min = rect.p_min;
    const auto &p_max = rect.p_max;
    // inside the rectangle: return 1, outside the rectangle: return 0
    if (pt.x > p_min.x && pt.x < p_max.x && pt.y > p_min.y && pt.y < p_max.y) {
        return 1;
    } else {
        return 0;
    }
}

DEVICE
int compute_winding_number(const Shape &shape, const BVHNode *bvh_nodes, const Vector2f &pt) {
    switch (shape.type) {
        case ShapeType::Circle:
            return compute_winding_number(*(const Circle *)shape.ptr, pt);
        case ShapeType::Ellipse:
            return compute_winding_number(*(const Ellipse *)shape.ptr, pt);
        case ShapeType::Path:
            return compute_winding_number(*(const Path *)shape.ptr, bvh_nodes, pt);
        case ShapeType::Rect:
            return compute_winding_number(*(const Rect *)shape.ptr, pt);
    }
    assert(false);
    return 0;
}

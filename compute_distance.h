#pragma once

#include "diffvg.h"
#include "edge_query.h"
#include "scene.h"
#include "shape.h"
#include "solve.h"
#include "vector.h"

#include <cassert>

struct ClosestPointPathInfo {
    int base_point_id;
    int point_id;
    float t_root;
};

DEVICE
inline
bool closest_point(const Circle &circle, const Vector2f &pt,
                   Vector2f *result) {
    *result = circle.center + circle.radius * normalize(pt - circle.center);
    return false;
}

DEVICE
inline
bool closest_point(const Path &path, const BVHNode *bvh_nodes, const Vector2f &pt, float max_radius,
                   ClosestPointPathInfo *path_info,
                   Vector2f *result) {
    auto min_dist = max_radius;
    auto ret_pt = Vector2f{0, 0};
    auto found = false;
    auto num_segments = path.num_base_points;
    constexpr auto max_bvh_size = 128;
    int bvh_stack[max_bvh_size];
    auto stack_size = 0;
    bvh_stack[stack_size++] = 2 * num_segments - 2;
    while (stack_size > 0) {
        const BVHNode &node = bvh_nodes[bvh_stack[--stack_size]];
        if (node.child1 < 0) {
            // leaf
            auto base_point_id = node.child0;
            auto point_id = - node.child1 - 1;
            assert(base_point_id < num_segments);
            assert(point_id < path.num_points);
            auto dist = 0.f;
            auto closest_pt = Vector2f{0, 0};
            auto t_root = 0.f;
            if (path.num_control_points[base_point_id] == 0) {
                // Straight line
                auto i0 = point_id;
                auto i1 = (point_id + 1) % path.num_points;
                auto p0 = Vector2f{path.points[2 * i0], path.points[2 * i0 + 1]};
                auto p1 = Vector2f{path.points[2 * i1], path.points[2 * i1 + 1]};
                // project pt to line
                auto t = dot(pt - p0, p1 - p0) / dot(p1 - p0, p1 - p0);
                if (t < 0) {
                    dist = distance(p0, pt);
                    closest_pt = p0;
                    t_root = 0;
                } else if (t > 1) {
                    dist = distance(p1, pt);
                    closest_pt = p1;
                    t_root = 1;
                } else {
                    dist = distance(p0 + t * (p1 - p0), pt);
                    closest_pt = p0 + t * (p1 - p0);
                    t_root = t;
                }
            } else if (path.num_control_points[base_point_id] == 1) {
                // Quadratic Bezier curve
                auto i0 = point_id;
                auto i1 = point_id + 1;
                auto i2 = (point_id + 2) % path.num_points;
                auto p0 = Vector2f{path.points[2 * i0], path.points[2 * i0 + 1]};
                auto p1 = Vector2f{path.points[2 * i1], path.points[2 * i1 + 1]};
                auto p2 = Vector2f{path.points[2 * i2], path.points[2 * i2 + 1]};
                if (path.use_distance_approx) {
                    closest_pt = quadratic_closest_pt_approx(p0, p1, p2, pt, &t_root);
                    dist = distance(closest_pt, pt);
                } else {
                    auto eval = [&](float t) -> Vector2f {
                        auto tt = 1 - t;
                        return (tt*tt)*p0 + (2*tt*t)*p1 + (t*t)*p2;
                    };
                    auto pt0 = eval(0);
                    auto pt1 = eval(1);
                    auto dist0 = distance(pt0, pt);
                    auto dist1 = distance(pt1, pt);
                    {
                        dist = dist0;
                        closest_pt = pt0;
                        t_root = 0;
                    }
                    if (dist1 < dist) {
                        dist = dist1;
                        closest_pt = pt1;
                        t_root = 1;
                    }
                    // The curve is (1-t)^2p0 + 2(1-t)tp1 + t^2p2
                    // = (p0-2p1+p2)t^2+(-2p0+2p1)t+p0 = q
                    // Want to solve (q - pt) dot q' = 0
                    // q' = (p0-2p1+p2)t + (-p0+p1)
                    // Expanding (p0-2p1+p2)^2 t^3 +
                    //           3(p0-2p1+p2)(-p0+p1) t^2 +
                    //           (2(-p0+p1)^2+(p0-2p1+p2)(p0-pt))t +
                    //           (-p0+p1)(p0-pt) = 0
                    auto A = sum((p0-2*p1+p2)*(p0-2*p1+p2));
                    auto B = sum(3*(p0-2*p1+p2)*(-p0+p1));
                    auto C = sum(2*(-p0+p1)*(-p0+p1)+(p0-2*p1+p2)*(p0-pt));
                    auto D = sum((-p0+p1)*(p0-pt));
                    float t[3];
                    int num_sol = solve_cubic(A, B, C, D, t);
                    for (int j = 0; j < num_sol; j++) {
                        if (t[j] >= 0 && t[j] <= 1) {
                            auto p = eval(t[j]);
                            auto distp = distance(p, pt);
                            if (distp < dist) {
                                dist = distp;
                                closest_pt = p;
                                t_root = t[j];
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
                auto eval = [&](float t) -> Vector2f {
                    auto tt = 1 - t;
                    return (tt*tt*tt)*p0 + (3*tt*tt*t)*p1 + (3*tt*t*t)*p2 + (t*t*t)*p3;
                };
                auto pt0 = eval(0);
                auto pt1 = eval(1);
                auto dist0 = distance(pt0, pt);
                auto dist1 = distance(pt1, pt);
                {
                    dist = dist0;
                    closest_pt = pt0;
                    t_root = 0;
                }
                if (dist1 < dist) {
                    dist = dist1;
                    closest_pt = pt1;
                    t_root = 1;
                }
                // The curve is (1 - t)^3 p0 + 3 * (1 - t)^2 t p1 + 3 * (1 - t) t^2 p2 + t^3 p3
                // = (-p0+3p1-3p2+p3) t^3 + (3p0-6p1+3p2) t^2 + (-3p0+3p1) t + p0
                // Want to solve (q - pt) dot q' = 0
                // q' = 3*(-p0+3p1-3p2+p3)t^2 + 2*(3p0-6p1+3p2)t + (-3p0+3p1)
                // Expanding 
                // 3*(-p0+3p1-3p2+p3)^2 t^5
                // 5*(-p0+3p1-3p2+p3)(3p0-6p1+3p2) t^4
                // 4*(-p0+3p1-3p2+p3)(-3p0+3p1) + 2*(3p0-6p1+3p2)^2 t^3
                // 3*(3p0-6p1+3p2)(-3p0+3p1) + 3*(-p0+3p1-3p2+p3)(p0-pt) t^2
                // (-3p0+3p1)^2+2(p0-pt)(3p0-6p1+3p2) t
                // (p0-pt)(-3p0+3p1)
                double A = 3*sum((-p0+3*p1-3*p2+p3)*(-p0+3*p1-3*p2+p3));
                double B = 5*sum((-p0+3*p1-3*p2+p3)*(3*p0-6*p1+3*p2));
                double C = 4*sum((-p0+3*p1-3*p2+p3)*(-3*p0+3*p1)) + 2*sum((3*p0-6*p1+3*p2)*(3*p0-6*p1+3*p2));
                double D = 3*(sum((3*p0-6*p1+3*p2)*(-3*p0+3*p1)) + sum((-p0+3*p1-3*p2+p3)*(p0-pt)));
                double E = sum((-3*p0+3*p1)*(-3*p0+3*p1)) + 2*sum((p0-pt)*(3*p0-6*p1+3*p2));
                double F = sum((p0-pt)*(-3*p0+3*p1));
                // normalize the polynomial
                B /= A;
                C /= A;
                D /= A;
                E /= A;
                F /= A;
                // Isolator Polynomials:
                // https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.133.2233&rep=rep1&type=pdf
                //                                       x/5 + B/25
                //                                    /-----------------------------------------------------
                // 5x^4 + 4B x^3 + 3C x^2 + 2D x + E /   x^5 +    B x^4 +       C x^3 +      D x^2 +      E x + F
                //                                       x^5 + 4B/5 x^4 +    3C/5 x^3 +   2D/5 x^2 +    E/5 x
                //                                      ----------------------------------------------------
                //                                              B/5 x^4 +    2C/5 x^3 +   3D/5 x^2 +   4E/5 x + F
                //                                              B/5 x^4 + 4B^2/25 x^3 + 3BC/25 x^2 + 2BD/25 x + BE/25
                //                                      ----------------------------------------------------
                //                                     (2C/5 - 4B^2/25)x^3 + (3D/5-3BC/25)x^2 + (4E/5-2BD/25) + (F-BE/25)
                auto p1A = ((2 / 5.f) * C - (4 / 25.f) * B * B);
                auto p1B = ((3 / 5.f) * D - (3 / 25.f) * B * C);
                auto p1C = ((4 / 5.f) * E - (2 / 25.f) * B * D);
                auto p1D = F - B * E / 25.f;
                // auto q1A = 1 / 5.f;
                // auto q1B = B / 25.f;
                // x/5 + B/25 = 0
                // x = -B/5
                auto q_root = -B/5.f;
                double p_roots[3];
                int num_sol = solve_cubic(p1A, p1B, p1C, p1D, p_roots);
                float intervals[4];
                if (q_root >= 0 && q_root <= 1) {
                    intervals[0] = q_root;
                }
                for (int j = 0; j < num_sol; j++) {
                    intervals[j + 1] = p_roots[j];
                }
                auto num_intervals = 1 + num_sol;
                // sort intervals
                for (int j = 1; j < num_intervals; j++) {
                    for (int k = j; k > 0 && intervals[k - 1] > intervals[k]; k--) {
                        auto tmp = intervals[k];
                        intervals[k] = intervals[k - 1];
                        intervals[k - 1] = tmp;
                    }
                }
                auto eval_polynomial = [&] (double t) {
                    return t*t*t*t*t+
                           B*t*t*t*t+
                           C*t*t*t+
                           D*t*t+
                           E*t+
                           F;
                };
                auto eval_polynomial_deriv = [&] (double t) {
                    return 5*t*t*t*t+
                           4*B*t*t*t+
                           3*C*t*t+
                           2*D*t+
                           E;
                };
                auto lower_bound = 0.f;
                for (int j = 0; j < num_intervals + 1; j++) {
                    if (j < num_intervals && intervals[j] < 0.f) {
                        continue;
                    }
                    auto upper_bound = j < num_intervals ?
                        min(intervals[j], 1.f) : 1.f;
                    auto lb = lower_bound;
                    auto ub = upper_bound;
                    auto lb_eval = eval_polynomial(lb);
                    auto ub_eval = eval_polynomial(ub);
                    if (lb_eval * ub_eval > 0) {
                        // Doesn't have root
                        continue;
                    }
                    if (lb_eval > ub_eval) {
                        swap_(lb, ub);
                    }
                    auto t = 0.5f * (lb + ub);
                    auto num_iter = 20;
                    for (int it = 0; it < num_iter; it++) {
                        if (!(t >= lb && t <= ub)) {
                            t = 0.5f * (lb + ub);
                        }
                        auto value = eval_polynomial(t);
                        if (fabs(value) < 1e-5f || it == num_iter - 1) {
                            break;
                        }
                        // The derivative may not be entirely accurate,
                        // but the bisection is going to handle this
                        if (value > 0.f) {
                            ub = t;
                        } else {
                            lb = t;
                        }
                        auto derivative = eval_polynomial_deriv(t);
                        t -= value / derivative;
                    }
                    auto p = eval(t);
                    auto distp = distance(p, pt);
                    if (distp < dist) {
                        dist = distp;
                        closest_pt = p;
                        t_root = t;
                    }
                    if (upper_bound >= 1.f) {
                        break;
                    }
                    lower_bound = upper_bound;
                }
            } else {
                assert(false);
            }
            if (dist < min_dist) {
                min_dist = dist;
                ret_pt = closest_pt;
                path_info->base_point_id = base_point_id;
                path_info->point_id = point_id;
                path_info->t_root = t_root;
                found = true;
            }
        } else {
            assert(node.child0 >= 0 && node.child1 >= 0);
            const AABB &b0 = bvh_nodes[node.child0].box;
            if (within_distance(b0, pt, min_dist)) {
                bvh_stack[stack_size++] = node.child0;
            }
            const AABB &b1 = bvh_nodes[node.child1].box;
            if (within_distance(b1, pt, min_dist)) {
                bvh_stack[stack_size++] = node.child1;
            }
            assert(stack_size <= max_bvh_size);
        }
    }
    if (found) {
        assert(path_info->base_point_id < num_segments);
    }
    *result = ret_pt;
    return found;
}

DEVICE
inline
bool closest_point(const Rect &rect, const Vector2f &pt,
                   Vector2f *result) {
    auto min_dist = 0.f;
    auto closest_pt = Vector2f{0, 0};
    auto update = [&](const Vector2f &p0, const Vector2f &p1, bool first) {
        // project pt to line
        auto t = dot(pt - p0, p1 - p0) / dot(p1 - p0, p1 - p0);
        if (t < 0) {
            auto d = distance(p0, pt);
            if (first || d < min_dist) {
                min_dist = d;
                closest_pt = p0;
            }
        } else if (t > 1) {
            auto d = distance(p1, pt);
            if (first || d < min_dist) {
                min_dist = d;
                closest_pt = p1;
            }
        } else {
            auto p = p0 + t * (p1 - p0);
            auto d = distance(p, pt);
            if (first || d < min_dist) {
                min_dist = d;
                closest_pt = p0;
            }
        }
    };
    auto left_top = rect.p_min;
    auto right_top = Vector2f{rect.p_max.x, rect.p_min.y};
    auto left_bottom = Vector2f{rect.p_min.x, rect.p_max.y};
    auto right_bottom = rect.p_max;
    update(left_top, left_bottom, true);
    update(left_top, right_top, false);
    update(right_top, right_bottom, false);
    update(left_bottom, right_bottom, false);
    *result = closest_pt;
    return true;
}

DEVICE
inline
bool closest_point(const Shape &shape, const BVHNode *bvh_nodes, const Vector2f &pt, float max_radius,
                   ClosestPointPathInfo *path_info,
                   Vector2f *result) {
    switch (shape.type) {
        case ShapeType::Circle:
            return closest_point(*(const Circle *)shape.ptr, pt, result);
        case ShapeType::Ellipse:
            // https://www.geometrictools.com/Documentation/DistancePointEllipseEllipsoid.pdf
            assert(false);
            return false;
        case ShapeType::Path:
            return closest_point(*(const Path *)shape.ptr, bvh_nodes, pt, max_radius, path_info, result);
        case ShapeType::Rect:
            return closest_point(*(const Rect *)shape.ptr, pt, result);
    }
    assert(false);
    return false;
}

DEVICE
inline
bool compute_distance(const SceneData &scene,
                      int shape_group_id,
                      const Vector2f &pt,
                      float max_radius,
                      int *min_shape_id,
                      Vector2f *closest_pt_,
                      ClosestPointPathInfo *path_info,
                      float *result) {
    const ShapeGroup &shape_group = scene.shape_groups[shape_group_id];
    // pt is in canvas space, transform it to shape's local space
    auto local_pt = xform_pt(shape_group.canvas_to_shape, pt);

    constexpr auto max_bvh_stack_size = 64;
    int bvh_stack[max_bvh_stack_size];
    auto stack_size = 0;
    bvh_stack[stack_size++] = 2 * shape_group.num_shapes - 2;
    const auto &bvh_nodes = scene.shape_groups_bvh_nodes[shape_group_id];

    auto min_dist = max_radius;
    auto found = false;

    while (stack_size > 0) {
        const BVHNode &node = bvh_nodes[bvh_stack[--stack_size]];
        if (node.child1 < 0) {
            // leaf
            auto shape_id = node.child0;
            const auto &shape = scene.shapes[shape_id];
            ClosestPointPathInfo local_path_info{-1, -1};
            auto local_closest_pt = Vector2f{0, 0};
            if (closest_point(shape, scene.path_bvhs[shape_id], local_pt, max_radius, &local_path_info, &local_closest_pt)) {
                auto closest_pt = xform_pt(shape_group.shape_to_canvas, local_closest_pt);
                auto dist = distance(closest_pt, pt);
                if (!found || dist < min_dist) {
                    found = true;
                    min_dist = dist;
                    if (min_shape_id != nullptr) {
                        *min_shape_id = shape_id;
                    }
                    if (closest_pt_ != nullptr) {
                        *closest_pt_ = closest_pt;
                    }
                    if (path_info != nullptr) {
                        *path_info = local_path_info;
                    }
                }
            }
        } else {
            assert(node.child0 >= 0 && node.child1 >= 0);
            const AABB &b0 = bvh_nodes[node.child0].box;
            if (inside(b0, local_pt, max_radius)) {
                bvh_stack[stack_size++] = node.child0;
            }
            const AABB &b1 = bvh_nodes[node.child1].box;
            if (inside(b1, local_pt, max_radius)) {
                bvh_stack[stack_size++] = node.child1;
            }
            assert(stack_size <= max_bvh_stack_size);
        }
    }

    *result = min_dist;
    return found;
}


DEVICE
inline
void d_closest_point(const Circle &circle,
                     const Vector2f &pt,
                     const Vector2f &d_closest_pt,
                     Circle &d_circle,
                     Vector2f &d_pt) {
    // return circle.center + circle.radius * normalize(pt - circle.center);
    auto d_center = d_closest_pt *
        (1 + d_normalize(pt - circle.center, circle.radius * d_closest_pt));
    atomic_add(&d_circle.center.x, d_center);
    atomic_add(&d_circle.radius, dot(d_closest_pt, normalize(pt - circle.center)));
}

DEVICE
inline
void d_closest_point(const Path &path,
                     const Vector2f &pt,
                     const Vector2f &d_closest_pt,
                     const ClosestPointPathInfo &path_info,
                     Path &d_path,
                     Vector2f &d_pt) {
    auto base_point_id = path_info.base_point_id;
    auto point_id = path_info.point_id;
    auto min_t_root = path_info.t_root;
    
    if (path.num_control_points[base_point_id] == 0) {
        // Straight line
        auto i0 = point_id;
        auto i1 = (point_id + 1) % path.num_points;
        auto p0 = Vector2f{path.points[2 * i0], path.points[2 * i0 + 1]};
        auto p1 = Vector2f{path.points[2 * i1], path.points[2 * i1 + 1]};
        // project pt to line
        auto t = dot(pt - p0, p1 - p0) / dot(p1 - p0, p1 - p0);
        auto d_p0 = Vector2f{0, 0};
        auto d_p1 = Vector2f{0, 0};
        if (t < 0) {
            d_p0 += d_closest_pt;
        } else if (t > 1) {
            d_p1 += d_closest_pt;
        } else {
            auto d_p = d_closest_pt;
            // p = p0 + t * (p1 - p0)
            d_p0 += d_p * (1 - t);
            d_p1 += d_p * t;
        }
        atomic_add(d_path.points + 2 * i0, d_p0);
        atomic_add(d_path.points + 2 * i1, d_p1);
    } else if (path.num_control_points[base_point_id] == 1) {
        // Quadratic Bezier curve
        auto i0 = point_id;
        auto i1 = point_id + 1;
        auto i2 = (point_id + 2) % path.num_points;
        auto p0 = Vector2f{path.points[2 * i0], path.points[2 * i0 + 1]};
        auto p1 = Vector2f{path.points[2 * i1], path.points[2 * i1 + 1]};
        auto p2 = Vector2f{path.points[2 * i2], path.points[2 * i2 + 1]};
        // auto eval = [&](float t) -> Vector2f {
        //     auto tt = 1 - t;
        //     return (tt*tt)*p0 + (2*tt*t)*p1 + (t*t)*p2;
        // };
        // auto dist0 = distance(eval(0), pt);
        // auto dist1 = distance(eval(1), pt);
        auto d_p0 = Vector2f{0, 0};
        auto d_p1 = Vector2f{0, 0};
        auto d_p2 = Vector2f{0, 0};
        auto t = min_t_root;
        if (t == 0) {
            d_p0 += d_closest_pt;
        } else if (t == 1) {
            d_p2 += d_closest_pt;
        } else {
            // The curve is (1-t)^2p0 + 2(1-t)tp1 + t^2p2
            // = (p0-2p1+p2)t^2+(-2p0+2p1)t+p0 = q
            // Want to solve (q - pt) dot q' = 0
            // q' = (p0-2p1+p2)t + (-p0+p1)
            // Expanding (p0-2p1+p2)^2 t^3 +
            //           3(p0-2p1+p2)(-p0+p1) t^2 +
            //           (2(-p0+p1)^2+(p0-2p1+p2)(p0-pt))t +
            //           (-p0+p1)(p0-pt) = 0
            auto A = sum((p0-2*p1+p2)*(p0-2*p1+p2));
            auto B = sum(3*(p0-2*p1+p2)*(-p0+p1));
            auto C = sum(2*(-p0+p1)*(-p0+p1)+(p0-2*p1+p2)*(p0-pt));
            // auto D = sum((-p0+p1)*(p0-pt));
            auto d_p = d_closest_pt;
            // p = eval(t)
            auto tt = 1 - t;
            // (tt*tt)*p0 + (2*tt*t)*p1 + (t*t)*p2
            auto d_tt = 2 * tt * dot(d_p, p0) + 2 * t * dot(d_p, p1);
            auto d_t = -d_tt + 2 * tt * dot(d_p, p1) + 2 * t * dot(d_p, p2);
            auto d_p0 = d_p * tt * tt;
            auto d_p1 = 2 * d_p * tt * t;
            auto d_p2 = d_p * t * t;
            // implicit function theorem: dt/dA = -1/(p'(t)) * dp/dA
            auto poly_deriv_t = 3 * A * t * t + 2 * B * t + C;
            if (fabs(poly_deriv_t) > 1e-6f) {
                auto d_A = - (d_t / poly_deriv_t) * t * t * t;
                auto d_B = - (d_t / poly_deriv_t) * t * t;
                auto d_C = - (d_t / poly_deriv_t) * t;
                auto d_D = - (d_t / poly_deriv_t);
                // A = sum((p0-2*p1+p2)*(p0-2*p1+p2))
                // B = sum(3*(p0-2*p1+p2)*(-p0+p1))
                // C = sum(2*(-p0+p1)*(-p0+p1)+(p0-2*p1+p2)*(p0-pt))
                // D = sum((-p0+p1)*(p0-pt))
                d_p0 += 2*d_A*(p0-2*p1+p2)+
                        3*d_B*((-p0+p1)-(p0-2*p1+p2))+
                        2*d_C*(-2*(-p0+p1))+
                          d_C*((p0-pt)+(p0-2*p1+p2))+
                        2*d_D*(-(p0-pt)+(-p0+p1));
                d_p1 += (-2)*2*d_A*(p0-2*p1+p2)+
                        3*d_B*(-2*(-p0+p1)+(p0-2*p1+p2))+
                        2*d_C*(2*(-p0+p1))+
                          d_C*((-2)*(p0-pt))+
                        d_D*(p0-pt);
                d_p2 += 2*d_A*(p0-2*p1+p2)+
                        3*d_B*(-p0+p1)+
                        d_C*(p0-pt);
                d_pt += d_C*(-(p0-2*p1+p2))+
                        d_D*(-(-p0+p1));
            }
        }
        atomic_add(d_path.points + 2 * i0, d_p0);
        atomic_add(d_path.points + 2 * i1, d_p1);
        atomic_add(d_path.points + 2 * i2, d_p2);
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
        // auto eval = [&](float t) -> Vector2f {
        //     auto tt = 1 - t;
        //     return (tt*tt*tt)*p0 + (3*tt*tt*t)*p1 + (3*tt*t*t)*p2 + (t*t*t)*p3;
        // };
        auto d_p0 = Vector2f{0, 0};
        auto d_p1 = Vector2f{0, 0};
        auto d_p2 = Vector2f{0, 0};
        auto d_p3 = Vector2f{0, 0};
        auto t = min_t_root;
        if (t == 0) {
            // closest_pt = p0
            d_p0 += d_closest_pt;
        } else if (t == 1) {
            // closest_pt = p1
            d_p3 += d_closest_pt;
        } else {
            // The curve is (1 - t)^3 p0 + 3 * (1 - t)^2 t p1 + 3 * (1 - t) t^2 p2 + t^3 p3
            // = (-p0+3p1-3p2+p3) t^3 + (3p0-6p1+3p2) t^2 + (-3p0+3p1) t + p0
            // Want to solve (q - pt) dot q' = 0
            // q' = 3*(-p0+3p1-3p2+p3)t^2 + 2*(3p0-6p1+3p2)t + (-3p0+3p1)
            // Expanding 
            // 3*(-p0+3p1-3p2+p3)^2 t^5
            // 5*(-p0+3p1-3p2+p3)(3p0-6p1+3p2) t^4
            // 4*(-p0+3p1-3p2+p3)(-3p0+3p1) + 2*(3p0-6p1+3p2)^2 t^3
            // 3*(3p0-6p1+3p2)(-3p0+3p1) + 3*(-p0+3p1-3p2+p3)(p0-pt) t^2
            // (-3p0+3p1)^2+2(p0-pt)(3p0-6p1+3p2) t
            // (p0-pt)(-3p0+3p1)
            double A = 3*sum((-p0+3*p1-3*p2+p3)*(-p0+3*p1-3*p2+p3));
            double B = 5*sum((-p0+3*p1-3*p2+p3)*(3*p0-6*p1+3*p2));
            double C = 4*sum((-p0+3*p1-3*p2+p3)*(-3*p0+3*p1)) + 2*sum((3*p0-6*p1+3*p2)*(3*p0-6*p1+3*p2));
            double D = 3*(sum((3*p0-6*p1+3*p2)*(-3*p0+3*p1)) + sum((-p0+3*p1-3*p2+p3)*(p0-pt)));
            double E = sum((-3*p0+3*p1)*(-3*p0+3*p1)) + 2*sum((p0-pt)*(3*p0-6*p1+3*p2));
            double F = sum((p0-pt)*(-3*p0+3*p1));
            B /= A;
            C /= A;
            D /= A;
            E /= A;
            F /= A;
            // auto eval_polynomial = [&] (double t) {
            //     return t*t*t*t*t+
            //            B*t*t*t*t+
            //            C*t*t*t+
            //            D*t*t+
            //            E*t+
            //            F;
            // };
            auto eval_polynomial_deriv = [&] (double t) {
                return 5*t*t*t*t+
                       4*B*t*t*t+
                       3*C*t*t+
                       2*D*t+
                       E;
            };

            // auto p = eval(t);
            auto d_p = d_closest_pt;
            // (tt*tt*tt)*p0 + (3*tt*tt*t)*p1 + (3*tt*t*t)*p2 + (t*t*t)*p3
            auto tt = 1 - t;
            auto d_tt = 3 * tt * tt * dot(d_p, p0) +
                        6 * tt * t * dot(d_p, p1) +
                        3 * t * t * dot(d_p, p2);
            auto d_t = -d_tt +
                       3 * tt * tt * dot(d_p, p1) +
                       6 * tt * t * dot(d_p, p2) +
                       3 * t * t * dot(d_p, p3);
            d_p0 += d_p * (tt * tt * tt);
            d_p1 += d_p * (3 * tt * tt * t);
            d_p2 += d_p * (3 * tt * t * t);
            d_p3 += d_p * (t * t * t);
            // implicit function theorem: dt/dA = -1/(p'(t)) * dp/dA
            auto poly_deriv_t = eval_polynomial_deriv(t);
            if (fabs(poly_deriv_t) > 1e-10f) {
                auto d_B = -(d_t / poly_deriv_t) * t * t * t * t;
                auto d_C = -(d_t / poly_deriv_t) * t * t * t;
                auto d_D = -(d_t / poly_deriv_t) * t * t;
                auto d_E = -(d_t / poly_deriv_t) * t;
                auto d_F = -(d_t / poly_deriv_t);
                // B = B' / A
                // C = C' / A
                // D = D' / A
                // E = E' / A
                // F = F' / A
                auto d_A = -d_B * B / A
                           -d_C * C / A
                           -d_D * D / A
                           -d_E * E / A
                           -d_F * F / A;
                d_B /= A;
                d_C /= A;
                d_D /= A;
                d_E /= A;
                d_F /= A;
                {
                    double A = 3*sum((-p0+3*p1-3*p2+p3)*(-p0+3*p1-3*p2+p3)) + 1e-3;
                    double B = 5*sum((-p0+3*p1-3*p2+p3)*(3*p0-6*p1+3*p2));
                    double C = 4*sum((-p0+3*p1-3*p2+p3)*(-3*p0+3*p1)) + 2*sum((3*p0-6*p1+3*p2)*(3*p0-6*p1+3*p2));
                    double D = 3*(sum((3*p0-6*p1+3*p2)*(-3*p0+3*p1)) + sum((-p0+3*p1-3*p2+p3)*(p0-pt)));
                    double E = sum((-3*p0+3*p1)*(-3*p0+3*p1)) + 2*sum((p0-pt)*(3*p0-6*p1+3*p2));
                    double F = sum((p0-pt)*(-3*p0+3*p1));
                    B /= A;
                    C /= A;
                    D /= A;
                    E /= A;
                    F /= A;
                    auto eval_polynomial = [&] (double t) {
                        return t*t*t*t*t+
                               B*t*t*t*t+
                               C*t*t*t+
                               D*t*t+
                               E*t+
                               F;
                    };
                    auto eval_polynomial_deriv = [&] (double t) {
                        return 5*t*t*t*t+
                               4*B*t*t*t+
                               3*C*t*t+
                               2*D*t+
                               E;
                    };
                    auto lb = t - 1e-2f;
                    auto ub = t + 1e-2f;
                    auto lb_eval = eval_polynomial(lb);
                    auto ub_eval = eval_polynomial(ub);
                    if (lb_eval > ub_eval) {
                        swap_(lb, ub);
                    }
                    auto t_ = 0.5f * (lb + ub);
                    auto num_iter = 20;
                    for (int it = 0; it < num_iter; it++) {
                        if (!(t_ >= lb && t_ <= ub)) {
                            t_ = 0.5f * (lb + ub);
                        }
                        auto value = eval_polynomial(t_);
                        if (fabs(value) < 1e-5f || it == num_iter - 1) {
                            break;
                        }
                        // The derivative may not be entirely accurate,
                        // but the bisection is going to handle this
                        if (value > 0.f) {
                            ub = t_;
                        } else {
                            lb = t_;
                        }
                        auto derivative = eval_polynomial_deriv(t);
                        t_ -= value / derivative;
                    }
                }
                // A = 3*sum((-p0+3*p1-3*p2+p3)*(-p0+3*p1-3*p2+p3))
                d_p0 += d_A * 3 * (-1) * 2 * (-p0+3*p1-3*p2+p3);
                d_p1 += d_A * 3 *   3  * 2 * (-p0+3*p1-3*p2+p3);
                d_p2 += d_A * 3 * (-3) * 2 * (-p0+3*p1-3*p2+p3);
                d_p3 += d_A * 3 *   1  * 2 * (-p0+3*p1-3*p2+p3);
                // B = 5*sum((-p0+3*p1-3*p2+p3)*(3*p0-6*p1+3*p2))
                d_p0 += d_B * 5 * ((-1) * (3*p0-6*p1+3*p2) + 3 * (-p0+3*p1-3*p2+p3));
                d_p1 += d_B * 5 * (3 * (3*p0-6*p1+3*p2) + (-6) * (-p0+3*p1-3*p2+p3));
                d_p2 += d_B * 5 * ((-3) * (3*p0-6*p1+3*p2) + 3 * (-p0+3*p1-3*p2+p3));
                d_p3 += d_B * 5 * (3*p0-6*p1+3*p2);
                // C = 4*sum((-p0+3*p1-3*p2+p3)*(-3*p0+3*p1)) + 2*sum((3*p0-6*p1+3*p2)*(3*p0-6*p1+3*p2))
                d_p0 += d_C * 4 * ((-1) * (-3*p0+3*p1) + (-3) * (-p0+3*p1-3*p2+p3)) +
                        d_C * 2 * (3 * 2 * (3*p0-6*p1+3*p2));
                d_p1 += d_C * 4 * (3 * (-3*p0+3*p1) + 3 * (-p0+3*p1-3*p2+p3)) +
                        d_C * 2 * ((-6) * 2 * (3*p0-6*p1+3*p2));
                d_p2 += d_C * 4 * ((-3) * (-3*p0+3*p1)) +
                        d_C * 2 * (3 * 2 * (3*p0-6*p1+3*p2));
                d_p3 += d_C * 4 * (-3*p0+3*p1);
                // D = 3*(sum((3*p0-6*p1+3*p2)*(-3*p0+3*p1)) + sum((-p0+3*p1-3*p2+p3)*(p0-pt)))
                d_p0 += d_D * 3 * (3 * (-3*p0+3*p1) + (-3) * (3*p0-6*p1+3*p2)) +
                        d_D * 3 * ((-1) * (p0-pt) + 1 * (-p0+3*p1-3*p2+p3));
                d_p1 += d_D * 3 * ((-6) * (-3*p0+3*p1) + (3) * (3*p0-6*p1+3*p2)) +
                        d_D * 3 * (3 * (p0-pt));
                d_p2 += d_D * 3 * (3 * (-3*p0+3*p1)) +
                        d_D * 3 * ((-3) * (p0-pt));
                d_pt += d_D * 3 * ((-1) * (-p0+3*p1-3*p2+p3));
                // E = sum((-3*p0+3*p1)*(-3*p0+3*p1)) + 2*sum((p0-pt)*(3*p0-6*p1+3*p2))
                d_p0 += d_E * ((-3) * 2 * (-3*p0+3*p1)) +
                        d_E * 2 * (1 * (3*p0-6*p1+3*p2) + 3 * (p0-pt));
                d_p1 += d_E * (  3  * 2 * (-3*p0+3*p1)) +
                        d_E * 2 * ((-6) * (p0-pt));
                d_p2 += d_E * 2 * (  3  * (p0-pt));
                d_pt += d_E * 2 * ((-1) * (3*p0-6*p1+3*p2));
                // F = sum((p0-pt)*(-3*p0+3*p1))
                d_p0 += d_F * (1 * (-3*p0+3*p1)) +
                        d_F * ((-3) * (p0-pt));
                d_p1 += d_F * (3 * (p0-pt));
                d_pt += d_F * ((-1) * (-3*p0+3*p1));
            }
        }
        atomic_add(d_path.points + 2 * i0, d_p0);
        atomic_add(d_path.points + 2 * i1, d_p1);
        atomic_add(d_path.points + 2 * i2, d_p2);
        atomic_add(d_path.points + 2 * i3, d_p3);
    } else {
        assert(false);
    }
}

DEVICE
inline
void d_closest_point(const Rect &rect,
                     const Vector2f &pt,
                     const Vector2f &d_closest_pt,
                     Rect &d_rect,
                     Vector2f &d_pt) {
    auto dist = [&](const Vector2f &p0, const Vector2f &p1) -> float {
        // project pt to line
        auto t = dot(pt - p0, p1 - p0) / dot(p1 - p0, p1 - p0);
        if (t < 0) {
            return distance(p0, pt);
        } else if (t > 1) {
            return distance(p1, pt);
        } else {
            return distance(p0 + t * (p1 - p0), pt);
        }
        // return 0;
    };
    auto left_top = rect.p_min;
    auto right_top = Vector2f{rect.p_max.x, rect.p_min.y};
    auto left_bottom = Vector2f{rect.p_min.x, rect.p_max.y};
    auto right_bottom = rect.p_max;
    auto left_dist = dist(left_top, left_bottom);
    auto top_dist = dist(left_top, right_top);
    auto right_dist = dist(right_top, right_bottom);
    auto bottom_dist = dist(left_bottom, right_bottom);
    int min_id = 0;
    auto min_dist = left_dist;
    if (top_dist < min_dist) { min_dist = top_dist; min_id = 1; }
    if (right_dist < min_dist) { min_dist = right_dist; min_id = 2; }
    if (bottom_dist < min_dist) { min_dist = bottom_dist; min_id = 3; }

    auto d_update = [&](const Vector2f &p0, const Vector2f &p1,
                        const Vector2f &d_closest_pt,
                        Vector2f &d_p0, Vector2f &d_p1) {
        // project pt to line
        auto t = dot(pt - p0, p1 - p0) / dot(p1 - p0, p1 - p0);
        if (t < 0) {
            d_p0 += d_closest_pt;
        } else if (t > 1) {
            d_p1 += d_closest_pt;
        } else {
            // p = p0 + t * (p1 - p0)
            auto d_p = d_closest_pt;
            d_p0 += d_p * (1 - t);
            d_p1 += d_p * t;
            auto d_t = sum(d_p * (p1 - p0));
            // t = dot(pt - p0, p1 - p0) / dot(p1 - p0, p1 - p0)
            auto d_numerator = d_t / dot(p1 - p0, p1 - p0);
            auto d_denominator = d_t * (-t) / dot(p1 - p0, p1 - p0);
            // numerator = dot(pt - p0, p1 - p0)
            d_pt += (p1 - p0) * d_numerator;
            d_p1 += (pt - p0) * d_numerator;
            d_p0 += ((p0 - p1) + (p0 - pt)) * d_numerator;
            // denominator = dot(p1 - p0, p1 - p0)
            d_p1 += 2 * (p1 - p0) * d_denominator;
            d_p0 += 2 * (p0 - p1) * d_denominator;
        }
    };
    auto d_left_top = Vector2f{0, 0};
    auto d_right_top = Vector2f{0, 0};
    auto d_left_bottom = Vector2f{0, 0};
    auto d_right_bottom = Vector2f{0, 0};
    if (min_id == 0) {
        d_update(left_top, left_bottom, d_closest_pt, d_left_top, d_left_bottom);
    } else if (min_id == 1) {
        d_update(left_top, right_top, d_closest_pt, d_left_top, d_right_top);
    } else if (min_id == 2) {
        d_update(right_top, right_bottom, d_closest_pt, d_right_top, d_right_bottom);
    } else {
        assert(min_id == 3);
        d_update(left_bottom, right_bottom, d_closest_pt, d_left_bottom, d_right_bottom);
    }
    auto d_p_min = Vector2f{0, 0};
    auto d_p_max = Vector2f{0, 0};
    // left_top = rect.p_min
    // right_top = Vector2f{rect.p_max.x, rect.p_min.y}
    // left_bottom = Vector2f{rect.p_min.x, rect.p_max.y}
    // right_bottom = rect.p_max
    d_p_min += d_left_top;
    d_p_max.x += d_right_top.x;
    d_p_min.y += d_right_top.y;
    d_p_min.x += d_left_bottom.x;
    d_p_max.y += d_left_bottom.y;
    d_p_max += d_right_bottom;
    atomic_add(d_rect.p_min, d_p_min);
    atomic_add(d_rect.p_max, d_p_max);
}

DEVICE
inline
void d_closest_point(const Shape &shape,
                     const Vector2f &pt,
                     const Vector2f &d_closest_pt,
                     const ClosestPointPathInfo &path_info,
                     Shape &d_shape,
                     Vector2f &d_pt) {
    switch (shape.type) {
        case ShapeType::Circle:
            d_closest_point(*(const Circle *)shape.ptr,
                            pt,
                            d_closest_pt,
                            *(Circle *)d_shape.ptr,
                            d_pt);
            break;
        case ShapeType::Ellipse:
            // https://www.geometrictools.com/Documentation/DistancePointEllipseEllipsoid.pdf
            assert(false);
            break;
        case ShapeType::Path:
            d_closest_point(*(const Path *)shape.ptr,
                            pt,
                            d_closest_pt,
                            path_info,
                            *(Path *)d_shape.ptr,
                            d_pt);
            break;
        case ShapeType::Rect:
            d_closest_point(*(const Rect *)shape.ptr,
                            pt,
                            d_closest_pt,
                            *(Rect *)d_shape.ptr,
                            d_pt);
            break;
    }
}

DEVICE
inline
void d_compute_distance(const Matrix3x3f &canvas_to_shape,
                        const Matrix3x3f &shape_to_canvas,
                        const Shape &shape,
                        const Vector2f &pt,
                        const Vector2f &closest_pt,
                        const ClosestPointPathInfo &path_info,
                        float d_dist,
                        Matrix3x3f &d_shape_to_canvas,
                        Shape &d_shape,
                        float *d_translation) {
    if (distance_squared(pt, closest_pt) < 1e-10f) {
        // The derivative at distance=0 is undefined
        return;
    }
    assert(isfinite(d_dist));
    // pt is in canvas space, transform it to shape's local space
    auto local_pt = xform_pt(canvas_to_shape, pt);
    auto local_closest_pt = xform_pt(canvas_to_shape, closest_pt);
    // auto local_closest_pt = closest_point(shape, local_pt);
    // auto closest_pt = xform_pt(shape_group.shape_to_canvas, local_closest_pt);
    // auto dist = distance(closest_pt, pt);
    auto d_pt = Vector2f{0, 0};
    auto d_closest_pt = Vector2f{0, 0};
    d_distance(closest_pt, pt, d_dist, d_closest_pt, d_pt);
    assert(isfinite(d_pt));
    assert(isfinite(d_closest_pt));
    // auto closest_pt = xform_pt(shape_group.shape_to_canvas, local_closest_pt);
    auto d_local_closest_pt = Vector2f{0, 0};
    auto d_shape_to_canvas_ = Matrix3x3f();
    d_xform_pt(shape_to_canvas, local_closest_pt, d_closest_pt,
               d_shape_to_canvas_, d_local_closest_pt);
    assert(isfinite(d_local_closest_pt));
    auto d_local_pt = Vector2f{0, 0};
    d_closest_point(shape, local_pt, d_local_closest_pt, path_info, d_shape, d_local_pt);
    assert(isfinite(d_local_pt));
    auto d_canvas_to_shape = Matrix3x3f();
    d_xform_pt(canvas_to_shape,
               pt,
               d_local_pt,
               d_canvas_to_shape,
               d_pt);
    // http://jack.valmadre.net/notes/2016/09/04/back-prop-differentials/#back-propagation-using-differentials
    auto tc2s = transpose(canvas_to_shape);
    d_shape_to_canvas_ += -tc2s * d_canvas_to_shape * tc2s;
    atomic_add(&d_shape_to_canvas(0, 0), d_shape_to_canvas_);
    if (d_translation != nullptr) {
        atomic_add(d_translation, -d_pt);
    }
}

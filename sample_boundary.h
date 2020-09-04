#pragma once

#include "diffvg.h"
#include "shape.h"
#include "scene.h"
#include "vector.h"
#include "cdf.h"

struct PathBoundaryData {
    int base_point_id;
    int point_id;
    float t;
};

struct BoundaryData {
    PathBoundaryData path;
    bool is_stroke;
};

DEVICE
Vector2f sample_boundary(const Circle &circle,
                         float t,
                         Vector2f &normal,
                         float &pdf,
                         BoundaryData &,
                         float stroke_perturb_direction,
                         float stroke_radius) {
    // Parametric form of a circle (t in [0, 1)):
    // x = center.x + r * cos(2pi * t)
    // y = center.y + r * sin(2pi * t)
    auto offset = Vector2f{
        circle.radius * cos(2 * float(M_PI) * t),
        circle.radius * sin(2 * float(M_PI) * t)
    };
    normal = normalize(offset);
    pdf /= (2 * float(M_PI) * circle.radius);
    auto ret = circle.center + offset;
    if (stroke_perturb_direction != 0.f) {
        ret += stroke_perturb_direction * stroke_radius * normal;
        if (stroke_perturb_direction < 0) {
            // normal should point towards the perturb direction
            normal = -normal;
        }
    }
    return ret;
}

DEVICE
Vector2f sample_boundary(const Ellipse &ellipse,
                         float t,
                         Vector2f &normal,
                         float &pdf,
                         BoundaryData &,
                         float stroke_perturb_direction,
                         float stroke_radius) {
    // Parametric form of a ellipse (t in [0, 1)):
    // x = center.x + r.x * cos(2pi * t)
    // y = center.y + r.y * sin(2pi * t)
    const auto &r = ellipse.radius;
    auto offset = Vector2f{
        r.x * cos(2 * float(M_PI) * t),
        r.y * sin(2 * float(M_PI) * t)
    };
    auto dxdt = -r.x * sin(2 * float(M_PI) * t) * 2 * float(M_PI);
    auto dydt = r.y * cos(2 * float(M_PI) * t) * 2 * float(M_PI);
    // tangent is normalize(dxdt, dydt)
    normal = normalize(Vector2f{dydt, -dxdt});
    pdf /= sqrt(square(dxdt) + square(dydt));
    auto ret = ellipse.center + offset;
    if (stroke_perturb_direction != 0.f) {
        ret += stroke_perturb_direction * stroke_radius * normal;
        if (stroke_perturb_direction < 0) {
            // normal should point towards the perturb direction
            normal = -normal;
        }
    }
    return ret;
}

DEVICE
Vector2f sample_boundary(const Path &path,
                         const float *path_length_cdf,
                         const float *path_length_pmf,
                         const int *point_id_map,
                         float path_length,
                         float t,
                         Vector2f &normal,
                         float &pdf,
                         BoundaryData &data,
                         float stroke_perturb_direction,
                         float stroke_radius) {
    if (stroke_perturb_direction != 0.f && !path.is_closed) {
        // We need to samples the "caps" of the path
        // length of a cap is pi * abs(stroke_perturb_direction)
        // there are two caps
        auto cap_length = 0.f;
        if (path.thickness != nullptr) {
            auto r0 = path.thickness[0];
            auto r1 = path.thickness[path.num_points - 1];
            cap_length = float(M_PI) * (r0 + r1);
        } else {
            cap_length = 2 * float(M_PI) * stroke_radius;
        }
        auto cap_prob = cap_length / (cap_length + path_length);
        if (t < cap_prob) {
            t = t / cap_prob;
            pdf *= cap_prob;
            auto r0 = stroke_radius;
            auto r1 = stroke_radius;
            if (path.thickness != nullptr) {
                r0 = path.thickness[0];
                r1 = path.thickness[path.num_points - 1];
            }
            // HACK: in theory we want to compute the tangent and
            //       sample the hemi-circle, but here we just sample the
            //       full circle since it's less typing
            if (stroke_perturb_direction < 0) {
                // Sample the cap at the beginning
                auto p0 = Vector2f{path.points[0], path.points[1]};
                auto offset = Vector2f{
                    r0 * cos(2 * float(M_PI) * t),
                    r0 * sin(2 * float(M_PI) * t)
                };
                normal = normalize(offset);
                pdf /= (2 * float(M_PI) * r0);
                data.path.base_point_id = 0;
                data.path.point_id = 0;
                data.path.t = 0;
                return p0 + offset;
            } else {
                // Sample the cap at the end
                auto p0 = Vector2f{path.points[2 * (path.num_points - 1)],
                                   path.points[2 * (path.num_points - 1) + 1]};
                auto offset = Vector2f{
                    r1 * cos(2 * float(M_PI) * t),
                    r1 * sin(2 * float(M_PI) * t)
                };
                normal = normalize(offset);
                pdf /= (2 * float(M_PI) * r1);
                data.path.base_point_id = path.num_base_points - 1;
                data.path.point_id = path.num_points - 2 - 
                                     path.num_control_points[data.path.base_point_id];
                data.path.t = 1;
                return p0 + offset;
            }
        } else {
            t = (t - cap_prob) / (1 - cap_prob);
            pdf *= (1 - cap_prob);
        }
    }
    // Binary search on path_length_cdf
    auto sample_id = sample(path_length_cdf,
                            path.num_base_points,
                            t,
                            &t);
    assert(sample_id >= 0 && sample_id < path.num_base_points);
    auto point_id = point_id_map[sample_id];
    if (path.num_control_points[sample_id] == 0) {
        // Straight line
        auto i0 = point_id;
        auto i1 = (i0 + 1) % path.num_points;
        assert(i0 < path.num_points);
        auto p0 = Vector2f{path.points[2 * i0], path.points[2 * i0 + 1]};
        auto p1 = Vector2f{path.points[2 * i1], path.points[2 * i1 + 1]};
        data.path.base_point_id = sample_id;
        data.path.point_id = point_id;
        data.path.t = t;
        if (t < -1e-3f || t > 1+1e-3f) {
            // return invalid sample
            pdf = 0;
            return Vector2f{0, 0};
        }
        auto tangent = (p1 - p0);
        auto tan_len = length(tangent);
        if (tan_len == 0) {
            // return invalid sample
            pdf = 0;
            return Vector2f{0, 0};
        }
        normal = Vector2f{-tangent.y, tangent.x} / tan_len;
        // length of tangent is the Jacobian of the sampling transformation
        pdf *= path_length_pmf[sample_id] / tan_len;
        auto ret = p0 + t * (p1 - p0);
        if (stroke_perturb_direction != 0.f) {
            auto r0 = stroke_radius;
            auto r1 = stroke_radius;
            if (path.thickness != nullptr) {
                r0 = path.thickness[i0];
                r1 = path.thickness[i1];
            }
            auto r = r0 + t * (r1 - r0);
            ret += stroke_perturb_direction * r * normal;
            if (stroke_perturb_direction < 0) {
                // normal should point towards the perturb direction
                normal = -normal;
            }
        }
        return ret;
    } else if (path.num_control_points[sample_id] == 1) {
        // Quadratic Bezier curve
        auto i0 = point_id;
        auto i1 = i0 + 1;
        auto i2 = (i0 + 2) % path.num_points;
        auto p0 = Vector2f{path.points[2 * i0], path.points[2 * i0 + 1]};
        auto p1 = Vector2f{path.points[2 * i1], path.points[2 * i1 + 1]};
        auto p2 = Vector2f{path.points[2 * i2], path.points[2 * i2 + 1]};
        auto eval = [&](float t) -> Vector2f {
            auto tt = 1 - t;
            return (tt*tt)*p0 + (2*tt*t)*p1 + (t*t)*p2;
        };
        data.path.base_point_id = sample_id;
        data.path.point_id = point_id;
        data.path.t = t;
        if (t < -1e-3f || t > 1+1e-3f) {
            // return invalid sample
            pdf = 0;
            return Vector2f{0, 0};
        }
        auto tangent = 2 * (1 - t) * (p1 - p0) + 2 * t * (p2 - p1);
        auto tan_len = length(tangent);
        if (tan_len == 0) {
            // return invalid sample
            pdf = 0;
            return Vector2f{0, 0};
        }
        normal = Vector2f{-tangent.y, tangent.x} / tan_len;
        // length of tangent is the Jacobian of the sampling transformation
        pdf *= path_length_pmf[sample_id] / tan_len;
        auto ret = eval(t);
        if (stroke_perturb_direction != 0.f) {
            auto r0 = stroke_radius;
            auto r1 = stroke_radius;
            auto r2 = stroke_radius;
            if (path.thickness != nullptr) {
                r0 = path.thickness[i0];
                r1 = path.thickness[i1];
                r2 = path.thickness[i2];
            }
            auto tt = 1 - t;
            auto r = (tt*tt)*r0 + (2*tt*t)*r1 + (t*t)*r2;
            ret += stroke_perturb_direction * r * normal;
            if (stroke_perturb_direction < 0) {
                // normal should point towards the perturb direction
                normal = -normal;
            }
        }
        return ret;
    } else if (path.num_control_points[sample_id] == 2) {
        // Cubic Bezier curve
        auto i0 = point_id;
        auto i1 = point_id + 1;
        auto i2 = point_id + 2;
        auto i3 = (point_id + 3) % path.num_points;
        assert(i0 >= 0 && i2 < path.num_points);
        auto p0 = Vector2f{path.points[2 * i0], path.points[2 * i0 + 1]};
        auto p1 = Vector2f{path.points[2 * i1], path.points[2 * i1 + 1]};
        auto p2 = Vector2f{path.points[2 * i2], path.points[2 * i2 + 1]};
        auto p3 = Vector2f{path.points[2 * i3], path.points[2 * i3 + 1]};
        auto eval = [&](float t) -> Vector2f {
            auto tt = 1 - t;
            return (tt*tt*tt)*p0 + (3*tt*tt*t)*p1 + (3*tt*t*t)*p2 + (t*t*t)*p3;
        };
        data.path.base_point_id = sample_id;
        data.path.point_id = point_id;
        data.path.t = t;
        if (t < -1e-3f || t > 1+1e-3f) {
            // return invalid sample
            pdf = 0;
            return Vector2f{0, 0};
        }
        auto tangent = 3 * square(1 - t) * (p1 - p0) + 6 * (1 - t) * t * (p2 - p1) + 3 * t * t * (p3 - p2);
        auto tan_len = length(tangent);
        if (tan_len == 0) {
            // return invalid sample
            pdf = 0;
            return Vector2f{0, 0};
        }
        normal = Vector2f{-tangent.y, tangent.x} / tan_len;
        // length of tangent is the Jacobian of the sampling transformation
        pdf *= path_length_pmf[sample_id] / tan_len;
        auto ret = eval(t);
        if (stroke_perturb_direction != 0.f) {
            auto r0 = stroke_radius;
            auto r1 = stroke_radius;
            auto r2 = stroke_radius;
            auto r3 = stroke_radius;
            if (path.thickness != nullptr) {
                r0 = path.thickness[i0];
                r1 = path.thickness[i1];
                r2 = path.thickness[i2];
                r3 = path.thickness[i3];
            }
            auto tt = 1 - t;
            auto r = (tt*tt*tt)*r0 + (3*tt*tt*t)*r1 + (3*tt*t*t)*r2 + (t*t*t)*r3;
            ret += stroke_perturb_direction * r * normal;
            if (stroke_perturb_direction < 0) {
                // normal should point towards the perturb direction
                normal = -normal;
            }
        }
        return ret;
    } else {
        assert(false);
    }
    assert(false);
    return Vector2f{0, 0};
}

DEVICE
Vector2f sample_boundary(const Rect &rect,
                         float t, Vector2f &normal,
                         float &pdf,
                         BoundaryData &,
                         float stroke_perturb_direction,
                         float stroke_radius) {
    // Roll a dice to decide whether to sample width or height
    auto w = rect.p_max.x - rect.p_min.x;
    auto h = rect.p_max.y - rect.p_min.y;
    pdf /= (2 * (w +h));
    if (t <= w / (w + h)) {
        // Sample width
        // reuse t for the next dice
        t *= (w + h) / w;
        // Roll a dice to decide whether to sample upper width or lower width
        if (t < 0.5f) {
            // Sample upper width
            normal = Vector2f{0, -1};
            auto ret = rect.p_min + 2 * t * Vector2f{rect.p_max.x - rect.p_min.x, 0.f};
            if (stroke_perturb_direction != 0.f) {
                ret += stroke_perturb_direction * stroke_radius * normal;
                if (stroke_perturb_direction < 0) {
                    // normal should point towards the perturb direction
                    normal = -normal;
                }
            }
            return ret;
        } else {
            // Sample lower width
            normal = Vector2f{0, 1};
            auto ret = Vector2f{rect.p_min.x, rect.p_max.y} +
                2 * (t - 0.5f) * Vector2f{rect.p_max.x - rect.p_min.x, 0.f};
            if (stroke_perturb_direction != 0.f) {
                ret += stroke_perturb_direction * stroke_radius * normal;
                if (stroke_perturb_direction < 0) {
                    // normal should point towards the perturb direction
                    normal = -normal;
                }
            }
            return ret;
        }
    } else {
        // Sample height
        // reuse t for the next dice
        assert(h > 0);
        t = (t - w / (w + h)) * (w + h) / h;
        // Roll a dice to decide whether to sample left height or right height
        if (t < 0.5f) {
            // Sample left height
            normal = Vector2f{-1, 0};
            auto ret = rect.p_min + 2 * t * Vector2f{0.f, rect.p_max.y - rect.p_min.y};
            if (stroke_perturb_direction != 0.f) {
                ret += stroke_perturb_direction * stroke_radius * normal;
                if (stroke_perturb_direction < 0) {
                    // normal should point towards the perturb direction
                    normal = -normal;
                }
            }
            return ret;
        } else {
            // Sample right height
            normal = Vector2f{1, 0};
            auto ret = Vector2f{rect.p_max.x, rect.p_min.y} +
                2 * (t - 0.5f) * Vector2f{0.f, rect.p_max.y - rect.p_min.y};
            if (stroke_perturb_direction != 0.f) {
                ret += stroke_perturb_direction * stroke_radius * normal;
                if (stroke_perturb_direction < 0) {
                    // normal should point towards the perturb direction
                    normal = -normal;
                }
            }
            return ret;
        }
    }
}

DEVICE
Vector2f sample_boundary(const SceneData &scene,
                         int shape_group_id,
                         int shape_id,
                         float t,
                         Vector2f &normal,
                         float &pdf,
                         BoundaryData &data) {
    const ShapeGroup &shape_group = scene.shape_groups[shape_group_id];
    const Shape &shape = scene.shapes[shape_id];
    pdf = 1;
    // Choose which one to sample: stroke discontinuities or fill discontinuities.
    // TODO: we don't need to sample fill discontinuities when stroke alpha is 1 and both
    // fill and stroke color exists
    auto stroke_perturb = false;
    if (shape_group.fill_color != nullptr && shape_group.stroke_color != nullptr) {
        if (t < 0.5f) {
            stroke_perturb = false;
            t = 2 * t;
            pdf = 0.5f;
        } else {
            stroke_perturb = true;
            t = 2 * (t - 0.5f);
            pdf = 0.5f;
        }
    } else if (shape_group.stroke_color != nullptr) {
        stroke_perturb = true;
    }
    data.is_stroke = stroke_perturb;
    auto stroke_perturb_direction = 0.f;
    if (stroke_perturb) {
        if (t < 0.5f) {
            stroke_perturb_direction = -1.f;
            t = 2 * t;
            pdf *= 0.5f;
        } else {
            stroke_perturb_direction = 1.f;
            t = 2 * (t - 0.5f);
            pdf *= 0.5f;
        }
    }
    switch (shape.type) {
        case ShapeType::Circle:
            return sample_boundary(
                *(const Circle *)shape.ptr, t, normal, pdf, data, stroke_perturb_direction, shape.stroke_width);
        case ShapeType::Ellipse:
            return sample_boundary(
                *(const Ellipse *)shape.ptr, t, normal, pdf, data, stroke_perturb_direction, shape.stroke_width);
        case ShapeType::Path:
            return sample_boundary(
                *(const Path *)shape.ptr,
                scene.path_length_cdf[shape_id],
                scene.path_length_pmf[shape_id],
                scene.path_point_id_map[shape_id],
                scene.shapes_length[shape_id],
                t,
                normal,
                pdf,
                data,
                stroke_perturb_direction,
                shape.stroke_width);
        case ShapeType::Rect:
            return sample_boundary(
                *(const Rect *)shape.ptr, t, normal, pdf, data, stroke_perturb_direction, shape.stroke_width);
    }
    assert(false);
    return Vector2f{};
}


#include "scene.h"
#include "aabb.h"
#include "cuda_utils.h"
#include "filter.h"
#include "shape.h"
#include <numeric>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <cstddef>

size_t align(size_t s) {
    auto a = alignof(std::max_align_t);
    return ((s + a - 1) / a) * a;
}

template <typename T>
void allocate(bool use_gpu, T **p) {
    if (use_gpu) {
#ifdef __NVCC__
        checkCuda(cudaMallocManaged(p, sizeof(T)));
#else
        throw std::runtime_error("diffvg not compiled with GPU");
        assert(false);
#endif
    } else {
        *p = (T*)malloc(sizeof(T));
    }
}

template <typename T>
void allocate(bool use_gpu, size_t size, T **p) {
    if (use_gpu) {
#ifdef __NVCC__
        checkCuda(cudaMallocManaged(p, size * sizeof(T)));
#else
        throw std::runtime_error("diffvg not compiled with GPU");
        assert(false);
#endif
    } else {
        *p = (T*)malloc(size * sizeof(T));
    }
}

void copy_and_init_shapes(Scene &scene,
                          const std::vector<const Shape *> &shape_list) {
    for (int shape_id = 0; shape_id < scene.num_shapes; shape_id++) {
        switch (shape_list[shape_id]->type) {
            case ShapeType::Circle: {
                Circle *p = (Circle *)scene.shapes[shape_id].ptr;
                const Circle *p_ = (const Circle*)(shape_list[shape_id]->ptr);
                *p = *p_;
                Circle *d_p = (Circle *)scene.d_shapes[shape_id].ptr;
                d_p->radius = 0;
                d_p->center = Vector2f{0, 0};
                break;
            } case ShapeType::Ellipse: {
                Ellipse *p = (Ellipse *)scene.shapes[shape_id].ptr;
                const Ellipse *p_ = (const Ellipse*)(shape_list[shape_id]->ptr);
                *p = *p_;
                Ellipse *d_p = (Ellipse *)scene.d_shapes[shape_id].ptr;
                d_p->radius = Vector2f{0, 0};
                d_p->center = Vector2f{0, 0};
                break;
            } case ShapeType::Path: {
                Path *p = (Path *)scene.shapes[shape_id].ptr;
                const Path *p_ = (const Path*)(shape_list[shape_id]->ptr);
                p->num_points = p_->num_points;
                p->num_base_points = p_->num_base_points;
                for (int i = 0; i < p_->num_base_points; i++) {
                    p->num_control_points[i] = p_->num_control_points[i];
                }
                for (int i = 0; i < 2 * p_->num_points; i++) {
                    p->points[i] = p_->points[i];
                }
                p->is_closed = p_->is_closed;
                p->use_distance_approx = p_->use_distance_approx;
                Path *d_p = (Path *)scene.d_shapes[shape_id].ptr;
                d_p->num_points = p_->num_points;
                d_p->num_base_points = p_->num_base_points;
                for (int i = 0; i < 2 * p_->num_points; i++) {
                    d_p->points[i] = 0;
                }
                d_p->is_closed = p_->is_closed;
                if (p_->thickness != nullptr) {
                    for (int i = 0; i < p_->num_points; i++) {
                        p->thickness[i] = p_->thickness[i];
                        d_p->thickness[i] = 0;
                    }
                }
                d_p->use_distance_approx = p_->use_distance_approx;
                break;
            } case ShapeType::Rect: {
                Rect *p = (Rect *)scene.shapes[shape_id].ptr;
                const Rect *p_ = (const Rect*)(shape_list[shape_id]->ptr);
                *p = *p_;
                Rect *d_p = (Rect *)scene.d_shapes[shape_id].ptr;
                d_p->p_min = Vector2f{0, 0};
                d_p->p_max = Vector2f{0, 0};
                break;
            } default: {
                assert(false);
                break;
            }
        }
        scene.shapes[shape_id].type = shape_list[shape_id]->type;
        scene.shapes[shape_id].stroke_width = shape_list[shape_id]->stroke_width;
        scene.d_shapes[shape_id].type = shape_list[shape_id]->type;
        scene.d_shapes[shape_id].stroke_width = 0;
    }
}

std::vector<float>
compute_shape_length(const std::vector<const Shape *> &shape_list) {
    int num_shapes = (int)shape_list.size();
    std::vector<float> shape_length_list(num_shapes, 0.f);
    for (int shape_id = 0; shape_id < num_shapes; shape_id++) {
        auto shape_length = 0.f;
        switch (shape_list[shape_id]->type) {
            case ShapeType::Circle: {
                const Circle *p_ = (const Circle*)(shape_list[shape_id]->ptr);
                shape_length += float(2.f * M_PI) * p_->radius;
                break;
            } case ShapeType::Ellipse: {
                const Ellipse *p_ = (const Ellipse*)(shape_list[shape_id]->ptr);
                // https://en.wikipedia.org/wiki/Ellipse#Circumference
                // Ramanujan's ellipse circumference approximation
                auto a = p_->radius.x;
                auto b = p_->radius.y;
                shape_length += float(M_PI) * (3 * (a + b) - sqrt((3 * a + b) * (a + 3 * b)));
                break;
            } case ShapeType::Path: {
                const Path *p_ = (const Path*)(shape_list[shape_id]->ptr);
                auto length = 0.f;
                auto point_id = 0;
                for (int i = 0; i < p_->num_base_points; i++) {
                    if (p_->num_control_points[i] == 0) {
                        // Straight line
                        auto i0 = point_id;
                        assert(i0 < p_->num_points);
                        auto i1 = (i0 + 1) % p_->num_points;
                        point_id += 1;
                        auto p0 = Vector2f{p_->points[2 * i0], p_->points[2 * i0 + 1]};
                        auto p1 = Vector2f{p_->points[2 * i1], p_->points[2 * i1 + 1]};
                        length += distance(p1, p0);
                    } else if (p_->num_control_points[i] == 1) {
                        // Quadratic Bezier curve
                        auto i0 = point_id;
                        auto i1 = i0 + 1;
                        auto i2 = (i0 + 2) % p_->num_points;
                        point_id += 2;
                        auto p0 = Vector2f{p_->points[2 * i0], p_->points[2 * i0 + 1]};
                        auto p1 = Vector2f{p_->points[2 * i1], p_->points[2 * i1 + 1]};
                        auto p2 = Vector2f{p_->points[2 * i2], p_->points[2 * i2 + 1]};
                        auto eval = [&](float t) -> Vector2f {
                            auto tt = 1 - t;
                            return (tt*tt)*p0 + (2*tt*t)*p1 + (t*t)*p2;
                        };
                        // We use 3-point samples to approximate the length
                        auto v0 = p0;
                        auto v1 = eval(0.5f);
                        auto v2 = p2;
                        length += distance(v1, v0) + distance(v1, v2);
                    } else if (p_->num_control_points[i] == 2) {
                        // Cubic Bezier curve
                        auto i0 = point_id;
                        auto i1 = i0 + 1;
                        auto i2 = i0 + 2;
                        auto i3 = (i0 + 3) % p_->num_points;
                        point_id += 3;
                        auto p0 = Vector2f{p_->points[2 * i0], p_->points[2 * i0 + 1]};
                        auto p1 = Vector2f{p_->points[2 * i1], p_->points[2 * i1 + 1]};
                        auto p2 = Vector2f{p_->points[2 * i2], p_->points[2 * i2 + 1]};
                        auto p3 = Vector2f{p_->points[2 * i3], p_->points[2 * i3 + 1]};
                        auto eval = [&](float t) -> Vector2f {
                            auto tt = 1 - t;
                            return (tt*tt*tt)*p0 + (3*tt*tt*t)*p1 + (3*tt*t*t)*p2 + (t*t*t)*p3;
                        };
                        // We use 4-point samples to approximate the length
                        auto v0 = p0;
                        auto v1 = eval(1.f/3.f);
                        auto v2 = eval(2.f/3.f);
                        auto v3 = p3;
                        length += distance(v1, v0) + distance(v1, v2) + distance(v2, v3);
                    } else {
                        assert(false);
                    }
                }
                assert(isfinite(length));
                shape_length += length;
                break;
            } case ShapeType::Rect: {
                const Rect *p_ = (const Rect*)(shape_list[shape_id]->ptr);
                shape_length += 2 * (p_->p_max.x - p_->p_min.x + p_->p_max.y - p_->p_min.y);
                break;
            } default: {
                assert(false);
                break;
            }
        }
        assert(isfinite(shape_length));
        shape_length_list[shape_id] = shape_length;
    }
    return shape_length_list;
}

void build_shape_cdfs(Scene &scene,
                      const std::vector<const ShapeGroup *> &shape_group_list,
                      const std::vector<float> &shape_length_list) {
    int sample_id = 0;
    for (int shape_group_id = 0; shape_group_id < (int)shape_group_list.size(); shape_group_id++) {
        const ShapeGroup *shape_group = shape_group_list[shape_group_id];
        for (int i = 0; i < shape_group->num_shapes; i++) {
            int shape_id = shape_group->shape_ids[i];
            float length = shape_length_list[shape_id];
            scene.sample_shape_id[sample_id] = shape_id;
            if (sample_id == 0) {
                scene.sample_shapes_cdf[sample_id] = length;
            } else {
                scene.sample_shapes_cdf[sample_id] = length +
                    scene.sample_shapes_cdf[sample_id - 1];
            }
            assert(isfinite(length));
            scene.sample_shapes_pmf[sample_id] = length;
            scene.sample_group_id[sample_id] = shape_group_id;
            sample_id++;
        }
    }
    assert(sample_id == scene.num_total_shapes);
    auto normalization = scene.sample_shapes_cdf[scene.num_total_shapes - 1];
    if (normalization <= 0) {
        char buf[256];
        sprintf(buf, "The total length of the shape boundaries in the scene is equal or less than 0. Length = %f", normalization);
        throw std::runtime_error(buf);
    }
    if (!isfinite(normalization)) {
        char buf[256];
        sprintf(buf, "The total length of the shape boundaries in the scene is not a number. Length = %f", normalization);
        throw std::runtime_error(buf);
    }
    assert(normalization > 0);
    for (int sample_id = 0; sample_id < scene.num_total_shapes; sample_id++) {
        scene.sample_shapes_cdf[sample_id] /= normalization;
        scene.sample_shapes_pmf[sample_id] /= normalization;
    }
}

void build_path_cdfs(Scene &scene,
                     const std::vector<const Shape *> &shape_list,
                     const std::vector<float> &shape_length_list) {
    for (int shape_id = 0; shape_id < scene.num_shapes; shape_id++) {
        if (shape_list[shape_id]->type == ShapeType::Path) {
            const Path &path = shape_list[shape_id]->as_path();
            float *pmf = scene.path_length_pmf[shape_id];
            float *cdf = scene.path_length_cdf[shape_id];
            int *point_id_map = scene.path_point_id_map[shape_id];
            auto path_length = shape_length_list[shape_id];
            auto inv_length = 1.f / path_length;
            auto point_id = 0;
            for (int i = 0; i < path.num_base_points; i++) {
                point_id_map[i] = point_id;
                if (path.num_control_points[i] == 0) {
                    // Straight line
                    auto i0 = point_id;
                    auto i1 = (i0 + 1) % path.num_points;
                    point_id += 1;
                    auto p0 = Vector2f{path.points[2 * i0], path.points[2 * i0 + 1]};
                    auto p1 = Vector2f{path.points[2 * i1], path.points[2 * i1 + 1]};
                    auto d = distance(p0, p1) * inv_length;
                    pmf[i] = d;
                    if (i == 0) {
                        cdf[i] = d;
                    } else {
                        cdf[i] = d + cdf[i - 1];
                    }
                } else if (path.num_control_points[i] == 1) {
                    // Quadratic Bezier curve
                    auto i0 = point_id;
                    auto i1 = i0 + 1;
                    auto i2 = (i0 + 2) % path.num_points;
                    point_id += 2;
                    auto p0 = Vector2f{path.points[2 * i0], path.points[2 * i0 + 1]};
                    auto p1 = Vector2f{path.points[2 * i1], path.points[2 * i1 + 1]};
                    auto p2 = Vector2f{path.points[2 * i2], path.points[2 * i2 + 1]};
                    auto eval = [&](float t) -> Vector2f {
                        auto tt = 1 - t;
                        return (tt*tt)*p0 + (2*tt*t)*p1 + (t*t)*p2;
                    };
                    // We use 3-point samples to approximate the length
                    auto v0 = p0;
                    auto v1 = eval(0.5f);
                    auto v2 = p2;
                    auto d = (distance(v0, v1) + distance(v1, v2)) * inv_length;
                    pmf[i] = d;
                    if (i == 0) {
                        cdf[i] = d;
                    } else {
                        cdf[i] = d + cdf[i - 1];
                    }
                } else if (path.num_control_points[i] == 2) {
                    // Cubic Bezier curve
                    auto i0 = point_id;
                    auto i1 = point_id + 1;
                    auto i2 = point_id + 2;
                    auto i3 = (point_id + 3) % path.num_points;
                    point_id += 3;
                    auto p0 = Vector2f{path.points[2 * i0], path.points[2 * i0 + 1]};
                    auto p1 = Vector2f{path.points[2 * i1], path.points[2 * i1 + 1]};
                    auto p2 = Vector2f{path.points[2 * i2], path.points[2 * i2 + 1]};
                    auto p3 = Vector2f{path.points[2 * i3], path.points[2 * i3 + 1]};
                    auto eval = [&](float t) -> Vector2f {
                        auto tt = 1 - t;
                        return (tt*tt*tt)*p0 + (3*tt*tt*t)*p1 + (3*tt*t*t)*p2 + (t*t*t)*p3;
                    };
                    // We use 4-point samples to approximate the length
                    auto v0 = p0;
                    auto v1 = eval(1.f/3.f);
                    auto v2 = eval(2.f/3.f);
                    auto v3 = p3;
                    auto d = (distance(v1, v0) + distance(v1, v2) + distance(v2, v3)) * inv_length;
                    pmf[i] = d;
                    if (i == 0) {
                        cdf[i] = d;
                    } else {
                        cdf[i] = d + cdf[i - 1];
                    }
                } else {
                    assert(false);
                }
            }
        }
    }
}

void copy_and_init_shape_groups(Scene &scene,
                                const std::vector<const ShapeGroup *> &shape_group_list) {
    for (int group_id = 0; group_id < scene.num_shape_groups; group_id++) {
        const ShapeGroup *shape_group = shape_group_list[group_id];
        auto copy_and_init_color = [&](const ColorType &color_type, void *color_ptr, void *target_ptr, void *d_target_ptr) {
            switch (color_type) {
                case ColorType::Constant: {
                    Constant *c = (Constant*)target_ptr;
                    Constant *d_c = (Constant*)d_target_ptr;
                    const Constant *c_ = (const Constant*)color_ptr;
                    *c = *c_;
                    d_c->color = Vector4{0, 0, 0, 0};
                    break;
                } case ColorType::LinearGradient: {
                    LinearGradient *c = (LinearGradient*)target_ptr;
                    LinearGradient *d_c = (LinearGradient*)d_target_ptr;
                    const LinearGradient *c_ = (const LinearGradient*)color_ptr;
                    c->begin = c_->begin;
                    c->end = c_->end;
                    c->num_stops = c_->num_stops;
                    for (int i = 0; i < c_->num_stops; i++) {
                        c->stop_offsets[i] = c_->stop_offsets[i];
                    }
                    for (int i = 0; i < 4 * c_->num_stops; i++) {
                        c->stop_colors[i] = c_->stop_colors[i];
                    }
                    d_c->begin = Vector2f{0, 0};
                    d_c->end = Vector2f{0, 0};
                    d_c->num_stops = c_->num_stops;
                    for (int i = 0; i < c_->num_stops; i++) {
                        d_c->stop_offsets[i] = 0;
                    }
                    for (int i = 0; i < 4 * c_->num_stops; i++) {
                        d_c->stop_colors[i] = 0;
                    }
                    break;
                } case ColorType::RadialGradient: {
                    RadialGradient *c = (RadialGradient*)target_ptr;
                    RadialGradient *d_c = (RadialGradient*)d_target_ptr;
                    const RadialGradient *c_ = (const RadialGradient*)color_ptr;
                    c->center = c_->center;
                    c->radius = c_->radius;
                    c->num_stops = c_->num_stops;
                    for (int i = 0; i < c_->num_stops; i++) {
                        c->stop_offsets[i] = c_->stop_offsets[i];
                    }
                    for (int i = 0; i < 4 * c_->num_stops; i++) {
                        c->stop_colors[i] = c_->stop_colors[i];
                    }
                    d_c->center = Vector2f{0, 0};
                    d_c->radius = Vector2f{0, 0};
                    d_c->num_stops = c_->num_stops;
                    for (int i = 0; i < c_->num_stops; i++) {
                        d_c->stop_offsets[i] = 0;
                    }
                    for (int i = 0; i < 4 * c_->num_stops; i++) {
                        d_c->stop_colors[i] = 0;
                    }
                    break;
                } default: {
                    assert(false);
                }
            }
        };
        for (int i = 0; i < shape_group->num_shapes; i++) {
            scene.shape_groups[group_id].shape_ids[i] = shape_group->shape_ids[i];
        }
        scene.shape_groups[group_id].num_shapes = shape_group->num_shapes;
        scene.shape_groups[group_id].use_even_odd_rule = shape_group->use_even_odd_rule;
        scene.shape_groups[group_id].canvas_to_shape = shape_group->canvas_to_shape;
        scene.shape_groups[group_id].shape_to_canvas = shape_group->shape_to_canvas;
        scene.d_shape_groups[group_id].shape_ids = nullptr;
        scene.d_shape_groups[group_id].num_shapes = shape_group->num_shapes;
        scene.d_shape_groups[group_id].use_even_odd_rule = shape_group->use_even_odd_rule;
        scene.d_shape_groups[group_id].canvas_to_shape = Matrix3x3f{};
        scene.d_shape_groups[group_id].shape_to_canvas = Matrix3x3f{};

        scene.shape_groups[group_id].fill_color_type = shape_group->fill_color_type;
        scene.d_shape_groups[group_id].fill_color_type = shape_group->fill_color_type;
        if (shape_group->fill_color != nullptr) {
            copy_and_init_color(shape_group->fill_color_type,
                                shape_group->fill_color,
                                scene.shape_groups[group_id].fill_color,
                                scene.d_shape_groups[group_id].fill_color);
        }
        scene.shape_groups[group_id].stroke_color_type = shape_group->stroke_color_type;
        scene.d_shape_groups[group_id].stroke_color_type = shape_group->stroke_color_type;
        if (shape_group->stroke_color != nullptr) {
            copy_and_init_color(shape_group->stroke_color_type,
                                shape_group->stroke_color,
                                scene.shape_groups[group_id].stroke_color,
                                scene.d_shape_groups[group_id].stroke_color);
        }
    }
}

DEVICE uint32_t morton2D(const Vector2f &p, int canvas_width, int canvas_height) {
    auto scene_bounds = Vector2f{canvas_width, canvas_height};
    auto pp = p / scene_bounds;
    TVector2<uint32_t> pp_i{pp.x * 1023, pp.y * 1023};
    return (expand_bits(pp_i.x) << 1u) |
           (expand_bits(pp_i.y) << 0u);
}

template <bool sort>
void build_bvh(const Scene &scene, BVHNode *nodes, int num_primitives) {
    auto bvh_size = 2 * num_primitives - 1;
    if (bvh_size > 1) {
        if (sort) {
            // Sort by Morton code
            std::sort(nodes, nodes + num_primitives,
                [&] (const BVHNode &n0, const BVHNode &n1) {
                    auto p0 = 0.5f * (n0.box.p_min + n0.box.p_max);
                    auto p1 = 0.5f * (n1.box.p_min + n1.box.p_max);
                    auto m0 = morton2D(p0, scene.canvas_width, scene.canvas_height);
                    auto m1 = morton2D(p1, scene.canvas_width, scene.canvas_height);
                    return m0 < m1;
            });
        }
        for (int i = num_primitives; i < bvh_size; i++) {
            nodes[i] = BVHNode{-1, -1, AABB{}, 0.f};
        }
        int prev_beg = 0;
        int prev_end = num_primitives;
        // For handling odd number of nodes at a level
        int leftover = prev_end % 2 == 0 ? -1 : prev_end - 1;
        while (prev_end - prev_beg >= 1 || leftover != -1) {
            int length = (prev_end - prev_beg) / 2;
            if ((prev_end - prev_beg) % 2 == 1 && leftover != -1 &&
                    leftover != prev_end - 1) {
                length += 1;
            }
            for (int i = 0; i < length; i++) {
                BVHNode node;
                node.child0 = prev_beg + 2 * i;
                node.child1 = prev_beg + 2 * i + 1;
                if (node.child1 >= prev_end) {
                    assert(leftover != -1);
                    node.child1 = leftover;
                    leftover = -1;
                }
                AABB child0_box = nodes[node.child0].box;
                AABB child1_box = nodes[node.child1].box;
                node.box = merge(child0_box, child1_box);
                node.max_radius = std::max(nodes[node.child0].max_radius,
                                           nodes[node.child1].max_radius);
                nodes[prev_end + i] = node;
            }
            if (length == 1 && leftover == -1) {
                break;
            }
            prev_beg = prev_end;
            prev_end = prev_beg + length;
            if (length % 2 == 1 && leftover == -1) {
                leftover = prev_end - 1;
            }
        }
    }
    assert(nodes[2 * num_primitives - 2].child0 != -1);
}

void compute_bounding_boxes(Scene &scene,
                            const std::vector<const Shape *> &shape_list,
                            const std::vector<const ShapeGroup *> &shape_group_list) {
    for (int shape_id = 0; shape_id < scene.num_shapes; shape_id++) {
        switch (shape_list[shape_id]->type) {
            case ShapeType::Circle: {
                const Circle *p = (const Circle*)(shape_list[shape_id]->ptr);
                scene.shapes_bbox[shape_id] = AABB{p->center - p->radius,
                                                   p->center + p->radius};
                break;
            } case ShapeType::Ellipse: {
                const Ellipse *p = (const Ellipse*)(shape_list[shape_id]->ptr);
                scene.shapes_bbox[shape_id] = AABB{p->center - p->radius,
                                                   p->center + p->radius};
                break;
            } case ShapeType::Path: {
                const Path *p = (const Path*)(shape_list[shape_id]->ptr);
                AABB box;
                if (p->num_points > 0) {
                    box = AABB{Vector2f{p->points[0], p->points[1]},
                               Vector2f{p->points[0], p->points[1]}};
                }
                for (int i = 1; i < p->num_points; i++) {
                    box = merge(box, Vector2f{p->points[2 * i], p->points[2 * i + 1]});
                }
                scene.shapes_bbox[shape_id] = box;
                std::vector<AABB> boxes(p->num_base_points);
                std::vector<float> thickness(p->num_base_points);
                std::vector<int> first_point_id(p->num_base_points);
                auto r = shape_list[shape_id]->stroke_width;
                auto point_id = 0;
                for (int i = 0; i < p->num_base_points; i++) {
                    first_point_id[i] = point_id;
                    if (p->num_control_points[i] == 0) {
                        // Straight line
                        auto i0 = point_id;
                        auto i1 = (i0 + 1) % p->num_points;
                        point_id += 1;
                        auto p0 = Vector2f{p->points[2 * i0], p->points[2 * i0 + 1]};
                        auto p1 = Vector2f{p->points[2 * i1], p->points[2 * i1 + 1]};
                        boxes[i] = AABB();
                        boxes[i] = merge(boxes[i], p0);
                        boxes[i] = merge(boxes[i], p1);
                        auto r0 = r;
                        auto r1 = r;
                        // override radius if path has thickness
                        if (p->thickness != nullptr) {
                            r0 = p->thickness[i0];
                            r1 = p->thickness[i1];
                        }
                        thickness[i] = max(r0, r1);
                    } else if (p->num_control_points[i] == 1) {
                        // Quadratic Bezier curve
                        auto i0 = point_id;
                        auto i1 = i0 + 1;
                        auto i2 = (i0 + 2) % p->num_points;
                        point_id += 2;
                        auto p0 = Vector2f{p->points[2 * i0], p->points[2 * i0 + 1]};
                        auto p1 = Vector2f{p->points[2 * i1], p->points[2 * i1 + 1]};
                        auto p2 = Vector2f{p->points[2 * i2], p->points[2 * i2 + 1]};
                        boxes[i] = AABB();
                        boxes[i] = merge(boxes[i], p0);
                        boxes[i] = merge(boxes[i], p1);
                        boxes[i] = merge(boxes[i], p2);
                        auto r0 = r;
                        auto r1 = r;
                        auto r2 = r;
                        // override radius if path has thickness
                        if (p->thickness != nullptr) {
                            r0 = p->thickness[i0];
                            r1 = p->thickness[i1];
                            r2 = p->thickness[i2];
                        }
                        thickness[i] = max(max(r0, r1), r2);
                    } else if (p->num_control_points[i] == 2) {
                        // Cubic Bezier curve
                        auto i0 = point_id;
                        auto i1 = i0 + 1;
                        auto i2 = i0 + 2;
                        auto i3 = (i0 + 3) % p->num_points;
                        point_id += 3;
                        auto p0 = Vector2f{p->points[2 * i0], p->points[2 * i0 + 1]};
                        auto p1 = Vector2f{p->points[2 * i1], p->points[2 * i1 + 1]};
                        auto p2 = Vector2f{p->points[2 * i2], p->points[2 * i2 + 1]};
                        auto p3 = Vector2f{p->points[2 * i3], p->points[2 * i3 + 1]};
                        boxes[i] = AABB();
                        boxes[i] = merge(boxes[i], p0);
                        boxes[i] = merge(boxes[i], p1);
                        boxes[i] = merge(boxes[i], p2);
                        boxes[i] = merge(boxes[i], p3);
                        auto r0 = r;
                        auto r1 = r;
                        auto r2 = r;
                        auto r3 = r;
                        // override radius if path has thickness
                        if (p->thickness != nullptr) {
                            r0 = p->thickness[i0];
                            r1 = p->thickness[i1];
                            r2 = p->thickness[i2];
                            r3 = p->thickness[i3];
                        }
                        thickness[i] = max(max(max(r0, r1), r2), r3);
                    } else {
                        assert(false);
                    }
                }
                // Sort the boxes by y
                std::vector<int> idx(boxes.size());
                std::iota(idx.begin(), idx.end(), 0);
                std::sort(idx.begin(), idx.end(), [&](int i0, int i1) {
                    const AABB &b0 = boxes[i0];
                    const AABB &b1 = boxes[i1];
                    auto b0y = 0.5f * (b0.p_min.y + b0.p_max.y);
                    auto b1y = 0.5f * (b1.p_min.y + b1.p_max.y);
                    return b0y < b1y;
                });
                BVHNode *nodes = scene.path_bvhs[shape_id];
                for (int i = 0; i < (int)idx.size(); i++) {
                    nodes[i] = BVHNode{idx[i],
                                       -(first_point_id[idx[i]]+1),
                                       boxes[idx[i]],
                                       thickness[idx[i]]};
                }
                build_bvh<false /*sort*/>(scene, nodes, boxes.size());
                break;
            } case ShapeType::Rect: {
                const Rect *p = (const Rect*)(shape_list[shape_id]->ptr);
                scene.shapes_bbox[shape_id] = AABB{p->p_min, p->p_max};
                break;
            } default: {
                assert(false);
                break;
            }
        }
    }
    
    for (int shape_group_id = 0; shape_group_id < (int)shape_group_list.size(); shape_group_id++) {
        const ShapeGroup *shape_group = shape_group_list[shape_group_id];
        // Build a BVH for each shape group
        BVHNode *nodes = scene.shape_groups_bvh_nodes[shape_group_id];
        for (int i = 0; i < shape_group->num_shapes; i++) {
            auto shape_id = shape_group->shape_ids[i];
            auto r = shape_group->stroke_color == nullptr ? 0 : shape_list[shape_id]->stroke_width;
            nodes[i] = BVHNode{shape_id,
                               -1,
                               scene.shapes_bbox[shape_id],
                               r};
        }
        build_bvh<true /*sort*/>(scene, nodes, shape_group->num_shapes);
    }

    BVHNode *nodes = scene.bvh_nodes;
    for (int shape_group_id = 0; shape_group_id < (int)shape_group_list.size(); shape_group_id++) {
        const ShapeGroup *shape_group = shape_group_list[shape_group_id];
        auto max_radius = shape_list[shape_group->shape_ids[0]]->stroke_width;
        if (shape_list[shape_group->shape_ids[0]]->type == ShapeType::Path) {
            const Path *p = (const Path*)(shape_list[shape_group->shape_ids[0]]->ptr);
            if (p->thickness != nullptr) {
                const BVHNode *nodes = scene.path_bvhs[shape_group->shape_ids[0]];
                max_radius = nodes[0].max_radius;
            }
        }
        for (int i = 1; i < shape_group->num_shapes; i++) {
            auto shape_id = shape_group->shape_ids[i];
            auto shape = shape_list[shape_id];
            auto r = shape->stroke_width;
            if (shape->type == ShapeType::Path) {
                const Path *p = (const Path*)(shape_list[shape_id]->ptr);
                if (p->thickness != nullptr) {
                    const BVHNode *nodes = scene.path_bvhs[shape_id];
                    r = nodes[0].max_radius;
                }
            }
            max_radius = std::max(max_radius, r);
        }
        // Fetch group bbox from BVH
        auto bbox = scene.shape_groups_bvh_nodes[shape_group_id][2 * shape_group->num_shapes - 2].box;
        // Transform box from local to world space
        nodes[shape_group_id].child0 = shape_group_id;
        nodes[shape_group_id].child1 = -1;
        nodes[shape_group_id].box = transform(shape_group->shape_to_canvas, bbox);
        if (shape_group->stroke_color == nullptr) {
            nodes[shape_group_id].max_radius = 0;
        } else {
            nodes[shape_group_id].max_radius = max_radius;
        }
    }
    build_bvh<true /*sort*/>(scene, nodes, shape_group_list.size());
}

template <bool alloc_mode>
size_t allocate_buffers(Scene &scene,
                        const std::vector<const Shape *> &shape_list,
                        const std::vector<const ShapeGroup *> &shape_group_list) {
    auto num_shapes = shape_list.size();
    auto num_shape_groups = shape_group_list.size();

    size_t buffer_size = 0;
    if (alloc_mode) scene.shapes = (Shape*)&scene.buffer[buffer_size];
    buffer_size += align(sizeof(Shape) * num_shapes);
    if (alloc_mode) scene.d_shapes = (Shape*)&scene.buffer[buffer_size];
    buffer_size += align(sizeof(Shape) * num_shapes); 
    if (alloc_mode) scene.shape_groups = (ShapeGroup*)&scene.buffer[buffer_size];
    buffer_size += align(sizeof(ShapeGroup) * num_shape_groups);
    if (alloc_mode) scene.d_shape_groups = (ShapeGroup*)&scene.buffer[buffer_size];
    buffer_size += align(sizeof(ShapeGroup) * num_shape_groups);
    if (alloc_mode) scene.sample_shapes_cdf = (float*)&scene.buffer[buffer_size];
    buffer_size += align(sizeof(float) * scene.num_total_shapes);
    if (alloc_mode) scene.sample_shapes_pmf = (float*)&scene.buffer[buffer_size];
    buffer_size += align(sizeof(float) * scene.num_total_shapes);
    if (alloc_mode) scene.sample_shape_id = (int*)&scene.buffer[buffer_size];
    buffer_size += align(sizeof(int) * scene.num_total_shapes);
    if (alloc_mode) scene.sample_group_id = (int*)&scene.buffer[buffer_size];
    buffer_size += align(sizeof(int) * scene.num_total_shapes);
    if (alloc_mode) scene.shapes_length = (float*)&scene.buffer[buffer_size];
    buffer_size += align(sizeof(float) * num_shapes);
    if (alloc_mode) scene.path_length_cdf = (float**)&scene.buffer[buffer_size];
    buffer_size += align(sizeof(float*) * num_shapes);
    if (alloc_mode) scene.path_length_pmf = (float**)&scene.buffer[buffer_size];
    buffer_size += align(sizeof(float*) * num_shapes);
    if (alloc_mode) scene.path_point_id_map = (int**)&scene.buffer[buffer_size];
    buffer_size += align(sizeof(int*) * num_shapes);
    if (alloc_mode) scene.filter = (Filter*)&scene.buffer[buffer_size];
    buffer_size += align(sizeof(Filter));
    if (alloc_mode) scene.d_filter = (DFilter*)&scene.buffer[buffer_size];
    buffer_size += align(sizeof(DFilter));
    if (alloc_mode) scene.shapes_bbox = (AABB*)&scene.buffer[buffer_size];
    buffer_size += align(sizeof(AABB) * num_shapes);
    if (alloc_mode) scene.path_bvhs = (BVHNode**)&scene.buffer[buffer_size];
    buffer_size += align(sizeof(BVHNode*) * num_shapes);
    if (alloc_mode) scene.shape_groups_bvh_nodes = (BVHNode**)&scene.buffer[buffer_size];
    buffer_size += align(sizeof(BVHNode*) * num_shape_groups);
    if (alloc_mode) scene.bvh_nodes = (BVHNode*)&scene.buffer[buffer_size];
    buffer_size += align(sizeof(BVHNode) * (2 * num_shape_groups - 1));

    if (alloc_mode) {
        for (int i = 0; i < num_shapes; i++) {
            scene.path_length_cdf[i] = nullptr;
            scene.path_length_pmf[i] = nullptr;
            scene.path_point_id_map[i] = nullptr;
            scene.path_bvhs[i] = nullptr;
        }
    }

    for (int shape_id = 0; shape_id < scene.num_shapes; shape_id++) {
        switch (shape_list[shape_id]->type) {
            case ShapeType::Circle: {
                if (alloc_mode) scene.shapes[shape_id].ptr = (Circle*)&scene.buffer[buffer_size];
                buffer_size += align(sizeof(Circle)); // scene.shapes[shape_id].ptr
                if (alloc_mode) scene.d_shapes[shape_id].ptr = (Circle*)&scene.buffer[buffer_size];
                buffer_size += align(sizeof(Circle)); // scene.d_shapes[shape_id].ptr
                break;
            } case ShapeType::Ellipse: {
                if (alloc_mode) scene.shapes[shape_id].ptr = (Ellipse*)&scene.buffer[buffer_size];
                buffer_size += align(sizeof(Ellipse)); // scene.shapes[shape_id].ptr
                if (alloc_mode) scene.d_shapes[shape_id].ptr = (Ellipse*)&scene.buffer[buffer_size];
                buffer_size += align(sizeof(Ellipse)); // scene.d_shapes[shape_id].ptr
                break;
            } case ShapeType::Path: {
                if (alloc_mode) scene.shapes[shape_id].ptr = (Path*)&scene.buffer[buffer_size];
                buffer_size += align(sizeof(Path)); // scene.shapes[shape_id].ptr
                if (alloc_mode) scene.d_shapes[shape_id].ptr = (Path*)&scene.buffer[buffer_size];
                buffer_size += align(sizeof(Path)); // scene.d_shapes[shape_id].ptr

                const Path *p_ = (const Path*)(shape_list[shape_id]->ptr);
                Path *p = nullptr, *d_p = nullptr;
                if (alloc_mode) p = (Path*)scene.shapes[shape_id].ptr;
                if (alloc_mode) d_p = (Path*)scene.d_shapes[shape_id].ptr; 
                if (alloc_mode) p->num_control_points = (int*)&scene.buffer[buffer_size];
                buffer_size += align(sizeof(int) * p_->num_base_points); // p->num_control_points
                if (alloc_mode) p->points = (float*)&scene.buffer[buffer_size];
                buffer_size += align(sizeof(float) * (2 * p_->num_points)); // p->points
                if (alloc_mode) d_p->points = (float*)&scene.buffer[buffer_size];
                buffer_size += align(sizeof(float) * (2 * p_->num_points)); // d_p->points
                if (p_->thickness != nullptr) {
                    if (alloc_mode) p->thickness = (float*)&scene.buffer[buffer_size];
                    buffer_size += align(sizeof(float) * p_->num_points); // p->thickness
                    if (alloc_mode) d_p->thickness = (float*)&scene.buffer[buffer_size];
                    buffer_size += align(sizeof(float) * p_->num_points); // d_p->thickness
                } else {
                    if (alloc_mode) p->thickness = nullptr;
                    if (alloc_mode) d_p->thickness = nullptr;
                }
                if (alloc_mode) scene.path_length_pmf[shape_id] = (float*)&scene.buffer[buffer_size];
                buffer_size += align(sizeof(float) * p_->num_base_points); // scene.path_length_pmf
                if (alloc_mode) scene.path_length_cdf[shape_id] = (float*)&scene.buffer[buffer_size];
                buffer_size += align(sizeof(float) * p_->num_base_points); // scene.path_length_cdf
                if (alloc_mode) scene.path_point_id_map[shape_id] = (int*)&scene.buffer[buffer_size];
                buffer_size += align(sizeof(int) * p_->num_base_points); // scene.path_point_id_map
                if (alloc_mode) scene.path_bvhs[shape_id] = (BVHNode*)&scene.buffer[buffer_size];
                buffer_size += align(sizeof(BVHNode) * (2 * p_->num_base_points - 1));
                break;
            } case ShapeType::Rect: {
                if (alloc_mode) scene.shapes[shape_id].ptr = (Ellipse*)&scene.buffer[buffer_size];
                buffer_size += align(sizeof(Rect)); // scene.shapes[shape_id].ptr
                if (alloc_mode) scene.d_shapes[shape_id].ptr = (Ellipse*)&scene.buffer[buffer_size];
                buffer_size += align(sizeof(Rect)); // scene.d_shapes[shape_id].ptr
                break;
            } default: {
                assert(false);
                break;
            }
        }
    }

    for (int group_id = 0; group_id < scene.num_shape_groups; group_id++) {
        const ShapeGroup *shape_group = shape_group_list[group_id];
        if (shape_group->fill_color != nullptr) {
            switch (shape_group->fill_color_type) {
                case ColorType::Constant: {
                    if (alloc_mode) scene.shape_groups[group_id].fill_color = (Constant*)&scene.buffer[buffer_size];
                    buffer_size += align(sizeof(Constant)); // color
                    if (alloc_mode) scene.d_shape_groups[group_id].fill_color = (Constant*)&scene.buffer[buffer_size];
                    buffer_size += align(sizeof(Constant)); // d_color
                    break;
                } case ColorType::LinearGradient: {
                    if (alloc_mode) scene.shape_groups[group_id].fill_color = (LinearGradient*)&scene.buffer[buffer_size];
                    buffer_size += align(sizeof(LinearGradient)); // color
                    if (alloc_mode) scene.d_shape_groups[group_id].fill_color = (LinearGradient*)&scene.buffer[buffer_size];
                    buffer_size += align(sizeof(LinearGradient)); // d_color

                    const LinearGradient *c_ = (const LinearGradient *)shape_group->fill_color;
                    LinearGradient *c = nullptr, *d_c = nullptr;
                    if (alloc_mode) c = (LinearGradient *)scene.shape_groups[group_id].fill_color;
                    if (alloc_mode) d_c = (LinearGradient *)scene.d_shape_groups[group_id].fill_color;
                    if (alloc_mode) c->stop_offsets = (float*)&scene.buffer[buffer_size];
                    buffer_size += align(sizeof(float) * c_->num_stops); // c->stop_offsets
                    if (alloc_mode) c->stop_colors = (float*)&scene.buffer[buffer_size];
                    buffer_size += align(sizeof(float) * 4 * c_->num_stops); // c->stop_colors
                    if (alloc_mode) d_c->stop_offsets = (float*)&scene.buffer[buffer_size];
                    buffer_size += align(sizeof(float) * c_->num_stops); // d_c->stop_offsets
                    if (alloc_mode) d_c->stop_colors = (float*)&scene.buffer[buffer_size];
                    buffer_size += align(sizeof(float) * 4 * c_->num_stops); // d_c->stop_colors
                    break;
                } case ColorType::RadialGradient: {
                    if (alloc_mode) scene.shape_groups[group_id].fill_color = (RadialGradient*)&scene.buffer[buffer_size];
                    buffer_size += align(sizeof(RadialGradient)); // color
                    if (alloc_mode) scene.d_shape_groups[group_id].fill_color = (RadialGradient*)&scene.buffer[buffer_size];
                    buffer_size += align(sizeof(RadialGradient)); // d_color

                    const RadialGradient *c_ = (const RadialGradient *)shape_group->fill_color;
                    RadialGradient *c = nullptr, *d_c = nullptr;
                    if (alloc_mode) c = (RadialGradient *)scene.shape_groups[group_id].fill_color;
                    if (alloc_mode) d_c = (RadialGradient *)scene.d_shape_groups[group_id].fill_color;
                    if (alloc_mode) c->stop_offsets = (float*)&scene.buffer[buffer_size];
                    buffer_size += align(sizeof(float) * c_->num_stops); // c->stop_offsets
                    if (alloc_mode) c->stop_colors = (float*)&scene.buffer[buffer_size];
                    buffer_size += align(sizeof(float) * 4 * c_->num_stops); // c->stop_colors
                    if (alloc_mode) d_c->stop_offsets = (float*)&scene.buffer[buffer_size];
                    buffer_size += align(sizeof(float) * c_->num_stops); // d_c->stop_offsets
                    if (alloc_mode) d_c->stop_colors = (float*)&scene.buffer[buffer_size];
                    buffer_size += align(sizeof(float) * 4 * c_->num_stops); // d_c->stop_colors
                    break;
                } default: {
                    assert(false);
                }
            }
        } else {
            if (alloc_mode) scene.shape_groups[group_id].fill_color = nullptr;
            if (alloc_mode) scene.d_shape_groups[group_id].fill_color = nullptr;
        }
        if (shape_group->stroke_color != nullptr) {
            switch (shape_group->stroke_color_type) {
                case ColorType::Constant: {
                    if (alloc_mode) scene.shape_groups[group_id].stroke_color = (Constant*)&scene.buffer[buffer_size];
                    buffer_size += align(sizeof(Constant)); // color
                    if (alloc_mode) scene.d_shape_groups[group_id].stroke_color = (Constant*)&scene.buffer[buffer_size];
                    buffer_size += align(sizeof(Constant)); // d_color
                    break;
                } case ColorType::LinearGradient: {
                    if (alloc_mode) scene.shape_groups[group_id].stroke_color = (LinearGradient*)&scene.buffer[buffer_size];
                    buffer_size += align(sizeof(LinearGradient)); // color
                    if (alloc_mode) scene.shape_groups[group_id].stroke_color = (LinearGradient*)&scene.buffer[buffer_size];
                    buffer_size += align(sizeof(LinearGradient)); // d_color

                    const LinearGradient *c_ = (const LinearGradient *)shape_group->stroke_color;
                    LinearGradient *c = nullptr, *d_c = nullptr;
                    if (alloc_mode) c = (LinearGradient *)scene.shape_groups[group_id].stroke_color;
                    if (alloc_mode) d_c = (LinearGradient *)scene.d_shape_groups[group_id].stroke_color;
                    if (alloc_mode) c->stop_offsets = (float*)&scene.buffer[buffer_size];
                    buffer_size += align(sizeof(float) * c_->num_stops); // c->stop_offsets
                    if (alloc_mode) c->stop_colors = (float*)&scene.buffer[buffer_size];
                    buffer_size += align(sizeof(float) * 4 * c_->num_stops); // c->stop_colors
                    if (alloc_mode) d_c->stop_offsets = (float*)&scene.buffer[buffer_size];
                    buffer_size += align(sizeof(float) * c_->num_stops); // d_c->stop_offsets
                    if (alloc_mode) d_c->stop_colors = (float*)&scene.buffer[buffer_size];
                    buffer_size += align(sizeof(float) * 4 * c_->num_stops); // d_c->stop_colors
                    break;
                } case ColorType::RadialGradient: {
                    if (alloc_mode) scene.shape_groups[group_id].stroke_color = (RadialGradient*)&scene.buffer[buffer_size];
                    buffer_size += align(sizeof(RadialGradient)); // color
                    if (alloc_mode) scene.shape_groups[group_id].stroke_color = (RadialGradient*)&scene.buffer[buffer_size];
                    buffer_size += align(sizeof(RadialGradient)); // d_color

                    const RadialGradient *c_ = (const RadialGradient *)shape_group->stroke_color;
                    RadialGradient *c = nullptr, *d_c = nullptr;
                    if (alloc_mode) c = (RadialGradient *)scene.shape_groups[group_id].stroke_color;
                    if (alloc_mode) d_c = (RadialGradient *)scene.d_shape_groups[group_id].stroke_color;
                    if (alloc_mode) c->stop_offsets = (float*)&scene.buffer[buffer_size];
                    buffer_size += align(sizeof(float) * c_->num_stops); // c->stop_offsets
                    if (alloc_mode) c->stop_colors = (float*)&scene.buffer[buffer_size];
                    buffer_size += align(sizeof(float) * 4 * c_->num_stops); // c->stop_colors
                    if (alloc_mode) d_c->stop_offsets = (float*)&scene.buffer[buffer_size];
                    buffer_size += align(sizeof(float) * c_->num_stops); // d_c->stop_offsets
                    if (alloc_mode) d_c->stop_colors = (float*)&scene.buffer[buffer_size];
                    buffer_size += align(sizeof(float) * 4 * c_->num_stops); // d_c->stop_colors
                    break;
                } default: {
                    assert(false);
                }
            }
        } else {
            if (alloc_mode) scene.shape_groups[group_id].stroke_color = nullptr;
            if (alloc_mode) scene.d_shape_groups[group_id].stroke_color = nullptr;
        }
        if (alloc_mode) scene.shape_groups[group_id].shape_ids = (int*)&scene.buffer[buffer_size];
        buffer_size += align(sizeof(int) * shape_group->num_shapes); // shape_group->shape_ids
        if (alloc_mode) scene.shape_groups_bvh_nodes[group_id] = (BVHNode*)&scene.buffer[buffer_size];
        buffer_size += align(sizeof(BVHNode) * (2 * shape_group->num_shapes - 1)); // scene.shape_groups_bvh_nodes[group_id]
    }
    return buffer_size;
}

Scene::Scene(int canvas_width,
             int canvas_height,
             const std::vector<const Shape *> &shape_list,
             const std::vector<const ShapeGroup *> &shape_group_list,
             const Filter &filter,
             bool use_gpu,
             int gpu_index)
    : canvas_width(canvas_width),
      canvas_height(canvas_height),
      num_shapes(shape_list.size()),
      num_shape_groups(shape_group_list.size()),
      use_gpu(use_gpu),
      gpu_index(gpu_index) {
    if (num_shapes == 0) {
        return;
    }
    // Shape group may reuse some of the shapes,
    // record the total number of shapes.
    int num_total_shapes = 0;
    for (const ShapeGroup *sg : shape_group_list) {
        num_total_shapes += sg->num_shapes;
    }
    this->num_total_shapes = num_total_shapes;

    // Memory initialization
#ifdef __NVCC__
    int old_device_id = -1;
#endif
    if (use_gpu) {
#ifdef __NVCC__
        checkCuda(cudaGetDevice(&old_device_id));
        if (gpu_index != -1) {
            checkCuda(cudaSetDevice(gpu_index));
        }
#else
        throw std::runtime_error("diffvg not compiled with GPU");
        assert(false);
#endif
    }

    size_t buffer_size = allocate_buffers<false /*alloc_mode*/>(*this, shape_list, shape_group_list);
    // Allocate a huge buffer for everything
    allocate<uint8_t>(use_gpu, buffer_size, &buffer);
    // memset(buffer, 111, buffer_size);
    // Actually distribute the buffer
    allocate_buffers<true /*alloc_mode*/>(*this, shape_list, shape_group_list);
    copy_and_init_shapes(*this, shape_list);
    copy_and_init_shape_groups(*this, shape_group_list);

    std::vector<float> shape_length_list = compute_shape_length(shape_list);
    // Copy shape_length
    if (use_gpu) {
#ifdef __NVCC__
        checkCuda(cudaMemcpy(this->shapes_length, &shape_length_list[0], num_shapes * sizeof(float), cudaMemcpyHostToDevice));
#else
        throw std::runtime_error("diffvg not compiled with GPU");
        assert(false);
#endif
    } else {
        memcpy(this->shapes_length, &shape_length_list[0], num_shapes * sizeof(float));
    }
    build_shape_cdfs(*this, shape_group_list, shape_length_list);
    build_path_cdfs(*this, shape_list, shape_length_list);
    compute_bounding_boxes(*this, shape_list, shape_group_list);

    // Filter initialization
    *(this->filter) = filter;
    this->d_filter->radius = 0;

    if (use_gpu) {
#ifdef __NVCC__
        if (old_device_id != -1) {
            checkCuda(cudaSetDevice(old_device_id));
        }
#else
        throw std::runtime_error("diffvg not compiled with GPU");
        assert(false);
#endif
    }
}

Scene::~Scene() {
    if (num_shapes == 0) {
        return;
    }
    if (use_gpu) {
#ifdef __NVCC__
        int old_device_id = -1;
        checkCuda(cudaGetDevice(&old_device_id));
        if (gpu_index != -1) {
            checkCuda(cudaSetDevice(gpu_index));
        }

        checkCuda(cudaFree(buffer));

        checkCuda(cudaSetDevice(old_device_id));
#else
        // Don't throw because C++ don't want a destructor to throw.
        std::cerr << "diffvg not compiled with GPU";
        exit(1);
#endif
    } else {
        free(buffer);
    }
}

Shape Scene::get_d_shape(int shape_id) const {
    return d_shapes[shape_id];
}

ShapeGroup Scene::get_d_shape_group(int group_id) const {
    return d_shape_groups[group_id];
}

float Scene::get_d_filter_radius() const {
    return d_filter->radius;
}

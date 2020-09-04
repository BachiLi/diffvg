#pragma once

#include "diffvg.h"
#include "color.h"
#include "ptr.h"
#include "vector.h"
#include "matrix.h"

enum class ShapeType {
    Circle,
    Ellipse,
    Path,
    Rect
};

struct Circle {
    float radius;
    Vector2f center;

    ptr<void> get_ptr() {
        return ptr<void>(this);
    }
};

struct Ellipse {
    Vector2f radius;
    Vector2f center;

    ptr<void> get_ptr() {
        return ptr<void>(this);
    }
};

struct Path {
    Path(ptr<int> num_control_points,
         ptr<float> points,
         ptr<float> thickness,
         int num_base_points,
         int num_points,
         bool is_closed,
         bool use_distance_approx) :
        num_control_points(num_control_points.get()),
        points(points.get()),
        thickness(thickness.get()),
        num_base_points(num_base_points),
        num_points(num_points),
        is_closed(is_closed),
        use_distance_approx(use_distance_approx) {}

    int *num_control_points;
    float *points;
    float *thickness;
    int num_base_points;
    int num_points;
    bool is_closed;
    bool use_distance_approx;

    bool has_thickness() const {
        return thickness != nullptr;
    }
    void copy_to(ptr<float> points, ptr<float> thickness) const;

    ptr<void> get_ptr() {
        return ptr<void>(this);
    }
};

struct Rect {
    Vector2f p_min;
    Vector2f p_max;

    ptr<void> get_ptr() {
        return ptr<void>(this);
    }
};

struct Shape {
    Shape() {}
    Shape(const ShapeType &type,
          ptr<void> shape_ptr,
          float stroke_width)    
        : type(type), ptr(shape_ptr.get()), stroke_width(stroke_width) {}

    Circle as_circle() const {
        return *(Circle*)ptr;
    }

    Ellipse as_ellipse() const {
        return *(Ellipse*)ptr;
    }

    Path as_path() const {
        return *(Path*)ptr;
    }

    Rect as_rect() const {
        return *(Rect*)ptr;
    }

    ShapeType type;
    void *ptr;
    float stroke_width;
};

struct ShapeGroup {
    ShapeGroup() {}
    ShapeGroup(ptr<int> shape_ids,
               int num_shapes,
               const ColorType &fill_color_type,
               ptr<void> fill_color,
               const ColorType &stroke_color_type,
               ptr<void> stroke_color,
               bool use_even_odd_rule,
               ptr<float> shape_to_canvas)
        : shape_ids(shape_ids.get()),
          num_shapes(num_shapes),
          fill_color_type(fill_color_type),
          fill_color(fill_color.get()),
          stroke_color_type(stroke_color_type),
          stroke_color(stroke_color.get()),
          use_even_odd_rule(use_even_odd_rule),
          shape_to_canvas(shape_to_canvas.get()) {
        canvas_to_shape = inverse(this->shape_to_canvas);
    }

    bool has_fill_color() const {
        return fill_color != nullptr;
    }

    Constant fill_color_as_constant() const {
        return *(Constant*)fill_color;
    }

    LinearGradient fill_color_as_linear_gradient() const {
        return *(LinearGradient*)fill_color;
    }

    RadialGradient fill_color_as_radial_gradient() const {
        return *(RadialGradient*)fill_color;
    }

    bool has_stroke_color() const {
        return stroke_color != nullptr;
    }

    Constant stroke_color_as_constant() const {
        return *(Constant*)stroke_color;
    }

    LinearGradient stroke_color_as_linear_gradient() const {
        return *(LinearGradient*)stroke_color;
    }

    RadialGradient stroke_color_as_radial_gradient() const {
        return *(RadialGradient*)stroke_color;
    }

    void copy_to(ptr<float> shape_to_canvas) const;

    int *shape_ids;
    int num_shapes;
    ColorType fill_color_type;
    void *fill_color;
    ColorType stroke_color_type;
    void *stroke_color;
    bool use_even_odd_rule;
    Matrix3x3f canvas_to_shape;
    Matrix3x3f shape_to_canvas;
};

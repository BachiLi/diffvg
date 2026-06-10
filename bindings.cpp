#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cassert>

#include "shape.h"
#include "color.h"
#include "filter.h"
#include "scene.h"
#include "ptr.h"
#include "vector.h"

void render(std::shared_ptr<Scene> scene,
            ptr<float> background_image,
            ptr<float> render_image,
            ptr<float> render_sdf,
            int width,
            int height,
            int num_samples_x,
            int num_samples_y,
            uint64_t seed,
            ptr<float> d_background_image,
            ptr<float> d_render_image,
            ptr<float> d_render_sdf,
            ptr<float> d_translation,
            bool use_prefiltering,
            ptr<float> eval_positions,
            int num_eval_positions);

namespace py = pybind11;

PYBIND11_MODULE(diffvg, m) {
    m.doc() = "Differential Vector Graphics";

    py::class_<ptr<void>>(m, "void_ptr")
        .def(py::init<std::size_t>())
        .def("as_size_t", &ptr<void>::as_size_t);
    py::class_<ptr<float>>(m, "float_ptr")
        .def(py::init<std::size_t>());
    py::class_<ptr<int>>(m, "int_ptr")
        .def(py::init<std::size_t>());

    py::class_<Vector2f>(m, "Vector2f")
        .def(py::init<float, float>())
        .def_readwrite("x", &Vector2f::x)
        .def_readwrite("y", &Vector2f::y);

    py::class_<Vector3f>(m, "Vector3f")
        .def(py::init<float, float, float>()) 
        .def_readwrite("x", &Vector3f::x)
        .def_readwrite("y", &Vector3f::y)
        .def_readwrite("z", &Vector3f::z);

    py::class_<Vector4f>(m, "Vector4f")
        .def(py::init<float, float, float, float>())
        .def_readwrite("x", &Vector4f::x)
        .def_readwrite("y", &Vector4f::y)
        .def_readwrite("z", &Vector4f::z)
        .def_readwrite("w", &Vector4f::w);

    py::enum_<ShapeType>(m, "ShapeType")
        .value("circle", ShapeType::Circle)
        .value("ellipse", ShapeType::Ellipse)
        .value("path", ShapeType::Path)
        .value("rect", ShapeType::Rect);

    py::class_<Circle>(m, "Circle")
        .def(py::init<float, Vector2f>())
        .def("get_ptr", &Circle::get_ptr)
        .def_readonly("radius", &Circle::radius)
        .def_readonly("center", &Circle::center);

    py::class_<Ellipse>(m, "Ellipse")
        .def(py::init<Vector2f, Vector2f>())
        .def("get_ptr", &Ellipse::get_ptr)
        .def_readonly("radius", &Ellipse::radius)
        .def_readonly("center", &Ellipse::center);

    py::class_<Path>(m, "Path")
        .def(py::init<ptr<int>, ptr<float>, ptr<float>, int, int, bool, bool>())
        .def("get_ptr", &Path::get_ptr)
        .def("has_thickness", &Path::has_thickness)
        .def("copy_to", &Path::copy_to)
        .def_readonly("num_points", &Path::num_points);

    py::class_<Rect>(m, "Rect")
        .def(py::init<Vector2f, Vector2f>())
        .def("get_ptr", &Rect::get_ptr)
        .def_readonly("p_min", &Rect::p_min)
        .def_readonly("p_max", &Rect::p_max);

    py::enum_<ColorType>(m, "ColorType")
        .value("constant", ColorType::Constant)
        .value("linear_gradient", ColorType::LinearGradient)
        .value("radial_gradient", ColorType::RadialGradient);

    py::class_<Constant>(m, "Constant")
        .def(py::init<Vector4f>())
        .def("get_ptr", &Constant::get_ptr)
        .def_readonly("color", &Constant::color);

    py::class_<LinearGradient>(m, "LinearGradient")
        .def(py::init<Vector2f, Vector2f, int, ptr<float>, ptr<float>>())
        .def("get_ptr", &LinearGradient::get_ptr)
        .def("copy_to", &LinearGradient::copy_to)
        .def_readonly("begin", &LinearGradient::begin)
        .def_readonly("end", &LinearGradient::end)
        .def_readonly("num_stops", &LinearGradient::num_stops);

    py::class_<RadialGradient>(m, "RadialGradient")
        .def(py::init<Vector2f, Vector2f, int, ptr<float>, ptr<float>>())
        .def("get_ptr", &RadialGradient::get_ptr)
        .def("copy_to", &RadialGradient::copy_to)
        .def_readonly("center", &RadialGradient::center)
        .def_readonly("radius", &RadialGradient::radius)
        .def_readonly("num_stops", &RadialGradient::num_stops);

    py::class_<Shape>(m, "Shape")
        .def(py::init<ShapeType, ptr<void>, float>())
        .def("as_circle", &Shape::as_circle)
        .def("as_ellipse", &Shape::as_ellipse)
        .def("as_path", &Shape::as_path)
        .def("as_rect", &Shape::as_rect)
        .def_readonly("type", &Shape::type)
        .def_readonly("stroke_width", &Shape::stroke_width);

    py::class_<ShapeGroup>(m, "ShapeGroup")
        .def(py::init<ptr<int>,
                      int,
                      ColorType,
                      ptr<void>,
                      ColorType,
                      ptr<void>,
                      bool,
                      ptr<float>>())
        .def("fill_color_as_constant", &ShapeGroup::fill_color_as_constant)
        .def("fill_color_as_linear_gradient", &ShapeGroup::fill_color_as_linear_gradient)
        .def("fill_color_as_radial_gradient", &ShapeGroup::fill_color_as_radial_gradient)
        .def("stroke_color_as_constant", &ShapeGroup::stroke_color_as_constant)
        .def("stroke_color_as_linear_gradient", &ShapeGroup::stroke_color_as_linear_gradient)
        .def("stroke_color_as_radial_gradient", &ShapeGroup::fill_color_as_radial_gradient)
        .def("has_fill_color", &ShapeGroup::has_fill_color)
        .def("has_stroke_color", &ShapeGroup::has_stroke_color)
        .def("copy_to", &ShapeGroup::copy_to)
        .def_readonly("fill_color_type", &ShapeGroup::fill_color_type)
        .def_readonly("stroke_color_type", &ShapeGroup::stroke_color_type);

    py::enum_<FilterType>(m, "FilterType")
        .value("box", FilterType::Box)
        .value("tent", FilterType::Tent)
        .value("parabolic", FilterType::RadialParabolic)
        .value("hann", FilterType::Hann);

    py::class_<Filter>(m, "Filter")
        .def(py::init<FilterType,
                      float>());

    py::class_<Scene, std::shared_ptr<Scene>>(m, "Scene")
        .def(py::init<int,
                      int,
                      const std::vector<const Shape*> &,
                      const std::vector<const ShapeGroup*> &,
                      const Filter &,
                      bool,
                      int>())
        .def("get_d_shape", &Scene::get_d_shape)
        .def("get_d_shape_group", &Scene::get_d_shape_group)
        .def("get_d_filter_radius", &Scene::get_d_filter_radius)
        .def_readonly("num_shapes", &Scene::num_shapes)
        .def_readonly("num_shape_groups", &Scene::num_shape_groups);

    m.def("render", &render, "");
}



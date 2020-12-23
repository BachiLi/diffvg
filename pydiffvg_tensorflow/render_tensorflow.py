import os
import tensorflow as tf
import diffvg
import pydiffvg_tensorflow as pydiffvg
import time
from enum import IntEnum
import warnings

print_timing = False
__EMPTY_TENSOR = tf.constant([])

def is_empty_tensor(tensor):
    return  tf.equal(tf.size(tensor), 0)

def set_print_timing(val):
    global print_timing
    print_timing=val

class OutputType(IntEnum):
    color = 1
    sdf = 2

class ShapeType:
    __shapetypes = [
        diffvg.ShapeType.circle,
        diffvg.ShapeType.ellipse,
        diffvg.ShapeType.path,
        diffvg.ShapeType.rect
    ]

    @staticmethod
    def asTensor(type):
        for i in range(len(ShapeType.__shapetypes)):
            if ShapeType.__shapetypes[i] == type:
                return tf.constant(i)

    @staticmethod
    def asShapeType(index: tf.Tensor):
        if is_empty_tensor(index):
            return None
        try:
            type = ShapeType.__shapetypes[index]
        except IndexError:
            print(f'{index} is out of range: [0, {len(ShapeType.__shapetypes)})')
            import sys
            sys.exit()
        else:
            return type

class ColorType:
    __colortypes = [
        diffvg.ColorType.constant,
        diffvg.ColorType.linear_gradient,
        diffvg.ColorType.radial_gradient
    ]

    @staticmethod
    def asTensor(type):
        for i in range(len(ColorType.__colortypes)):
            if ColorType.__colortypes[i] == type:
                return tf.constant(i)

    @staticmethod
    def asColorType(index: tf.Tensor):
        if is_empty_tensor(index):
            return None
        try:
            type = ColorType.__colortypes[index]
        except IndexError:
            print(f'{index} is out of range: [0, {len(ColorType.__colortypes)})')
            import sys
            sys.exit()
        else:
            return type

class FilterType:
    __filtertypes = [
        diffvg.FilterType.box,
        diffvg.FilterType.tent,
        diffvg.FilterType.hann
    ]

    @staticmethod
    def asTensor(type):
        for i in range(len(FilterType.__filtertypes)):
            if FilterType.__filtertypes[i] == type:
                return tf.constant(i)    

    @staticmethod
    def asFilterType(index: tf.Tensor):
        if is_empty_tensor(index):
            return None
        try:
            type = FilterType.__filtertypes[index]
        except IndexError:
            print(f'{index} is out of range: [0, {len(FilterType.__filtertypes)})')
            import sys
            sys.exit()
        else:
            return type

def serialize_scene(canvas_width,
                    canvas_height,
                    shapes,
                    shape_groups,
                    filter = pydiffvg.PixelFilter(type = diffvg.FilterType.box,
                                                  radius = tf.constant(0.5)),
                    output_type = OutputType.color,
                    use_prefiltering = False):
    """
        Given a list of shapes, convert them to a linear list of argument,
        so that we can use it in TF.
    """
    with tf.device('/device:cpu:' + str(pydiffvg.get_cpu_device_id())):
        num_shapes = len(shapes)
        num_shape_groups = len(shape_groups)
        args = []
        args.append(tf.constant(canvas_width))
        args.append(tf.constant(canvas_height))
        args.append(tf.constant(num_shapes))
        args.append(tf.constant(num_shape_groups))
        args.append(tf.constant(output_type))
        args.append(tf.constant(use_prefiltering))
        for shape in shapes:
            if isinstance(shape, pydiffvg.Circle):
                args.append(ShapeType.asTensor(diffvg.ShapeType.circle))
                args.append(tf.identity(shape.radius))
                args.append(tf.identity(shape.center))
            elif isinstance(shape, pydiffvg.Ellipse):
                args.append(ShapeType.asTensor(diffvg.ShapeType.ellipse))
                args.append(tf.identity(shape.radius))
                args.append(tf.identity(shape.center))
            elif isinstance(shape, pydiffvg.Path):
                assert(shape.points.shape[1] == 2)
                args.append(ShapeType.asTensor(diffvg.ShapeType.path))
                args.append(tf.identity(shape.num_control_points))
                args.append(tf.identity(shape.points))
                args.append(tf.constant(shape.is_closed))
                args.append(tf.constant(shape.use_distance_approx))
            elif isinstance(shape, pydiffvg.Polygon):
                assert(shape.points.shape[1] == 2)
                args.append(ShapeType.asTensor(diffvg.ShapeType.path))
                if shape.is_closed:
                    args.append(tf.zeros(shape.points.shape[0], dtype = tf.int32))
                else:
                    args.append(tf.zeros(shape.points.shape[0] - 1, dtype = tf.int32))
                args.append(tf.identity(shape.points))
                args.append(tf.constant(shape.is_closed))
            elif isinstance(shape, pydiffvg.Rect):
                args.append(ShapeType.asTensor(diffvg.ShapeType.rect))
                args.append(tf.identity(shape.p_min))
                args.append(tf.identity(shape.p_max))
            else:
                assert(False)
            args.append(tf.identity(shape.stroke_width))

        for shape_group in shape_groups:
            args.append(tf.identity(shape_group.shape_ids))
            # Fill color
            if shape_group.fill_color is None:
                args.append(__EMPTY_TENSOR)
            elif tf.is_tensor(shape_group.fill_color):
                args.append(ColorType.asTensor(diffvg.ColorType.constant))
                args.append(tf.identity(shape_group.fill_color))
            elif isinstance(shape_group.fill_color, pydiffvg.LinearGradient):
                args.append(ColorType.asTensor(diffvg.ColorType.linear_gradient))
                args.append(tf.identity(shape_group.fill_color.begin))
                args.append(tf.identity(shape_group.fill_color.end))
                args.append(tf.identity(shape_group.fill_color.offsets))
                args.append(tf.identity(shape_group.fill_color.stop_colors))
            elif isinstance(shape_group.fill_color, pydiffvg.RadialGradient):
                args.append(ColorType.asTensor(diffvg.ColorType.radial_gradient))
                args.append(tf.identity(shape_group.fill_color.center))
                args.append(tf.identity(shape_group.fill_color.radius))
                args.append(tf.identity(shape_group.fill_color.offsets))
                args.append(tf.identity(shape_group.fill_color.stop_colors))

            if shape_group.fill_color is not None:
                # go through the underlying shapes and check if they are all closed
                for shape_id in shape_group.shape_ids:
                    if isinstance(shapes[shape_id], pydiffvg.Path):
                        if not shapes[shape_id].is_closed:
                            warnings.warn("Detected non-closed paths with fill color. This might causes unexpected results.", Warning)

            # Stroke color
            if shape_group.stroke_color is None:
                args.append(__EMPTY_TENSOR)
            elif tf.is_tensor(shape_group.stroke_color):
                args.append(tf.constant(0))
                args.append(tf.identity(shape_group.stroke_color))
            elif isinstance(shape_group.stroke_color, pydiffvg.LinearGradient):
                args.append(ColorType.asTensor(diffvg.ColorType.linear_gradient))
                args.append(tf.identity(shape_group.stroke_color.begin))
                args.append(tf.identity(shape_group.stroke_color.end))
                args.append(tf.identity(shape_group.stroke_color.offsets))
                args.append(tf.identity(shape_group.stroke_color.stop_colors))
            elif isinstance(shape_group.stroke_color, pydiffvg.RadialGradient):
                args.append(ColorType.asTensor(diffvg.ColorType.radial_gradient))
                args.append(tf.identity(shape_group.stroke_color.center))
                args.append(tf.identity(shape_group.stroke_color.radius))
                args.append(tf.identity(shape_group.stroke_color.offsets))
                args.append(tf.identity(shape_group.stroke_color.stop_colors))
            args.append(tf.constant(shape_group.use_even_odd_rule))
            # Transformation
            args.append(tf.identity(shape_group.shape_to_canvas))
        args.append(FilterType.asTensor(filter.type))
        args.append(tf.constant(filter.radius))
    return args

class Context: pass

def forward(width,
            height,
            num_samples_x,
            num_samples_y,
            seed,
            *args):
    """
        Forward rendering pass: given a serialized scene and output an image.
    """
    # Unpack arguments
    with tf.device('/device:cpu:' + str(pydiffvg.get_cpu_device_id())):
        current_index = 0
        canvas_width = int(args[current_index])
        current_index += 1
        canvas_height = int(args[current_index])
        current_index += 1
        num_shapes = int(args[current_index])
        current_index += 1
        num_shape_groups = int(args[current_index])
        current_index += 1
        output_type = OutputType(int(args[current_index]))
        current_index += 1
        use_prefiltering = bool(args[current_index])
        current_index += 1
        shapes = []
        shape_groups = []
        shape_contents = [] # Important to avoid GC deleting the shapes
        color_contents = [] # Same as above
        for shape_id in range(num_shapes):
            shape_type = ShapeType.asShapeType(args[current_index])
            current_index += 1
            if shape_type == diffvg.ShapeType.circle:
                radius = args[current_index]
                current_index += 1
                center = args[current_index]
                current_index += 1
                shape = diffvg.Circle(float(radius),
                                      diffvg.Vector2f(float(center[0]), float(center[1])))
            elif shape_type == diffvg.ShapeType.ellipse:
                radius = args[current_index]
                current_index += 1
                center = args[current_index]
                current_index += 1
                shape = diffvg.Ellipse(diffvg.Vector2f(float(radius[0]), float(radius[1])),
                                       diffvg.Vector2f(float(center[0]), float(center[1])))
            elif shape_type == diffvg.ShapeType.path:
                num_control_points = args[current_index]
                current_index += 1
                points = args[current_index]
                current_index += 1
                is_closed = args[current_index]
                current_index += 1
                use_distance_approx = args[current_index]
                current_index += 1
                shape = diffvg.Path(diffvg.int_ptr(pydiffvg.data_ptr(num_control_points)),
                                    diffvg.float_ptr(pydiffvg.data_ptr(points)),
                                    diffvg.float_ptr(0), # thickness
                                    num_control_points.shape[0],
                                    points.shape[0],
                                    is_closed,
                                    use_distance_approx)
            elif shape_type == diffvg.ShapeType.rect:
                p_min = args[current_index]
                current_index += 1
                p_max = args[current_index]
                current_index += 1
                shape = diffvg.Rect(diffvg.Vector2f(float(p_min[0]), float(p_min[1])),
                                    diffvg.Vector2f(float(p_max[0]), float(p_max[1])))
            else:
                assert(False)
            stroke_width = args[current_index]
            current_index += 1
            shapes.append(diffvg.Shape(\
                shape_type, shape.get_ptr(), float(stroke_width)))
            shape_contents.append(shape)

        for shape_group_id in range(num_shape_groups):
            shape_ids = args[current_index]
            current_index += 1
            fill_color_type = ColorType.asColorType(args[current_index])
            current_index += 1
            if fill_color_type == diffvg.ColorType.constant:
                color = args[current_index]
                current_index += 1
                fill_color = diffvg.Constant(\
                    diffvg.Vector4f(color[0], color[1], color[2], color[3]))
            elif fill_color_type == diffvg.ColorType.linear_gradient:
                beg = args[current_index]
                current_index += 1
                end = args[current_index]
                current_index += 1
                offsets = args[current_index]
                current_index += 1
                stop_colors = args[current_index]
                current_index += 1
                assert(offsets.shape[0] == stop_colors.shape[0])
                fill_color = diffvg.LinearGradient(diffvg.Vector2f(float(beg[0]), float(beg[1])),
                                                   diffvg.Vector2f(float(end[0]), float(end[1])),
                                                   offsets.shape[0],
                                                   diffvg.float_ptr(pydiffvg.data_ptr(offsets)),
                                                   diffvg.float_ptr(pydiffvg.data_ptr(stop_colors)))
            elif fill_color_type == diffvg.ColorType.radial_gradient:
                center = args[current_index]
                current_index += 1
                radius = args[current_index]
                current_index += 1
                offsets = args[current_index]
                current_index += 1
                stop_colors = args[current_index]
                current_index += 1
                assert(offsets.shape[0] == stop_colors.shape[0])
                fill_color = diffvg.RadialGradient(diffvg.Vector2f(float(center[0]), float(center[1])),
                                                   diffvg.Vector2f(float(radius[0]), float(radius[1])),
                                                   offsets.shape[0],
                                                   diffvg.float_ptr(pydiffvg.data_ptr(offsets)),
                                                   diffvg.float_ptr(pydiffvg.data_ptr(stop_colors)))
            elif fill_color_type is None:
                fill_color = None
            else:
                assert(False)

            stroke_color_type = ColorType.asColorType(args[current_index])
            current_index += 1
            if stroke_color_type == diffvg.ColorType.constant:
                color = args[current_index]
                current_index += 1
                stroke_color = diffvg.Constant(\
                    diffvg.Vector4f(float(color[0]),
                                    float(color[1]),
                                    float(color[2]),
                                    float(color[3])))
            elif stroke_color_type == diffvg.ColorType.linear_gradient:
                beg = args[current_index]
                current_index += 1
                end = args[current_index]
                current_index += 1
                offsets = args[current_index]
                current_index += 1
                stop_colors = args[current_index]
                current_index += 1
                assert(offsets.shape[0] == stop_colors.shape[0])
                stroke_color = diffvg.LinearGradient(\
                    diffvg.Vector2f(float(beg[0]), float(beg[1])),
                    diffvg.Vector2f(float(end[0]), float(end[1])),
                    offsets.shape[0],
                    diffvg.float_ptr(pydiffvg.data_ptr(offsets)),
                    diffvg.float_ptr(stop_colors.data_ptr()))
            elif stroke_color_type == diffvg.ColorType.radial_gradient:
                center = args[current_index]
                current_index += 1
                radius = args[current_index]
                current_index += 1
                offsets = args[current_index]
                current_index += 1
                stop_colors = args[current_index]
                current_index += 1
                assert(offsets.shape[0] == stop_colors.shape[0])
                stroke_color = diffvg.RadialGradient(\
                    diffvg.Vector2f(float(center[0]), float(center[1])),
                    diffvg.Vector2f(float(radius[0]), float(radius[1])),
                    offsets.shape[0],
                    diffvg.float_ptr(pydiffvg.data_ptr(offsets)),
                    diffvg.float_ptr(pydiffvg.data_ptr(stop_colors)))
            elif stroke_color_type is None:
                stroke_color = None
            else:
                assert(False)
            use_even_odd_rule = bool(args[current_index])
            current_index += 1
            shape_to_canvas = args[current_index]
            current_index += 1

            if fill_color is not None:
                color_contents.append(fill_color)
            if stroke_color is not None:
                color_contents.append(stroke_color)
            shape_groups.append(diffvg.ShapeGroup(\
                diffvg.int_ptr(pydiffvg.data_ptr(shape_ids)),
                shape_ids.shape[0],
                diffvg.ColorType.constant if fill_color_type is None else fill_color_type,
                diffvg.void_ptr(0) if fill_color is None else fill_color.get_ptr(),
                diffvg.ColorType.constant if stroke_color_type is None else stroke_color_type,
                diffvg.void_ptr(0) if stroke_color is None else stroke_color.get_ptr(),
                use_even_odd_rule,
                diffvg.float_ptr(pydiffvg.data_ptr(shape_to_canvas))))

        filter_type = FilterType.asFilterType(args[current_index])
        current_index += 1
        filter_radius = args[current_index]
        current_index += 1
        filt = diffvg.Filter(filter_type, filter_radius)

    device_name = pydiffvg.get_device_name()
    device_spec = tf.DeviceSpec.from_string(device_name)
    use_gpu = device_spec.device_type == 'GPU'
    gpu_index = device_spec.device_index if device_spec.device_index is not None else 0

    start = time.time()
    scene = diffvg.Scene(canvas_width,
                         canvas_height,
                         shapes,
                         shape_groups,
                         filt,
                         use_gpu,
                         gpu_index)
    time_elapsed = time.time() - start
    global print_timing
    if print_timing:
        print('Scene construction, time: %.5f s' % time_elapsed)

    with tf.device(device_name):
        if output_type == OutputType.color:
            rendered_image = tf.zeros((int(height), int(width), 4), dtype = tf.float32)
        else:
            assert(output_type == OutputType.sdf)
            rendered_image = tf.zeros((int(height), int(width), 1), dtype = tf.float32)

        start = time.time()
        diffvg.render(scene,
                      diffvg.float_ptr(0), # background image
                      diffvg.float_ptr(pydiffvg.data_ptr(rendered_image) if output_type == OutputType.color else 0),
                      diffvg.float_ptr(pydiffvg.data_ptr(rendered_image) if output_type == OutputType.sdf else 0),
                      width,
                      height,
                      int(num_samples_x),
                      int(num_samples_y),
                      seed,
                      diffvg.float_ptr(0), # d_background_image
                      diffvg.float_ptr(0), # d_render_image
                      diffvg.float_ptr(0), # d_render_sdf
                      diffvg.float_ptr(0), # d_translation
                      use_prefiltering,
                      diffvg.float_ptr(0), # eval_positions
                      0 ) # num_eval_positions (automatically set to entire raster)
        time_elapsed = time.time() - start
        if print_timing:
            print('Forward pass, time: %.5f s' % time_elapsed)

    ctx = Context()
    ctx.scene = scene
    ctx.shape_contents = shape_contents
    ctx.color_contents = color_contents
    ctx.filter = filt
    ctx.width = width
    ctx.height = height
    ctx.num_samples_x = num_samples_x
    ctx.num_samples_y = num_samples_y
    ctx.seed = seed
    ctx.output_type = output_type
    ctx.use_prefiltering = use_prefiltering
    return rendered_image, ctx

@tf.custom_gradient
def render(*x):
    """
        The main TensorFlow interface of C++ diffvg.
    """
    assert(tf.executing_eagerly())
    if pydiffvg.get_use_gpu() and os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH') != 'true':
        print('******************** WARNING ********************')
        print('Tensorflow by default allocates all GPU memory,')
        print('causing huge amount of page faults when rendering.')
        print('Please set the environment variable TF_FORCE_GPU_ALLOW_GROWTH to true,')
        print('so that Tensorflow allocates memory on demand.')
        print('*************************************************')

    width = x[0]
    height = x[1]
    num_samples_x = x[2]
    num_samples_y = x[3]
    seed = x[4]
    args = x[5:]
    img, ctx = forward(width, height, num_samples_x, num_samples_y, seed, *args)

    def backward(grad_img):
        scene = ctx.scene
        width = ctx.width
        height = ctx.height
        num_samples_x = ctx.num_samples_x
        num_samples_y = ctx.num_samples_y
        seed = ctx.seed
        output_type = ctx.output_type
        use_prefiltering = ctx.use_prefiltering

        start = time.time()
        with tf.device(pydiffvg.get_device_name()):
            diffvg.render(scene,
                          diffvg.float_ptr(0), # background_image
                          diffvg.float_ptr(0), # render_image
                          diffvg.float_ptr(0), # render_sdf
                          width,
                          height,
                          num_samples_x,
                          num_samples_y,
                          seed,
                          diffvg.float_ptr(0), # d_background_image
                          diffvg.float_ptr(pydiffvg.data_ptr(grad_img) if output_type == OutputType.color else 0),
                          diffvg.float_ptr(pydiffvg.data_ptr(grad_img) if output_type == OutputType.sdf else 0),
                          diffvg.float_ptr(0), # d_translation
                          use_prefiltering,
                          diffvg.float_ptr(0), # eval_positions
                          0 ) # num_eval_positions (automatically set to entire raster))
        time_elapsed = time.time() - start
        global print_timing
        if print_timing:
            print('Backward pass, time: %.5f s' % time_elapsed)

        with tf.device('/device:cpu:' + str(pydiffvg.get_cpu_device_id())):
            d_args = []
            d_args.append(None) # width
            d_args.append(None) # height
            d_args.append(None) # num_samples_x
            d_args.append(None) # num_samples_y
            d_args.append(None) # seed
            d_args.append(None) # canvas_width
            d_args.append(None) # canvas_height
            d_args.append(None) # num_shapes
            d_args.append(None) # num_shape_groups
            d_args.append(None) # output_type
            d_args.append(None) # use_prefiltering
            for shape_id in range(scene.num_shapes):
                d_args.append(None) # type
                d_shape = scene.get_d_shape(shape_id)
                if d_shape.type == diffvg.ShapeType.circle:
                    d_circle = d_shape.as_circle()
                    radius = tf.constant(d_circle.radius)
                    d_args.append(radius)
                    c = d_circle.center
                    c = tf.constant((c.x, c.y))
                    d_args.append(c)
                elif d_shape.type == diffvg.ShapeType.ellipse:
                    d_ellipse = d_shape.as_ellipse()
                    r = d_ellipse.radius
                    r = tf.constant((d_ellipse.radius.x, d_ellipse.radius.y))
                    d_args.append(r)
                    c = d_ellipse.center
                    c = tf.constant((c.x, c.y))
                    d_args.append(c)
                elif d_shape.type == diffvg.ShapeType.path:
                    d_path = d_shape.as_path()
                    points = tf.zeros((d_path.num_points, 2), dtype=tf.float32)
                    d_path.copy_to(diffvg.float_ptr(pydiffvg.data_ptr(points)),diffvg.float_ptr(0))
                    d_args.append(None) # num_control_points
                    d_args.append(points)
                    d_args.append(None) # is_closed
                    d_args.append(None) # use_distance_approx
                elif d_shape.type == diffvg.ShapeType.rect:
                    d_rect = d_shape.as_rect()
                    p_min = tf.constant((d_rect.p_min.x, d_rect.p_min.y))
                    p_max = tf.constant((d_rect.p_max.x, d_rect.p_max.y))
                    d_args.append(p_min)
                    d_args.append(p_max)
                else:
                    assert(False)
                w = tf.constant((d_shape.stroke_width))
                d_args.append(w)

            for group_id in range(scene.num_shape_groups):
                d_shape_group = scene.get_d_shape_group(group_id)
                d_args.append(None) # shape_ids
                d_args.append(None) # fill_color_type
                if d_shape_group.has_fill_color():
                    if d_shape_group.fill_color_type == diffvg.ColorType.constant:
                        d_constant = d_shape_group.fill_color_as_constant()
                        c = d_constant.color
                        d_args.append(tf.constant((c.x, c.y, c.z, c.w)))
                    elif d_shape_group.fill_color_type == diffvg.ColorType.linear_gradient:
                        d_linear_gradient = d_shape_group.fill_color_as_linear_gradient()
                        beg = d_linear_gradient.begin
                        d_args.append(tf.constant((beg.x, beg.y)))
                        end = d_linear_gradient.end
                        d_args.append(tf.constant((end.x, end.y)))
                        offsets = tf.zeros((d_linear_gradient.num_stops), dtype=tf.float32)
                        stop_colors = tf.zeros((d_linear_gradient.num_stops, 4), dtype=tf.float32)
                        # HACK: tensorflow's eager mode uses a cache to store scalar
                        #       constants to avoid memory copy. If we pass scalar tensors
                        #       into the C++ code and modify them, we would corrupt the
                        #       cache, causing incorrect result in future scalar constant
                        #       creations. Thus we force tensorflow to copy by plusing a zero.
                        # (also see https://github.com/tensorflow/tensorflow/issues/11186
                        #  for more discussion regarding copying tensors)
                        if offsets.shape.num_elements() == 1:
                            offsets = offsets + 0
                        d_linear_gradient.copy_to(\
                            diffvg.float_ptr(pydiffvg.data_ptr(offsets)),
                            diffvg.float_ptr(pydiffvg.data_ptr(stop_colors)))
                        d_args.append(offsets)
                        d_args.append(stop_colors)
                    elif d_shape_group.fill_color_type == diffvg.ColorType.radial_gradient:
                        d_radial_gradient = d_shape_group.fill_color_as_radial_gradient()
                        center = d_radial_gradient.center
                        d_args.append(tf.constant((center.x, center.y)))
                        radius = d_radial_gradient.radius
                        d_args.append(tf.constant((radius.x, radius.y)))
                        offsets = tf.zeros((d_radial_gradient.num_stops))
                        if offsets.shape.num_elements() == 1:
                            offsets = offsets + 0
                        stop_colors = tf.zeros((d_radial_gradient.num_stops, 4))
                        d_radial_gradient.copy_to(\
                            diffvg.float_ptr(pydiffvg.data_ptr(offsets)),
                            diffvg.float_ptr(pydiffvg.data_ptr(stop_colors)))
                        d_args.append(offsets)
                        d_args.append(stop_colors)
                    else:
                        assert(False)
                d_args.append(None) # stroke_color_type
                if d_shape_group.has_stroke_color():
                    if d_shape_group.stroke_color_type == diffvg.ColorType.constant:
                        d_constant = d_shape_group.stroke_color_as_constant()
                        c = d_constant.color
                        d_args.append(tf.constant((c.x, c.y, c.z, c.w)))
                    elif d_shape_group.stroke_color_type == diffvg.ColorType.linear_gradient:
                        d_linear_gradient = d_shape_group.stroke_color_as_linear_gradient()
                        beg = d_linear_gradient.begin
                        d_args.append(tf.constant((beg.x, beg.y)))
                        end = d_linear_gradient.end
                        d_args.append(tf.constant((end.x, end.y)))
                        offsets = tf.zeros((d_linear_gradient.num_stops))
                        stop_colors = tf.zeros((d_linear_gradient.num_stops, 4))
                        if offsets.shape.num_elements() == 1:
                            offsets = offsets + 0
                        d_linear_gradient.copy_to(\
                            diffvg.float_ptr(pydiffvg.data_ptr(offsets)),
                            diffvg.float_ptr(pydiffvg.data_ptr(stop_colors)))
                        d_args.append(offsets)
                        d_args.append(stop_colors)
                    elif d_shape_group.fill_color_type == diffvg.ColorType.radial_gradient:
                        d_radial_gradient = d_shape_group.stroke_color_as_radial_gradient()
                        center = d_radial_gradient.center
                        d_args.append(tf.constant((center.x, center.y)))
                        radius = d_radial_gradient.radius
                        d_args.append(tf.constant((radius.x, radius.y)))
                        offsets = tf.zeros((d_radial_gradient.num_stops))
                        stop_colors = tf.zeros((d_radial_gradient.num_stops, 4))
                        if offsets.shape.num_elements() == 1:
                            offsets = offsets + 0
                        d_radial_gradient.copy_to(\
                            diffvg.float_ptr(pydiffvg.data_ptr(offsets)),
                            diffvg.float_ptr(pydiffvg.data_ptr(stop_colors)))
                        d_args.append(offsets)
                        d_args.append(stop_colors)
                    else:
                        assert(False)
                d_args.append(None) # use_even_odd_rule
                d_shape_to_canvas = tf.zeros((3, 3), dtype = tf.float32)
                d_shape_group.copy_to(diffvg.float_ptr(pydiffvg.data_ptr(d_shape_to_canvas)))
                d_args.append(d_shape_to_canvas)
            d_args.append(None) # filter_type
            d_args.append(tf.constant(scene.get_d_filter_radius()))

        return d_args

    return img, backward

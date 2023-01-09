import torch
import diffvg
import pydiffvg
import time
from enum import IntEnum
import warnings

print_timing = False

def set_print_timing(val):
    global print_timing
    print_timing=val

class OutputType(IntEnum):
    color = 1
    sdf = 2

class RenderFunction(torch.autograd.Function):
    """
        The PyTorch interface of diffvg.
    """
    @staticmethod
    def serialize_scene(canvas_width,
                        canvas_height,
                        shapes,
                        shape_groups,
                        filter = pydiffvg.PixelFilter(type = diffvg.FilterType.box,
                                                      radius = torch.tensor(0.5)),
                        output_type = OutputType.color,
                        use_prefiltering = False,
                        eval_positions = torch.tensor([])):
        """
            Given a list of shapes, convert them to a linear list of argument,
            so that we can use it in PyTorch.
        """
        num_shapes = len(shapes)
        num_shape_groups = len(shape_groups)
        args = []
        args.append(canvas_width)
        args.append(canvas_height)
        args.append(num_shapes)
        args.append(num_shape_groups)
        args.append(output_type)
        args.append(use_prefiltering)
        args.append(eval_positions.to(pydiffvg.get_device()))
        for shape in shapes:
            use_thickness = False
            if isinstance(shape, pydiffvg.Circle):
                assert(shape.center.is_contiguous())
                args.append(diffvg.ShapeType.circle)
                args.append(shape.radius.cpu())
                args.append(shape.center.cpu())
            elif isinstance(shape, pydiffvg.Ellipse):
                assert(shape.radius.is_contiguous())
                assert(shape.center.is_contiguous())
                args.append(diffvg.ShapeType.ellipse)
                args.append(shape.radius.cpu())
                args.append(shape.center.cpu())
            elif isinstance(shape, pydiffvg.Path):
                assert(shape.num_control_points.is_contiguous())
                assert(shape.points.is_contiguous())
                assert(shape.points.shape[1] == 2)
                assert(torch.isfinite(shape.points).all())
                args.append(diffvg.ShapeType.path)
                args.append(shape.num_control_points.to(torch.int32).cpu())
                args.append(shape.points.cpu())
                if len(shape.stroke_width.shape) > 0 and shape.stroke_width.shape[0] > 1:
                    assert(torch.isfinite(shape.stroke_width).all())
                    use_thickness = True
                    args.append(shape.stroke_width.cpu())
                else:
                    args.append(None)
                args.append(shape.is_closed)
                args.append(shape.use_distance_approx)
            elif isinstance(shape, pydiffvg.Polygon):
                assert(shape.points.is_contiguous())
                assert(shape.points.shape[1] == 2)
                args.append(diffvg.ShapeType.path)
                if shape.is_closed:
                    args.append(torch.zeros(shape.points.shape[0], dtype = torch.int32))
                else:
                    args.append(torch.zeros(shape.points.shape[0] - 1, dtype = torch.int32))
                args.append(shape.points.cpu())
                args.append(None)  
                args.append(shape.is_closed)
                args.append(False) # use_distance_approx
            elif isinstance(shape, pydiffvg.Rect):
                assert(shape.p_min.is_contiguous())
                assert(shape.p_max.is_contiguous())
                args.append(diffvg.ShapeType.rect)
                args.append(shape.p_min.cpu())
                args.append(shape.p_max.cpu())
            else:
                assert(False)
            if use_thickness:
                args.append(torch.tensor(0.0))
            else:
                args.append(shape.stroke_width.cpu())

        for shape_group in shape_groups:
            assert(shape_group.shape_ids.is_contiguous())
            args.append(shape_group.shape_ids.to(torch.int32).cpu())
            # Fill color
            if shape_group.fill_color is None:
                args.append(None)
            elif isinstance(shape_group.fill_color, torch.Tensor):
                assert(shape_group.fill_color.is_contiguous())
                args.append(diffvg.ColorType.constant)
                args.append(shape_group.fill_color.cpu())
            elif isinstance(shape_group.fill_color, pydiffvg.LinearGradient):
                assert(shape_group.fill_color.begin.is_contiguous())
                assert(shape_group.fill_color.end.is_contiguous())
                assert(shape_group.fill_color.offsets.is_contiguous())
                assert(shape_group.fill_color.stop_colors.is_contiguous())
                args.append(diffvg.ColorType.linear_gradient)
                args.append(shape_group.fill_color.begin.cpu())
                args.append(shape_group.fill_color.end.cpu())
                args.append(shape_group.fill_color.offsets.cpu())
                args.append(shape_group.fill_color.stop_colors.cpu())
            elif isinstance(shape_group.fill_color, pydiffvg.RadialGradient):
                assert(shape_group.fill_color.center.is_contiguous())
                assert(shape_group.fill_color.radius.is_contiguous())
                assert(shape_group.fill_color.offsets.is_contiguous())
                assert(shape_group.fill_color.stop_colors.is_contiguous())
                args.append(diffvg.ColorType.radial_gradient)
                args.append(shape_group.fill_color.center.cpu())
                args.append(shape_group.fill_color.radius.cpu())
                args.append(shape_group.fill_color.offsets.cpu())
                args.append(shape_group.fill_color.stop_colors.cpu())

            if shape_group.fill_color is not None:
                # go through the underlying shapes and check if they are all closed
                for shape_id in shape_group.shape_ids:
                    if isinstance(shapes[shape_id], pydiffvg.Path):
                        if not shapes[shape_id].is_closed:
                            warnings.warn("Detected non-closed paths with fill color. This might causes unexpected results.", Warning)

            # Stroke color
            if shape_group.stroke_color is None:
                args.append(None)
            elif isinstance(shape_group.stroke_color, torch.Tensor):
                assert(shape_group.stroke_color.is_contiguous())
                args.append(diffvg.ColorType.constant)
                args.append(shape_group.stroke_color.cpu())
            elif isinstance(shape_group.stroke_color, pydiffvg.LinearGradient):
                assert(shape_group.stroke_color.begin.is_contiguous())
                assert(shape_group.stroke_color.end.is_contiguous())
                assert(shape_group.stroke_color.offsets.is_contiguous())
                assert(shape_group.stroke_color.stop_colors.is_contiguous())
                assert(torch.isfinite(shape_group.stroke_color.stop_colors).all())
                args.append(diffvg.ColorType.linear_gradient)
                args.append(shape_group.stroke_color.begin.cpu())
                args.append(shape_group.stroke_color.end.cpu())
                args.append(shape_group.stroke_color.offsets.cpu())
                args.append(shape_group.stroke_color.stop_colors.cpu())
            elif isinstance(shape_group.stroke_color, pydiffvg.RadialGradient):
                assert(shape_group.stroke_color.center.is_contiguous())
                assert(shape_group.stroke_color.radius.is_contiguous())
                assert(shape_group.stroke_color.offsets.is_contiguous())
                assert(shape_group.stroke_color.stop_colors.is_contiguous())
                assert(torch.isfinite(shape_group.stroke_color.stop_colors).all())
                args.append(diffvg.ColorType.radial_gradient)
                args.append(shape_group.stroke_color.center.cpu())
                args.append(shape_group.stroke_color.radius.cpu())
                args.append(shape_group.stroke_color.offsets.cpu())
                args.append(shape_group.stroke_color.stop_colors.cpu())
            args.append(shape_group.use_even_odd_rule)
            # Transformation
            args.append(shape_group.shape_to_canvas.contiguous().cpu())
        args.append(filter.type)
        args.append(filter.radius.cpu())
        return args

    @staticmethod
    def forward(ctx,
                width,
                height,
                num_samples_x,
                num_samples_y,
                seed,
                background_image,
                *args):
        """
            Forward rendering pass.
        """
        # Unpack arguments
        current_index = 0
        canvas_width = args[current_index]
        current_index += 1
        canvas_height = args[current_index]
        current_index += 1
        num_shapes = args[current_index]
        current_index += 1
        num_shape_groups = args[current_index]
        current_index += 1
        output_type = args[current_index]
        current_index += 1
        use_prefiltering = args[current_index]
        current_index += 1
        eval_positions = args[current_index]
        current_index += 1
        shapes = []
        shape_groups = []
        shape_contents = [] # Important to avoid GC deleting the shapes
        color_contents = [] # Same as above
        for shape_id in range(num_shapes):
            shape_type = args[current_index]
            current_index += 1
            if shape_type == diffvg.ShapeType.circle:
                radius = args[current_index]
                current_index += 1
                center = args[current_index]
                current_index += 1
                shape = diffvg.Circle(radius, diffvg.Vector2f(center[0], center[1]))
            elif shape_type == diffvg.ShapeType.ellipse:
                radius = args[current_index]
                current_index += 1
                center = args[current_index]
                current_index += 1
                shape = diffvg.Ellipse(diffvg.Vector2f(radius[0], radius[1]),
                                       diffvg.Vector2f(center[0], center[1]))
            elif shape_type == diffvg.ShapeType.path:
                num_control_points = args[current_index]
                current_index += 1
                points = args[current_index]
                current_index += 1
                thickness = args[current_index]
                current_index += 1
                is_closed = args[current_index]
                current_index += 1
                use_distance_approx = args[current_index]
                current_index += 1
                shape = diffvg.Path(diffvg.int_ptr(num_control_points.data_ptr()),
                                    diffvg.float_ptr(points.data_ptr()),
                                    diffvg.float_ptr(thickness.data_ptr() if thickness is not None else 0),
                                    num_control_points.shape[0],
                                    points.shape[0],
                                    is_closed,
                                    use_distance_approx)
            elif shape_type == diffvg.ShapeType.rect:
                p_min = args[current_index]
                current_index += 1
                p_max = args[current_index]
                current_index += 1
                shape = diffvg.Rect(diffvg.Vector2f(p_min[0], p_min[1]),
                                    diffvg.Vector2f(p_max[0], p_max[1]))
            else:
                assert(False)
            stroke_width = args[current_index]
            current_index += 1
            shapes.append(diffvg.Shape(\
                shape_type, shape.get_ptr(), stroke_width.item()))
            shape_contents.append(shape)

        for shape_group_id in range(num_shape_groups):
            shape_ids = args[current_index]
            current_index += 1
            fill_color_type = args[current_index]
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
                fill_color = diffvg.LinearGradient(diffvg.Vector2f(beg[0], beg[1]),
                                                   diffvg.Vector2f(end[0], end[1]),
                                                   offsets.shape[0],
                                                   diffvg.float_ptr(offsets.data_ptr()),
                                                   diffvg.float_ptr(stop_colors.data_ptr()))
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
                fill_color = diffvg.RadialGradient(diffvg.Vector2f(center[0], center[1]),
                                                   diffvg.Vector2f(radius[0], radius[1]),
                                                   offsets.shape[0],
                                                   diffvg.float_ptr(offsets.data_ptr()),
                                                   diffvg.float_ptr(stop_colors.data_ptr()))
            elif fill_color_type is None:
                fill_color = None
            else:
                assert(False)
            stroke_color_type = args[current_index]
            current_index += 1
            if stroke_color_type == diffvg.ColorType.constant:
                color = args[current_index]
                current_index += 1
                stroke_color = diffvg.Constant(\
                    diffvg.Vector4f(color[0], color[1], color[2], color[3]))
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
                stroke_color = diffvg.LinearGradient(diffvg.Vector2f(beg[0], beg[1]),
                                                     diffvg.Vector2f(end[0], end[1]),
                                                     offsets.shape[0],
                                                     diffvg.float_ptr(offsets.data_ptr()),
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
                stroke_color = diffvg.RadialGradient(diffvg.Vector2f(center[0], center[1]),
                                                     diffvg.Vector2f(radius[0], radius[1]),
                                                     offsets.shape[0],
                                                     diffvg.float_ptr(offsets.data_ptr()),
                                                     diffvg.float_ptr(stop_colors.data_ptr()))
            elif stroke_color_type is None:
                stroke_color = None
            else:
                assert(False)
            use_even_odd_rule = args[current_index]
            current_index += 1
            shape_to_canvas = args[current_index]
            current_index += 1

            if fill_color is not None:
                color_contents.append(fill_color)
            if stroke_color is not None:
                color_contents.append(stroke_color)
            shape_groups.append(diffvg.ShapeGroup(\
                diffvg.int_ptr(shape_ids.data_ptr()),
                shape_ids.shape[0],
                diffvg.ColorType.constant if fill_color_type is None else fill_color_type,
                diffvg.void_ptr(0) if fill_color is None else fill_color.get_ptr(),
                diffvg.ColorType.constant if stroke_color_type is None else stroke_color_type,
                diffvg.void_ptr(0) if stroke_color is None else stroke_color.get_ptr(),
                use_even_odd_rule,
                diffvg.float_ptr(shape_to_canvas.data_ptr())))

        filter_type = args[current_index]
        current_index += 1
        filter_radius = args[current_index]
        current_index += 1
        filt = diffvg.Filter(filter_type, filter_radius)

        start = time.time()
        scene = diffvg.Scene(canvas_width, canvas_height,
            shapes, shape_groups, filt, pydiffvg.get_use_gpu(),
            pydiffvg.get_device().index if pydiffvg.get_device().index is not None else -1)
        time_elapsed = time.time() - start
        global print_timing
        if print_timing:
            print('Scene construction, time: %.5f s' % time_elapsed)

        if output_type == OutputType.color:
            assert(eval_positions.shape[0] == 0)
            rendered_image = torch.zeros(height, width, 4, device = pydiffvg.get_device())
        else:
            assert(output_type == OutputType.sdf)          
            if eval_positions.shape[0] == 0:
                rendered_image = torch.zeros(height, width, 1, device = pydiffvg.get_device())
            else:
                rendered_image = torch.zeros(eval_positions.shape[0], 1, device = pydiffvg.get_device())

        if background_image is not None:
            background_image = background_image.to(pydiffvg.get_device())
            if background_image.shape[2] == 3:
                raise NotImplementedError('Background image must have 4 channels, not 3. Add a fourth channel with all ones via torch.ones().')
            background_image = background_image.contiguous()
            assert(background_image.shape[0] == rendered_image.shape[0])
            assert(background_image.shape[1] == rendered_image.shape[1])
            assert(background_image.shape[2] == 4)

        start = time.time()
        diffvg.render(scene,
                      diffvg.float_ptr(background_image.data_ptr() if background_image is not None else 0),
                      diffvg.float_ptr(rendered_image.data_ptr() if output_type == OutputType.color else 0),
                      diffvg.float_ptr(rendered_image.data_ptr() if output_type == OutputType.sdf else 0),
                      width,
                      height,
                      num_samples_x,
                      num_samples_y,
                      seed,
                      diffvg.float_ptr(0), # d_background_image
                      diffvg.float_ptr(0), # d_render_image
                      diffvg.float_ptr(0), # d_render_sdf
                      diffvg.float_ptr(0), # d_translation
                      use_prefiltering,
                      diffvg.float_ptr(eval_positions.data_ptr()),
                      eval_positions.shape[0])
        assert(torch.isfinite(rendered_image).all())
        time_elapsed = time.time() - start
        if print_timing:
            print('Forward pass, time: %.5f s' % time_elapsed)

        ctx.scene = scene
        ctx.background_image = background_image
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
        ctx.eval_positions = eval_positions
        return rendered_image

    @staticmethod
    def render_grad(grad_img,
                    width,
                    height,
                    num_samples_x,
                    num_samples_y,
                    seed,
                    background_image,
                    *args):
        if not grad_img.is_contiguous():
            grad_img = grad_img.contiguous()
        assert(torch.isfinite(grad_img).all())

        # Unpack arguments
        current_index = 0
        canvas_width = args[current_index]
        current_index += 1
        canvas_height = args[current_index]
        current_index += 1
        num_shapes = args[current_index]
        current_index += 1
        num_shape_groups = args[current_index]
        current_index += 1
        output_type = args[current_index]
        current_index += 1
        use_prefiltering = args[current_index]
        current_index += 1
        eval_positions = args[current_index]
        current_index += 1        
        shapes = []
        shape_groups = []
        shape_contents = [] # Important to avoid GC deleting the shapes
        color_contents = [] # Same as above
        for shape_id in range(num_shapes):
            shape_type = args[current_index]
            current_index += 1
            if shape_type == diffvg.ShapeType.circle:
                radius = args[current_index]
                current_index += 1
                center = args[current_index]
                current_index += 1
                shape = diffvg.Circle(radius, diffvg.Vector2f(center[0], center[1]))
            elif shape_type == diffvg.ShapeType.ellipse:
                radius = args[current_index]
                current_index += 1
                center = args[current_index]
                current_index += 1
                shape = diffvg.Ellipse(diffvg.Vector2f(radius[0], radius[1]),
                                       diffvg.Vector2f(center[0], center[1]))
            elif shape_type == diffvg.ShapeType.path:
                num_control_points = args[current_index]
                current_index += 1
                points = args[current_index]
                current_index += 1
                thickness = args[current_index]
                current_index += 1
                is_closed = args[current_index]
                current_index += 1
                use_distance_approx = args[current_index]
                current_index += 1
                shape = diffvg.Path(diffvg.int_ptr(num_control_points.data_ptr()),
                                    diffvg.float_ptr(points.data_ptr()),
                                    diffvg.float_ptr(thickness.data_ptr() if thickness is not None else 0),
                                    num_control_points.shape[0],
                                    points.shape[0],
                                    is_closed,
                                    use_distance_approx)
            elif shape_type == diffvg.ShapeType.rect:
                p_min = args[current_index]
                current_index += 1
                p_max = args[current_index]
                current_index += 1
                shape = diffvg.Rect(diffvg.Vector2f(p_min[0], p_min[1]),
                                    diffvg.Vector2f(p_max[0], p_max[1]))
            else:
                assert(False)
            stroke_width = args[current_index]
            current_index += 1
            shapes.append(diffvg.Shape(\
                shape_type, shape.get_ptr(), stroke_width.item()))
            shape_contents.append(shape)

        for shape_group_id in range(num_shape_groups):
            shape_ids = args[current_index]
            current_index += 1
            fill_color_type = args[current_index]
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
                fill_color = diffvg.LinearGradient(diffvg.Vector2f(beg[0], beg[1]),
                                                   diffvg.Vector2f(end[0], end[1]),
                                                   offsets.shape[0],
                                                   diffvg.float_ptr(offsets.data_ptr()),
                                                   diffvg.float_ptr(stop_colors.data_ptr()))
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
                fill_color = diffvg.RadialGradient(diffvg.Vector2f(center[0], center[1]),
                                                   diffvg.Vector2f(radius[0], radius[1]),
                                                   offsets.shape[0],
                                                   diffvg.float_ptr(offsets.data_ptr()),
                                                   diffvg.float_ptr(stop_colors.data_ptr()))
            elif fill_color_type is None:
                fill_color = None
            else:
                assert(False)
            stroke_color_type = args[current_index]
            current_index += 1
            if stroke_color_type == diffvg.ColorType.constant:
                color = args[current_index]
                current_index += 1
                stroke_color = diffvg.Constant(\
                    diffvg.Vector4f(color[0], color[1], color[2], color[3]))
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
                stroke_color = diffvg.LinearGradient(diffvg.Vector2f(beg[0], beg[1]),
                                                     diffvg.Vector2f(end[0], end[1]),
                                                     offsets.shape[0],
                                                     diffvg.float_ptr(offsets.data_ptr()),
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
                stroke_color = diffvg.RadialGradient(diffvg.Vector2f(center[0], center[1]),
                                                     diffvg.Vector2f(radius[0], radius[1]),
                                                     offsets.shape[0],
                                                     diffvg.float_ptr(offsets.data_ptr()),
                                                     diffvg.float_ptr(stop_colors.data_ptr()))
            elif stroke_color_type is None:
                stroke_color = None
            else:
                assert(False)
            use_even_odd_rule = args[current_index]
            current_index += 1
            shape_to_canvas = args[current_index]
            current_index += 1

            if fill_color is not None:
                color_contents.append(fill_color)
            if stroke_color is not None:
                color_contents.append(stroke_color)
            shape_groups.append(diffvg.ShapeGroup(\
                diffvg.int_ptr(shape_ids.data_ptr()),
                shape_ids.shape[0],
                diffvg.ColorType.constant if fill_color_type is None else fill_color_type,
                diffvg.void_ptr(0) if fill_color is None else fill_color.get_ptr(),
                diffvg.ColorType.constant if stroke_color_type is None else stroke_color_type,
                diffvg.void_ptr(0) if stroke_color is None else stroke_color.get_ptr(),
                use_even_odd_rule,
                diffvg.float_ptr(shape_to_canvas.data_ptr())))

        filter_type = args[current_index]
        current_index += 1
        filter_radius = args[current_index]
        current_index += 1
        filt = diffvg.Filter(filter_type, filter_radius)

        scene = diffvg.Scene(canvas_width, canvas_height,
            shapes, shape_groups, filt, pydiffvg.get_use_gpu(),
            pydiffvg.get_device().index if pydiffvg.get_device().index is not None else -1)

        if output_type == OutputType.color:
            assert(grad_img.shape[2] == 4)
        else:
            assert(grad_img.shape[2] == 1)

        if background_image is not None:
            background_image = background_image.to(pydiffvg.get_device())
            if background_image.shape[2] == 3:
                background_image = torch.cat((\
                    background_image, torch.ones(background_image.shape[0], background_image.shape[1], 1,
                        device = background_image.device)), dim = 2)
            background_image = background_image.contiguous()
            assert(background_image.shape[0] == rendered_image.shape[0])
            assert(background_image.shape[1] == rendered_image.shape[1])
            assert(background_image.shape[2] == 4)

        translation_grad_image = \
            torch.zeros(height, width, 2, device = pydiffvg.get_device())
        start = time.time()
        diffvg.render(scene,
                      diffvg.float_ptr(background_image.data_ptr() if background_image is not None else 0),
                      diffvg.float_ptr(0), # render_image
                      diffvg.float_ptr(0), # render_sdf
                      width,
                      height,
                      num_samples_x,
                      num_samples_y,
                      seed,
                      diffvg.float_ptr(0), # d_background_image
                      diffvg.float_ptr(grad_img.data_ptr() if output_type == OutputType.color else 0),
                      diffvg.float_ptr(grad_img.data_ptr() if output_type == OutputType.sdf else 0),
                      diffvg.float_ptr(translation_grad_image.data_ptr()),
                      use_prefiltering,
                      diffvg.float_ptr(eval_positions.data_ptr()),
                      eval_positions.shape[0])
        time_elapsed = time.time() - start
        if print_timing:
            print('Gradient pass, time: %.5f s' % time_elapsed)
        assert(torch.isfinite(translation_grad_image).all())

        return translation_grad_image

    @staticmethod
    def backward(ctx,
                 grad_img):
        if not grad_img.is_contiguous():
            grad_img = grad_img.contiguous()
        assert(torch.isfinite(grad_img).all())

        scene = ctx.scene
        width = ctx.width
        height = ctx.height
        num_samples_x = ctx.num_samples_x
        num_samples_y = ctx.num_samples_y
        seed = ctx.seed
        output_type = ctx.output_type
        use_prefiltering = ctx.use_prefiltering
        eval_positions = ctx.eval_positions
        background_image = ctx.background_image

        if background_image is not None:
            d_background_image = torch.zeros_like(background_image)
        else:
            d_background_image = None

        start = time.time()
        diffvg.render(scene,
                      diffvg.float_ptr(background_image.data_ptr() if background_image is not None else 0),
                      diffvg.float_ptr(0), # render_image
                      diffvg.float_ptr(0), # render_sdf
                      width,
                      height,
                      num_samples_x,
                      num_samples_y,
                      seed,
                      diffvg.float_ptr(d_background_image.data_ptr() if background_image is not None else 0),
                      diffvg.float_ptr(grad_img.data_ptr() if output_type == OutputType.color else 0),
                      diffvg.float_ptr(grad_img.data_ptr() if output_type == OutputType.sdf else 0),
                      diffvg.float_ptr(0), # d_translation
                      use_prefiltering,
                      diffvg.float_ptr(eval_positions.data_ptr()),
                      eval_positions.shape[0])
        time_elapsed = time.time() - start
        global print_timing
        if print_timing:
            print('Backward pass, time: %.5f s' % time_elapsed)

        d_args = []
        d_args.append(None) # width
        d_args.append(None) # height
        d_args.append(None) # num_samples_x
        d_args.append(None) # num_samples_y
        d_args.append(None) # seed
        d_args.append(d_background_image)
        d_args.append(None) # canvas_width
        d_args.append(None) # canvas_height
        d_args.append(None) # num_shapes
        d_args.append(None) # num_shape_groups
        d_args.append(None) # output_type
        d_args.append(None) # use_prefiltering
        d_args.append(None) # eval_positions
        for shape_id in range(scene.num_shapes):
            d_args.append(None) # type
            d_shape = scene.get_d_shape(shape_id)
            use_thickness = False
            if d_shape.type == diffvg.ShapeType.circle:
                d_circle = d_shape.as_circle()
                radius = torch.tensor(d_circle.radius)
                assert(torch.isfinite(radius).all())
                d_args.append(radius)
                c = d_circle.center
                c = torch.tensor((c.x, c.y))
                assert(torch.isfinite(c).all())
                d_args.append(c)
            elif d_shape.type == diffvg.ShapeType.ellipse:
                d_ellipse = d_shape.as_ellipse()
                r = d_ellipse.radius
                r = torch.tensor((d_ellipse.radius.x, d_ellipse.radius.y))
                assert(torch.isfinite(r).all())
                d_args.append(r)
                c = d_ellipse.center
                c = torch.tensor((c.x, c.y))
                assert(torch.isfinite(c).all())
                d_args.append(c)
            elif d_shape.type == diffvg.ShapeType.path:
                d_path = d_shape.as_path()
                points = torch.zeros((d_path.num_points, 2))
                thickness = None
                if d_path.has_thickness():
                    use_thickness = True
                    thickness = torch.zeros(d_path.num_points)
                    d_path.copy_to(diffvg.float_ptr(points.data_ptr()), diffvg.float_ptr(thickness.data_ptr()))
                else:
                    d_path.copy_to(diffvg.float_ptr(points.data_ptr()), diffvg.float_ptr(0))
                assert(torch.isfinite(points).all())
                if thickness is not None:
                    assert(torch.isfinite(thickness).all())
                d_args.append(None) # num_control_points
                d_args.append(points)
                d_args.append(thickness)
                d_args.append(None) # is_closed
                d_args.append(None) # use_distance_approx
            elif d_shape.type == diffvg.ShapeType.rect:
                d_rect = d_shape.as_rect()
                p_min = torch.tensor((d_rect.p_min.x, d_rect.p_min.y))
                p_max = torch.tensor((d_rect.p_max.x, d_rect.p_max.y))
                assert(torch.isfinite(p_min).all())
                assert(torch.isfinite(p_max).all())
                d_args.append(p_min)
                d_args.append(p_max)
            else:
                assert(False)
            if use_thickness:
                d_args.append(None)
            else:
                w = torch.tensor((d_shape.stroke_width))
                assert(torch.isfinite(w).all())
                d_args.append(w)

        for group_id in range(scene.num_shape_groups):
            d_shape_group = scene.get_d_shape_group(group_id)
            d_args.append(None) # shape_ids
            d_args.append(None) # fill_color_type
            if d_shape_group.has_fill_color():
                if d_shape_group.fill_color_type == diffvg.ColorType.constant:
                    d_constant = d_shape_group.fill_color_as_constant()
                    c = d_constant.color
                    d_args.append(torch.tensor((c.x, c.y, c.z, c.w)))
                elif d_shape_group.fill_color_type == diffvg.ColorType.linear_gradient:
                    d_linear_gradient = d_shape_group.fill_color_as_linear_gradient()
                    beg = d_linear_gradient.begin
                    d_args.append(torch.tensor((beg.x, beg.y)))
                    end = d_linear_gradient.end
                    d_args.append(torch.tensor((end.x, end.y)))
                    offsets = torch.zeros((d_linear_gradient.num_stops))
                    stop_colors = torch.zeros((d_linear_gradient.num_stops, 4))
                    d_linear_gradient.copy_to(\
                        diffvg.float_ptr(offsets.data_ptr()),
                        diffvg.float_ptr(stop_colors.data_ptr()))
                    assert(torch.isfinite(stop_colors).all())
                    d_args.append(offsets)
                    d_args.append(stop_colors)
                elif d_shape_group.fill_color_type == diffvg.ColorType.radial_gradient:
                    d_radial_gradient = d_shape_group.fill_color_as_radial_gradient()
                    center = d_radial_gradient.center
                    d_args.append(torch.tensor((center.x, center.y)))
                    radius = d_radial_gradient.radius
                    d_args.append(torch.tensor((radius.x, radius.y)))
                    offsets = torch.zeros((d_radial_gradient.num_stops))
                    stop_colors = torch.zeros((d_radial_gradient.num_stops, 4))
                    d_radial_gradient.copy_to(\
                        diffvg.float_ptr(offsets.data_ptr()),
                        diffvg.float_ptr(stop_colors.data_ptr()))
                    assert(torch.isfinite(stop_colors).all())
                    d_args.append(offsets)
                    d_args.append(stop_colors)
                else:
                    assert(False)
            d_args.append(None) # stroke_color_type
            if d_shape_group.has_stroke_color():
                if d_shape_group.stroke_color_type == diffvg.ColorType.constant:
                    d_constant = d_shape_group.stroke_color_as_constant()
                    c = d_constant.color
                    d_args.append(torch.tensor((c.x, c.y, c.z, c.w)))
                elif d_shape_group.stroke_color_type == diffvg.ColorType.linear_gradient:
                    d_linear_gradient = d_shape_group.stroke_color_as_linear_gradient()
                    beg = d_linear_gradient.begin
                    d_args.append(torch.tensor((beg.x, beg.y)))
                    end = d_linear_gradient.end
                    d_args.append(torch.tensor((end.x, end.y)))
                    offsets = torch.zeros((d_linear_gradient.num_stops))
                    stop_colors = torch.zeros((d_linear_gradient.num_stops, 4))
                    d_linear_gradient.copy_to(\
                        diffvg.float_ptr(offsets.data_ptr()),
                        diffvg.float_ptr(stop_colors.data_ptr()))
                    assert(torch.isfinite(stop_colors).all())
                    d_args.append(offsets)
                    d_args.append(stop_colors)
                elif d_shape_group.fill_color_type == diffvg.ColorType.radial_gradient:
                    d_radial_gradient = d_shape_group.stroke_color_as_radial_gradient()
                    center = d_radial_gradient.center
                    d_args.append(torch.tensor((center.x, center.y)))
                    radius = d_radial_gradient.radius
                    d_args.append(torch.tensor((radius.x, radius.y)))
                    offsets = torch.zeros((d_radial_gradient.num_stops))
                    stop_colors = torch.zeros((d_radial_gradient.num_stops, 4))
                    d_radial_gradient.copy_to(\
                        diffvg.float_ptr(offsets.data_ptr()),
                        diffvg.float_ptr(stop_colors.data_ptr()))
                    assert(torch.isfinite(stop_colors).all())
                    d_args.append(offsets)
                    d_args.append(stop_colors)
                else:
                    assert(False)
            d_args.append(None) # use_even_odd_rule
            d_shape_to_canvas = torch.zeros((3, 3))
            d_shape_group.copy_to(diffvg.float_ptr(d_shape_to_canvas.data_ptr()))
            assert(torch.isfinite(d_shape_to_canvas).all())
            d_args.append(d_shape_to_canvas)
        d_args.append(None) # filter_type
        d_args.append(torch.tensor(scene.get_d_filter_radius()))

        return tuple(d_args)

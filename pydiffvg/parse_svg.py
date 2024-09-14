import torch
import xml.etree.ElementTree as etree
import numpy as np
import diffvg
import os
import pydiffvg
import svgpathtools
import svgpathtools.parser
import re
import warnings
import cssutils
import logging
import matplotlib.colors 
cssutils.log.setLevel(logging.ERROR)

def remove_namespaces(s):
    """
        {...} ... -> ...
    """
    return re.sub('{.*}', '', s)

def parse_style(s, defs):
    style_dict = {}
    for e in s.split(';'):
        key_value = e.split(':')
        if len(key_value) == 2:
            key = key_value[0].strip()
            value = key_value[1].strip()
            if key == 'fill' or key == 'stroke':
                # Special case: convert colors into tensor in definitions so
                # that different shapes can share the same color
                value = parse_color(value, defs)
            style_dict[key] = value
    return style_dict

def parse_hex(s):
    """
        Hex to tuple
    """
    s = s.lstrip('#')
    if len(s) == 3:
        s = s[0] + s[0] + s[1] + s[1] + s[2] + s[2]
    rgb = tuple(int(s[i:i+2], 16) for i in (0, 2, 4))
    # sRGB to RGB
    # return torch.pow(torch.tensor([rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0]), 2.2)
    return torch.pow(torch.tensor([rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0]), 1.0)

def parse_int(s):
    """
        trim alphabets
    """
    return int(float(''.join(i for i in s if (not i.isalpha()))))

def parse_color(s, defs):
    if s is None:
        return None
    if isinstance(s, torch.Tensor):
        return s
    s = s.lstrip(' ')
    color = torch.tensor([0.0, 0.0, 0.0, 1.0])
    if s[0] == '#':
        color[:3] = parse_hex(s)
    elif s[:3] == 'url':
        # url(#id)
        color = defs[s[4:-1].lstrip('#')]
    elif s == 'none':
        color = None
    elif s[:4] == 'rgb(':
        rgb = s[4:-1].split(',')
        color = torch.tensor([int(rgb[0]) / 255.0, int(rgb[1]) / 255.0, int(rgb[2]) / 255.0, 1.0])
    elif s == 'none':
        return None
    else:
        try : 
            rgba = matplotlib.colors.to_rgba(s)
            color = torch.tensor(rgba)
        except ValueError : 
            warnings.warn('Unknown color command ' + s)
    return color

# https://github.com/mathandy/svgpathtools/blob/7ebc56a831357379ff22216bec07e2c12e8c5bc6/svgpathtools/parser.py
def _parse_transform_substr(transform_substr):
    type_str, value_str = transform_substr.split('(')
    value_str = value_str.replace(',', ' ')
    values = list(map(float, filter(None, value_str.split(' '))))

    transform = np.identity(3)
    if 'matrix' in type_str:
        transform[0:2, 0:3] = np.array([values[0:6:2], values[1:6:2]])
    elif 'translate' in transform_substr:
        transform[0, 2] = values[0]
        if len(values) > 1:
            transform[1, 2] = values[1]
    elif 'scale' in transform_substr:
        x_scale = values[0]
        y_scale = values[1] if (len(values) > 1) else x_scale
        transform[0, 0] = x_scale
        transform[1, 1] = y_scale
    elif 'rotate' in transform_substr:
        angle = values[0] * np.pi / 180.0
        if len(values) == 3:
            offset = values[1:3]
        else:
            offset = (0, 0)
        tf_offset = np.identity(3)
        tf_offset[0:2, 2:3] = np.array([[offset[0]], [offset[1]]])
        tf_rotate = np.identity(3)
        tf_rotate[0:2, 0:2] = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        tf_offset_neg = np.identity(3)
        tf_offset_neg[0:2, 2:3] = np.array([[-offset[0]], [-offset[1]]])

        transform = tf_offset.dot(tf_rotate).dot(tf_offset_neg)
    elif 'skewX' in transform_substr:
        transform[0, 1] = np.tan(values[0] * np.pi / 180.0)
    elif 'skewY' in transform_substr:
        transform[1, 0] = np.tan(values[0] * np.pi / 180.0)
    else:
        # Return an identity matrix if the type of transform is unknown, and warn the user
        warnings.warn('Unknown SVG transform type: {0}'.format(type_str))
    return transform

def parse_transform(transform_str):
    """
        Converts a valid SVG transformation string into a 3x3 matrix.
        If the string is empty or null, this returns a 3x3 identity matrix
    """
    if not transform_str:
        return np.identity(3)
    elif not isinstance(transform_str, str):
        raise TypeError('Must provide a string to parse')

    total_transform = np.identity(3)
    transform_substrs = transform_str.split(')')[:-1]  # Skip the last element, because it should be empty
    for substr in transform_substrs:
        total_transform = total_transform.dot(_parse_transform_substr(substr))

    return torch.from_numpy(total_transform).type(torch.float32)

def parse_linear_gradient(node, transform, defs):
    begin = torch.tensor([0.0, 0.0])
    end = torch.tensor([0.0, 0.0])
    offsets = []
    stop_colors = []
    # Inherit from parent
    for key in node.attrib:
        if remove_namespaces(key) == 'href':
            value = node.attrib[key]
            parent = defs[value.lstrip('#')]
            begin = parent.begin
            end = parent.end
            offsets = parent.offsets
            stop_colors = parent.stop_colors

    for attrib in node.attrib:
        attrib = remove_namespaces(attrib)
        if attrib == 'x1':
            begin[0] = float(node.attrib['x1'])
        elif attrib == 'y1':
            begin[1] = float(node.attrib['y1'])
        elif attrib == 'x2':
            end[0] = float(node.attrib['x2'])
        elif attrib == 'y2':
            end[1] = float(node.attrib['y2'])
        elif attrib == 'gradientTransform':
            transform = transform @ parse_transform(node.attrib['gradientTransform'])

    begin = transform @ torch.cat((begin, torch.ones([1])))
    begin = begin / begin[2]
    begin = begin[:2]
    end = transform @ torch.cat((end, torch.ones([1])))
    end = end / end[2]
    end = end[:2]

    for child in node:
        tag = remove_namespaces(child.tag)
        if tag == 'stop':
            offset = float(child.attrib['offset'])
            color = [0.0, 0.0, 0.0, 1.0]
            if 'stop-color' in child.attrib:
                c = parse_color(child.attrib['stop-color'], defs)
                color[:3] = [c[0], c[1], c[2]]
            if 'stop-opacity' in child.attrib:
                color[3] = float(child.attrib['stop-opacity'])
            if 'style' in child.attrib:
                style = parse_style(child.attrib['style'], defs)
                if 'stop-color' in style:
                    c = parse_color(style['stop-color'], defs)
                    color[:3] = [c[0], c[1], c[2]]
                if 'stop-opacity' in style:
                    color[3] = float(style['stop-opacity'])
            offsets.append(offset)
            stop_colors.append(color)
    if isinstance(offsets, list):
        offsets = torch.tensor(offsets)
    if isinstance(stop_colors, list):
        stop_colors = torch.tensor(stop_colors)

    return pydiffvg.LinearGradient(begin, end, offsets, stop_colors)


def parse_radial_gradient(node, transform, defs):
    begin = torch.tensor([0.0, 0.0])
    end = torch.tensor([0.0, 0.0])
    center = torch.tensor([0.0, 0.0])
    radius = torch.tensor([0.0, 0.0])
    offsets = []
    stop_colors = []
    # Inherit from parent
    for key in node.attrib:
        if remove_namespaces(key) == 'href':
            value = node.attrib[key]
            parent = defs[value.lstrip('#')]
            begin = parent.begin
            end = parent.end
            offsets = parent.offsets
            stop_colors = parent.stop_colors

    for attrib in node.attrib:
        attrib = remove_namespaces(attrib)
        if attrib == 'cx':
            center[0] = float(node.attrib['cx'])
        elif attrib == 'cy':
            center[1] = float(node.attrib['cy'])
        elif attrib == 'fx':
            radius[0] = float(node.attrib['fx'])
        elif attrib == 'fy':
            radius[1] = float(node.attrib['fy'])
        elif attrib == 'fr':
            radius[0] = float(node.attrib['fr'])
            radius[1] = float(node.attrib['fr'])
        elif attrib == 'gradientTransform':
            transform = transform @ parse_transform(node.attrib['gradientTransform'])

    # TODO: this is incorrect
    center = transform @ torch.cat((center, torch.ones([1])))
    center = center / center[2]
    center = center[:2]

    for child in node:
        tag = remove_namespaces(child.tag)
        if tag == 'stop':
            offset = float(child.attrib['offset'])
            color = [0.0, 0.0, 0.0, 1.0]
            if 'stop-color' in child.attrib:
                c = parse_color(child.attrib['stop-color'], defs)
                color[:3] = [c[0], c[1], c[2]]
            if 'stop-opacity' in child.attrib:
                color[3] = float(child.attrib['stop-opacity'])
            if 'style' in child.attrib:
                style = parse_style(child.attrib['style'], defs)
                if 'stop-color' in style:
                    c = parse_color(style['stop-color'], defs)
                    color[:3] = [c[0], c[1], c[2]]
                if 'stop-opacity' in style:
                    color[3] = float(style['stop-opacity'])
            offsets.append(offset)
            stop_colors.append(color)
    if isinstance(offsets, list):
        offsets = torch.tensor(offsets)
    if isinstance(stop_colors, list):
        stop_colors = torch.tensor(stop_colors)

    return pydiffvg.RadialGradient(begin, end, offsets, stop_colors)

def parse_stylesheet(node, transform, defs):
    # collect CSS classes
    sheet = cssutils.parseString(node.text)
    for rule in sheet:
        if hasattr(rule, 'selectorText') and hasattr(rule, 'style'):
            name = rule.selectorText
            if len(name) >= 2 and name[0] == '.':
                defs[name[1:]] = parse_style(rule.style.getCssText(), defs)
    return defs

def parse_defs(node, transform, defs):
    for child in node:
        tag = remove_namespaces(child.tag)
        if tag == 'linearGradient':
            if 'id' in child.attrib:
                defs[child.attrib['id']] = parse_linear_gradient(child, transform, defs)
        elif tag == 'radialGradient':
            if 'id' in child.attrib:
                defs[child.attrib['id']] = parse_radial_gradient(child, transform, defs)
        elif tag == 'style':
            defs = parse_stylesheet(child, transform, defs)
    return defs

def parse_common_attrib(node, transform, fill_color, defs):
    attribs = {}
    if 'class' in node.attrib:
        attribs.update(defs[node.attrib['class']])
    attribs.update(node.attrib)

    name = ''
    if 'id' in node.attrib:
        name = node.attrib['id']

    stroke_color = None
    stroke_width = torch.tensor(0.5)
    use_even_odd_rule = False

    new_transform = transform
    if 'transform' in attribs:
        new_transform = transform @ parse_transform(attribs['transform'])
    if 'fill' in attribs:
        fill_color = parse_color(attribs['fill'], defs)
    fill_opacity = 1.0
    if 'fill-opacity' in attribs:
        fill_opacity *= float(attribs['fill-opacity'])
    if 'opacity' in attribs:
        fill_opacity *= float(attribs['opacity'])
    # Ignore opacity if the color is a gradient
    if isinstance(fill_color, torch.Tensor):
        fill_color[3] = fill_opacity

    if 'fill-rule' in attribs:
        if attribs['fill-rule'] == "evenodd":
            use_even_odd_rule = True
        elif attribs['fill-rule'] == "nonzero":
            use_even_odd_rule = False
        else:
            warnings.warn('Unknown fill-rule: {}'.format(attribs['fill-rule']))

    if 'stroke' in attribs:
        stroke_color = parse_color(attribs['stroke'], defs)

    if 'stroke-width' in attribs:
        stroke_width = attribs['stroke-width']
        if stroke_width[-2:] == 'px':
            stroke_width = stroke_width[:-2]
        stroke_width = torch.tensor(float(stroke_width) / 2.0)

    if 'stroke-opacity' in attribs:
        stroke_color[3] = torch.tensor(float(attribs['stroke-opacity']))

    if 'style' in attribs:
        style = parse_style(attribs['style'], defs)
        if 'fill' in style:
            fill_color = parse_color(style['fill'], defs)
        fill_opacity = 1.0
        if 'fill-opacity' in style:
            fill_opacity *= float(style['fill-opacity'])
        if 'opacity' in style:
            fill_opacity *= float(style['opacity'])
        if 'fill-rule' in style:
            if style['fill-rule'] == "evenodd":
                use_even_odd_rule = True
            elif style['fill-rule'] == "nonzero":
                use_even_odd_rule = False
            else:
                warnings.warn('Unknown fill-rule: {}'.format(style['fill-rule']))
        # Ignore opacity if the color is a gradient
        if isinstance(fill_color, torch.Tensor):
            fill_color[3] = fill_opacity
        if 'stroke' in style:
            if style['stroke'] != 'none':
                stroke_color = parse_color(style['stroke'], defs)
                # Ignore opacity if the color is a gradient
                if isinstance(stroke_color, torch.Tensor):
                    if 'stroke-opacity' in style:
                        stroke_color[3] = float(style['stroke-opacity'])
                    if 'opacity' in style:
                        stroke_color[3] *= float(style['opacity'])
                if 'stroke-width' in style:
                    stroke_width = style['stroke-width']
                    if stroke_width[-2:] == 'px':
                        stroke_width = stroke_width[:-2]
                    stroke_width = torch.tensor(float(stroke_width) / 2.0)

        if isinstance(fill_color, pydiffvg.LinearGradient):
            fill_color.begin = new_transform @ torch.cat((fill_color.begin, torch.ones([1])))
            fill_color.begin = fill_color.begin / fill_color.begin[2]
            fill_color.begin = fill_color.begin[:2]
            fill_color.end = new_transform @ torch.cat((fill_color.end, torch.ones([1])))
            fill_color.end = fill_color.end / fill_color.end[2]
            fill_color.end = fill_color.end[:2]
        if isinstance(stroke_color, pydiffvg.LinearGradient):
            stroke_color.begin = new_transform @ torch.cat((stroke_color.begin, torch.ones([1])))
            stroke_color.begin = stroke_color.begin / stroke_color.begin[2]
            stroke_color.begin = stroke_color.begin[:2]
            stroke_color.end = new_transform @ torch.cat((stroke_color.end, torch.ones([1])))
            stroke_color.end = stroke_color.end / stroke_color.end[2]
            stroke_color.end = stroke_color.end[:2]
        if 'filter' in style:
            print('*** WARNING ***: Ignoring filter for path with id "{}"'.format(name))

    return new_transform, fill_color, stroke_color, stroke_width, use_even_odd_rule

def is_shape(tag):
    return tag == 'path' or tag == 'polygon' or tag == 'line' or tag == 'circle' or tag == 'rect'

def parse_shape(node, transform, fill_color, shapes, shape_groups, defs):
    tag = remove_namespaces(node.tag)
    new_transform, new_fill_color, stroke_color, stroke_width, use_even_odd_rule = \
        parse_common_attrib(node, transform, fill_color, defs)
    if tag == 'path':
        d = node.attrib['d']
        name = ''
        if 'id' in node.attrib:
            name = node.attrib['id']
        force_closing = new_fill_color is not None
        paths = pydiffvg.from_svg_path(d, new_transform, force_closing)
        for idx, path in enumerate(paths):
            assert(path.points.shape[1] == 2)
            path.stroke_width = stroke_width
            path.source_id = name
            path.id = "{}-{}".format(name,idx) if len(paths)>1 else name
        prev_shapes_size = len(shapes)
        shapes = shapes + paths
        shape_ids = torch.tensor(list(range(prev_shapes_size, len(shapes))))
        shape_groups.append(pydiffvg.ShapeGroup(\
            shape_ids = shape_ids,
            fill_color = new_fill_color,
            stroke_color = stroke_color,
            use_even_odd_rule = use_even_odd_rule,
            id = name))
    elif tag == 'polygon':
        name = ''
        if 'id' in node.attrib:
            name = node.attrib['id']
        force_closing = new_fill_color is not None
        pts = node.attrib['points'].strip()
        pts = pts.split(' ')
        # import ipdb; ipdb.set_trace()
        pts = [[float(y) for y in re.split(',| ', x)] for x in pts if x]
        pts = torch.tensor(pts, dtype=torch.float32).view(-1, 2)
        polygon = pydiffvg.Polygon(pts, force_closing)
        polygon.stroke_width = stroke_width
        shape_ids = torch.tensor([len(shapes)])
        shapes.append(polygon)
        shape_groups.append(pydiffvg.ShapeGroup(\
            shape_ids = shape_ids,
            fill_color = new_fill_color,
            stroke_color = stroke_color,
            use_even_odd_rule = use_even_odd_rule,
            shape_to_canvas = new_transform,
            id = name))
    elif tag == 'line':
        x1 = float(node.attrib['x1'])
        y1 = float(node.attrib['y1'])
        x2 = float(node.attrib['x2'])
        y2 = float(node.attrib['y2'])
        p1 = torch.tensor([x1, y1])
        p2 = torch.tensor([x2, y2])
        points = torch.stack((p1, p2))
        line = pydiffvg.Polygon(points, False)
        line.stroke_width = stroke_width
        shape_ids = torch.tensor([len(shapes)])
        shapes.append(line)
        shape_groups.append(pydiffvg.ShapeGroup(\
            shape_ids = shape_ids,
            fill_color = new_fill_color,
            stroke_color = stroke_color,
            use_even_odd_rule = use_even_odd_rule,
            shape_to_canvas = new_transform))
    elif tag == 'circle':
        radius = float(node.attrib['r'])
        cx = float(node.attrib['cx'])
        cy = float(node.attrib['cy'])
        name = ''
        if 'id' in node.attrib:
            name = node.attrib['id']
        center = torch.tensor([cx, cy])
        circle = pydiffvg.Circle(radius = torch.tensor(radius),
                                 center = center)
        circle.stroke_width = stroke_width
        shape_ids = torch.tensor([len(shapes)])
        shapes.append(circle)
        shape_groups.append(pydiffvg.ShapeGroup(\
            shape_ids = shape_ids,
            fill_color = new_fill_color,
            stroke_color = stroke_color,
            use_even_odd_rule = use_even_odd_rule,
            shape_to_canvas = new_transform))
    elif tag == 'ellipse':
        rx = float(node.attrib['rx'])
        ry = float(node.attrib['ry'])
        cx = float(node.attrib['cx'])
        cy = float(node.attrib['cy'])
        name = ''
        if 'id' in node.attrib:
            name = node.attrib['id']
        center = torch.tensor([cx, cy])
        circle = pydiffvg.Circle(radius = torch.tensor(radius),
                                 center = center)
        circle.stroke_width = stroke_width
        shape_ids = torch.tensor([len(shapes)])
        shapes.append(circle)
        shape_groups.append(pydiffvg.ShapeGroup(\
            shape_ids = shape_ids,
            fill_color = new_fill_color,
            stroke_color = stroke_color,
            use_even_odd_rule = use_even_odd_rule,
            shape_to_canvas = new_transform))
    elif tag == 'rect':
        x = 0.0
        y = 0.0
        if x in node.attrib:
            x = float(node.attrib['x'])
        if y in node.attrib:
            y = float(node.attrib['y'])
        w = float(node.attrib['width'])
        h = float(node.attrib['height'])
        p_min = torch.tensor([x, y])
        p_max = torch.tensor([x + w, x + h])
        rect = pydiffvg.Rect(p_min = p_min, p_max = p_max)
        rect.stroke_width = stroke_width
        shape_ids = torch.tensor([len(shapes)])
        shapes.append(rect)
        shape_groups.append(pydiffvg.ShapeGroup(\
            shape_ids = shape_ids,
            fill_color = new_fill_color,
            stroke_color = stroke_color,
            use_even_odd_rule = use_even_odd_rule,
            shape_to_canvas = new_transform))
    return shapes, shape_groups

def parse_group(node, transform, fill_color, shapes, shape_groups, defs):
    if 'transform' in node.attrib:
        transform = transform @ parse_transform(node.attrib['transform'])
    if 'fill' in node.attrib:
        fill_color = parse_color(node.attrib['fill'], defs)
    for child in node:
        tag = remove_namespaces(child.tag)
        if is_shape(tag):
            shapes, shape_groups = parse_shape(\
                child, transform, fill_color, shapes, shape_groups, defs)
        elif tag == 'g':
            shapes, shape_groups = parse_group(\
                child, transform, fill_color, shapes, shape_groups, defs)
    return shapes, shape_groups

def parse_scene(node):
    canvas_width = -1
    canvas_height = -1
    defs = {}
    shapes = []
    shape_groups = []
    fill_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
    transform = torch.eye(3)
    if 'viewBox' in node.attrib:
        view_box_array = node.attrib['viewBox'].split()
        canvas_width = parse_int(view_box_array[2])
        canvas_height = parse_int(view_box_array[3])
    else:
        if 'width' in node.attrib:
            canvas_width = parse_int(node.attrib['width'])
        else:
            print('Warning: Can\'t find canvas width.')
        if 'height' in node.attrib:
            canvas_height = parse_int(node.attrib['height'])
        else:
            print('Warning: Can\'t find canvas height.')
    for child in node:
        tag = remove_namespaces(child.tag)
        if tag == 'defs':
            defs = parse_defs(child, transform, defs)
        elif tag == 'style':
            defs = parse_stylesheet(child, transform, defs)
        elif tag == 'linearGradient':
            if 'id' in child.attrib:
                defs[child.attrib['id']] = parse_linear_gradient(child, transform, defs)
        elif tag == 'radialGradient':
            if 'id' in child.attrib:
                defs[child.attrib['id']] = parse_radial_gradient(child, transform, defs)
        elif is_shape(tag):
            shapes, shape_groups = parse_shape(\
                child, transform, fill_color, shapes, shape_groups, defs)
        elif tag == 'g':
            shapes, shape_groups = parse_group(\
                child, transform, fill_color, shapes, shape_groups, defs)
    return canvas_width, canvas_height, shapes, shape_groups

def svg_to_scene(filename):
    """
        Load from a SVG file and convert to PyTorch tensors.
    """

    tree = etree.parse(filename)
    root = tree.getroot()
    cwd = os.getcwd()
    if (os.path.dirname(filename) != ''):
        os.chdir(os.path.dirname(filename))
    ret = parse_scene(root)
    os.chdir(cwd)
    return ret

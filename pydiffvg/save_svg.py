import torch
import pydiffvg
import xml.etree.ElementTree as etree
from xml.dom import minidom

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = etree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def save_svg(filename, width, height, shapes, shape_groups, use_gamma = False):
    root = etree.Element('svg')
    root.set('version', '1.1')
    root.set('xmlns', 'http://www.w3.org/2000/svg')
    root.set('width', str(width))
    root.set('height', str(height))
    defs = etree.SubElement(root, 'defs')
    g = etree.SubElement(root, 'g')
    if use_gamma:
        f = etree.SubElement(defs, 'filter')
        f.set('id', 'gamma')
        f.set('x', '0')
        f.set('y', '0')
        f.set('width', '100%')
        f.set('height', '100%')
        gamma = etree.SubElement(f, 'feComponentTransfer')
        gamma.set('color-interpolation-filters', 'sRGB')
        feFuncR = etree.SubElement(gamma, 'feFuncR')
        feFuncR.set('type', 'gamma')
        feFuncR.set('amplitude', str(1))
        feFuncR.set('exponent', str(1/2.2))
        feFuncG = etree.SubElement(gamma, 'feFuncG')
        feFuncG.set('type', 'gamma')
        feFuncG.set('amplitude', str(1))
        feFuncG.set('exponent', str(1/2.2))
        feFuncB = etree.SubElement(gamma, 'feFuncB')
        feFuncB.set('type', 'gamma')
        feFuncB.set('amplitude', str(1))
        feFuncB.set('exponent', str(1/2.2))
        feFuncA = etree.SubElement(gamma, 'feFuncA')
        feFuncA.set('type', 'gamma')
        feFuncA.set('amplitude', str(1))
        feFuncA.set('exponent', str(1/2.2))
        g.set('style', 'filter:url(#gamma)')

    # Store color
    for i, shape_group in enumerate(shape_groups):
        def add_color(shape_color, name):
            if isinstance(shape_color, pydiffvg.LinearGradient):
                lg = shape_color
                color = etree.SubElement(defs, 'linearGradient')
                color.set('id', name)
                color.set('x1', str(lg.begin[0].item()))
                color.set('y1', str(lg.begin[1].item()))
                color.set('x2', str(lg.end[0].item()))
                color.set('y2', str(lg.end[1].item()))
                offsets = lg.offsets.data.cpu().numpy()
                stop_colors = lg.stop_colors.data.cpu().numpy()
                for j in range(offsets.shape[0]):
                    stop = etree.SubElement(color, 'stop')
                    stop.set('offset', str(offsets[j]))
                    c = lg.stop_colors[j, :]
                    stop.set('stop-color', 'rgb({}, {}, {})'.format(\
                        int(255 * c[0]), int(255 * c[1]), int(255 * c[2])))
                    stop.set('stop-opacity', '{}'.format(c[3]))

        if shape_group.fill_color is not None:
            add_color(shape_group.fill_color, 'shape_{}_fill'.format(i))
        if shape_group.stroke_color is not None:
            add_color(shape_group.stroke_color, 'shape_{}_stroke'.format(i))

    for i, shape_group in enumerate(shape_groups):
        shape = shapes[shape_group.shape_ids[0]]
        if isinstance(shape, pydiffvg.Circle):
            shape_node = etree.SubElement(g, 'circle')
            shape_node.set('r', str(shape.radius.item()))
            shape_node.set('cx', str(shape.center[0].item()))
            shape_node.set('cy', str(shape.center[1].item()))
        elif isinstance(shape, pydiffvg.Polygon):
            shape_node = etree.SubElement(g, 'polygon')
            points = shape.points.data.cpu().numpy()
            path_str = ''
            for j in range(0, shape.points.shape[0]):
                path_str += '{} {}'.format(points[j, 0], points[j, 1])
                if j != shape.points.shape[0] - 1:
                    path_str +=  ' '
            shape_node.set('points', path_str)
        elif isinstance(shape, pydiffvg.Path):
            shape_node = etree.SubElement(g, 'path')
            num_segments = shape.num_control_points.shape[0]
            num_control_points = shape.num_control_points.data.cpu().numpy()
            points = shape.points.data.cpu().numpy()
            num_points = shape.points.shape[0]
            path_str = 'M {} {}'.format(points[0, 0], points[0, 1])
            point_id = 1
            for j in range(0, num_segments):
                if num_control_points[j] == 0:
                    p = point_id % num_points
                    path_str += ' L {} {}'.format(\
                            points[p, 0], points[p, 1])
                    point_id += 1
                elif num_control_points[j] == 1:
                    p1 = (point_id + 1) % num_points
                    path_str += ' Q {} {} {} {}'.format(\
                            points[point_id, 0], points[point_id, 1],
                            points[p1, 0], points[p1, 1])
                    point_id += 2
                elif num_control_points[j] == 2:
                    p2 = (point_id + 2) % num_points
                    path_str += ' C {} {} {} {} {} {}'.format(\
                            points[point_id, 0], points[point_id, 1],
                            points[point_id + 1, 0], points[point_id + 1, 1],
                            points[p2, 0], points[p2, 1])
                    point_id += 3
            shape_node.set('d', path_str)
        elif isinstance(shape, pydiffvg.Rect):
            shape_node = etree.SubElement(g, 'rect')
            shape_node.set('x', str(shape.p_min[0].item()))
            shape_node.set('y', str(shape.p_min[1].item()))
            shape_node.set('width', str(shape.p_max[0].item() - shape.p_min[0].item()))
            shape_node.set('height', str(shape.p_max[1].item() - shape.p_min[1].item()))
        elif isinstance(shape, pydiffvg.Ellipse):
            shape_node = etree.SubElement(g, 'ellipse')
            shape_node.set('cx', str(shape.center[0].item()))
            shape_node.set('cy', str(shape.center[1].item()))
            shape_node.set('rx', str(shape.radius[0].item()))
            shape_node.set('ry', str(shape.radius[1].item()))
        else:
            assert(False)

        shape_node.set('stroke-width', str(2 * shape.stroke_width.data.cpu().item()))
        if shape_group.fill_color is not None:
            if isinstance(shape_group.fill_color, pydiffvg.LinearGradient):
                shape_node.set('fill', 'url(#shape_{}_fill)'.format(i))
            else:
                c = shape_group.fill_color.data.cpu().numpy()
                shape_node.set('fill', 'rgb({}, {}, {})'.format(\
                    int(255 * c[0]), int(255 * c[1]), int(255 * c[2])))
                shape_node.set('opacity', str(c[3]))
        else:
            shape_node.set('fill', 'none')
        if shape_group.stroke_color is not None:
            if isinstance(shape_group.stroke_color, pydiffvg.LinearGradient):
                shape_node.set('stroke', 'url(#shape_{}_stroke)'.format(i))
            else:
                c = shape_group.stroke_color.data.cpu().numpy()
                shape_node.set('stroke', 'rgb({}, {}, {})'.format(\
                    int(255 * c[0]), int(255 * c[1]), int(255 * c[2])))
                shape_node.set('stroke-opacity', str(c[3]))
            shape_node.set('stroke-linecap', 'round')
            shape_node.set('stroke-linejoin', 'round')

    with open(filename, "w") as f:
        f.write(prettify(root))

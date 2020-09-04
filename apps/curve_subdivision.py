import svgpathtools
import numpy as np
import math

def split_cubic(c, t):
    c0, c1 = svgpathtools.split_bezier(c, t)
    return svgpathtools.CubicBezier(c0[0], c0[1], c0[2], c0[3]), svgpathtools.CubicBezier(c1[0], c1[1], c1[2], c1[3])

def cubic_to_quadratic(curve):
    # Best L2 approximation
    m = (-curve.start + 3 * curve.control1 + 3 * curve.control2 - curve.end) / 4.0
    return svgpathtools.QuadraticBezier(curve.start, m, curve.end)

def convert_and_write_svg(cubic, filename):
    cubic_path = svgpathtools.Path(cubic)
    cubic_ctrl = svgpathtools.Path(svgpathtools.Line(cubic.start, cubic.control1),
                                   svgpathtools.Line(cubic.control1, cubic.control2),
                                   svgpathtools.Line(cubic.control2, cubic.end))
    cubic_color = (50, 50, 200)
    cubic_ctrl_color = (150, 150, 150)

    r = 4.0

    paths = [cubic_path, cubic_ctrl]
    colors = [cubic_color, cubic_ctrl_color]
    dots = [cubic_path[0].start, cubic_path[0].control1, cubic_path[0].control2, cubic_path[0].end]
    ncols = ['green', 'green', 'green', 'green']
    nradii = [r, r, r, r]
    stroke_widths = [3.0, 1.5]

    def add_quadratic(q):
        paths.append(q)
        q_ctrl = svgpathtools.Path(svgpathtools.Line(q.start, q.control),
                                   svgpathtools.Line(q.control, q.end))
        paths.append(q_ctrl)
        colors.append((200, 50, 50)) # q_color
        colors.append((150, 150, 150)) # q_ctrl_color
        dots.append(q.start)
        dots.append(q.control)
        dots.append(q.end)
        ncols.append('purple')
        ncols.append('purple')
        ncols.append('purple')
        nradii.append(r)
        nradii.append(r)
        nradii.append(r)
        stroke_widths.append(3.0)
        stroke_widths.append(1.5)

    prec = 1.0
    queue = [cubic]
    num_quadratics = 0
    while len(queue) > 0:
        c = queue[-1]
        queue = queue[:-1]

        # Criteria for conversion
        # http://caffeineowl.com/graphics/2d/vectorial/cubic2quad01.html
        p = c.end - 3 * c.control2 + 3 * c.control1 - c.start
        d = math.sqrt(p.real * p.real + p.imag * p.imag) * math.sqrt(3.0) / 36
        t = math.pow(1.0 / d, 1.0 / 3.0)

        if t < 1.0:
            c0, c1 = split_cubic(c, 0.5)
            queue.append(c0)
            queue.append(c1)
        else:
            quadratic = cubic_to_quadratic(c)
            print(quadratic)
            add_quadratic(quadratic)
            num_quadratics += 1
    print('num_quadratics:', num_quadratics)

    svgpathtools.wsvg(paths,
                      colors = colors,
                      stroke_widths = stroke_widths,
                      nodes = dots,
                      node_colors = ncols,
                      node_radii = nradii,
                      filename = filename)

convert_and_write_svg(svgpathtools.CubicBezier(100+200j, 426+50j, 50+50j, 300+200j),
                      'results/curve_subdivision/subdiv_curve0.svg')
convert_and_write_svg(svgpathtools.CubicBezier(100+200j, 427+50j, 50+50j, 300+200j),
                      'results/curve_subdivision/subdiv_curve1.svg')

"""Collection of generative models."""

import torch as th
import ttools

import rendering
import modules

LOG = ttools.get_logger(__name__)


class BaseModel(th.nn.Module):
    def sample_z(self, bs, device="cpu"):
        return th.randn(bs, self.zdim).to(device)


class BaseVectorModel(BaseModel):
    def get_vector(self, z):
        _, scenes = self._forward(z)
        return scenes

    def _forward(self, x):
        raise NotImplementedError()

    def forward(self, z):
        # Only return the raster
        return self._forward(z)[0]


class BezierVectorGenerator(BaseVectorModel):
    NUM_SEGMENTS = 2
    def __init__(self, num_strokes=4,
                 zdim=128, width=32, imsize=32,
                 color_output=False,
                 stroke_width=None):
        super(BezierVectorGenerator, self).__init__()

        if stroke_width is None:
            self.stroke_width = (0.5, 3.0)
            LOG.warning("Setting default stroke with %s", self.stroke_width)
        else:
            self.stroke_width = stroke_width

        self.imsize = imsize
        self.num_strokes = num_strokes
        self.zdim = zdim

        self.trunk = th.nn.Sequential(
            th.nn.Linear(zdim, width),
            th.nn.SELU(inplace=True),

            th.nn.Linear(width, 2*width),
            th.nn.SELU(inplace=True),

            th.nn.Linear(2*width, 4*width),
            th.nn.SELU(inplace=True),

            th.nn.Linear(4*width, 8*width),
            th.nn.SELU(inplace=True),
        )

        # 4 points bezier with n_segments -> 3*n_segments + 1 points
        self.point_predictor = th.nn.Sequential(
            th.nn.Linear(8*width, 
                         2*self.num_strokes*(
                             BezierVectorGenerator.NUM_SEGMENTS*3 + 1)),
            th.nn.Tanh()  # bound spatial extent
        )

        self.width_predictor = th.nn.Sequential(
            th.nn.Linear(8*width, self.num_strokes),
            th.nn.Sigmoid()
        )

        self.alpha_predictor = th.nn.Sequential(
            th.nn.Linear(8*width, self.num_strokes),
            th.nn.Sigmoid()
        )

        self.color_predictor = None
        if color_output:
            self.color_predictor = th.nn.Sequential(
                th.nn.Linear(8*width, 3*self.num_strokes),
                th.nn.Sigmoid()
            )

    def _forward(self, z):
        bs = z.shape[0]

        feats = self.trunk(z)
        all_points = self.point_predictor(feats)
        all_alphas = self.alpha_predictor(feats)

        if self.color_predictor:
            all_colors = self.color_predictor(feats)
            all_colors = all_colors.view(bs, self.num_strokes, 3)
        else:
            all_colors = None

        all_widths = self.width_predictor(feats)
        min_width = self.stroke_width[0]
        max_width = self.stroke_width[1]
        all_widths = (max_width - min_width) * all_widths + min_width

        all_points = all_points.view(
            bs, self.num_strokes, BezierVectorGenerator.NUM_SEGMENTS*3+1, 2)

        output, scenes = rendering.bezier_render(all_points, all_widths, all_alphas,
                                         colors=all_colors,
                                         canvas_size=self.imsize)

        # map to [-1, 1]
        output = output*2.0 - 1.0

        return output, scenes


class VectorGenerator(BaseVectorModel):
    def __init__(self, num_strokes=4,
                 zdim=128, width=32, imsize=32,
                 color_output=False,
                 stroke_width=None):
        super(VectorGenerator, self).__init__()

        if stroke_width is None:
            self.stroke_width = (0.5, 3.0)
            LOG.warning("Setting default stroke with %s", self.stroke_width)
        else:
            self.stroke_width = stroke_width

        self.imsize = imsize
        self.num_strokes = num_strokes
        self.zdim = zdim

        self.trunk = th.nn.Sequential(
            th.nn.Linear(zdim, width),
            th.nn.SELU(inplace=True),

            th.nn.Linear(width, 2*width),
            th.nn.SELU(inplace=True),

            th.nn.Linear(2*width, 4*width),
            th.nn.SELU(inplace=True),

            th.nn.Linear(4*width, 8*width),
            th.nn.SELU(inplace=True),
        )

        # straight lines so n_segments -> n_segments - 1 points
        self.point_predictor = th.nn.Sequential(
            th.nn.Linear(8*width, 2*(self.num_strokes*2)),
            th.nn.Tanh()  # bound spatial extent
        )

        self.width_predictor = th.nn.Sequential(
            th.nn.Linear(8*width, self.num_strokes),
            th.nn.Sigmoid()
        )

        self.alpha_predictor = th.nn.Sequential(
            th.nn.Linear(8*width, self.num_strokes),
            th.nn.Sigmoid()
        )

        self.color_predictor = None
        if color_output:
            self.color_predictor = th.nn.Sequential(
                th.nn.Linear(8*width, 3*self.num_strokes),
                th.nn.Sigmoid()
            )

    def _forward(self, z):
        bs = z.shape[0]

        feats = self.trunk(z)

        all_points = self.point_predictor(feats)

        all_alphas = self.alpha_predictor(feats)

        if self.color_predictor:
            all_colors = self.color_predictor(feats)
            all_colors = all_colors.view(bs, self.num_strokes, 3)
        else:
            all_colors = None

        all_widths = self.width_predictor(feats)
        min_width = self.stroke_width[0]
        max_width = self.stroke_width[1]
        all_widths = (max_width - min_width) * all_widths + min_width

        all_points = all_points.view(bs, self.num_strokes, 2, 2)
        output, scenes = rendering.line_render(all_points, all_widths, all_alphas,
                                       colors=all_colors,
                                       canvas_size=self.imsize)

        # map to [-1, 1]
        output = output*2.0 - 1.0

        return output, scenes


class RNNVectorGenerator(BaseVectorModel):
    def __init__(self, num_strokes=64,
                 zdim=128, width=32, imsize=32,
                 hidden_size=512, dropout=0.9,
                 color_output=False,
                 num_layers=3, stroke_width=None):
        super(RNNVectorGenerator, self).__init__()


        if stroke_width is None:
            self.stroke_width = (0.5, 3.0)
            LOG.warning("Setting default stroke with %s", self.stroke_width)
        else:
            self.stroke_width = stroke_width

        self.num_layers = num_layers
        self.imsize = imsize
        self.num_strokes = num_strokes
        self.hidden_size = hidden_size
        self.zdim = zdim

        self.hidden_cell_predictor = th.nn.Linear(
            zdim, 2*hidden_size*num_layers)

        self.lstm = th.nn.LSTM(
            zdim, hidden_size,
            num_layers=self.num_layers, dropout=dropout,
            batch_first=True)

        # straight lines so n_segments -> n_segments - 1 points
        self.point_predictor = th.nn.Sequential(
            th.nn.Linear(hidden_size, 2*2),  # 2 points, (x,y)
            th.nn.Tanh()  # bound spatial extent
        )

        self.width_predictor = th.nn.Sequential(
            th.nn.Linear(hidden_size, 1),
            th.nn.Sigmoid()
        )

        self.alpha_predictor = th.nn.Sequential(
            th.nn.Linear(hidden_size, 1),
            th.nn.Sigmoid()
        )

    def _forward(self, z, hidden_and_cell=None):
        steps = self.num_strokes

        # z is passed at each step, duplicate it
        bs = z.shape[0]
        expanded_z = z.unsqueeze(1).repeat(1, steps, 1)

        # First step in the RNN
        if hidden_and_cell is None:
            # Initialize from latent vector
            hidden_and_cell = self.hidden_cell_predictor(th.tanh(z))
            hidden = hidden_and_cell[:, :self.hidden_size*self.num_layers]
            hidden = hidden.view(-1, self.num_layers, self.hidden_size)
            hidden = hidden.permute(1, 0, 2).contiguous()
            cell = hidden_and_cell[:, self.hidden_size*self.num_layers:]
            cell = cell.view(-1, self.num_layers, self.hidden_size)
            cell = cell.permute(1, 0, 2).contiguous()
            hidden_and_cell = (hidden, cell)

        feats, hidden_and_cell = self.lstm(expanded_z, hidden_and_cell)
        hidden, cell = hidden_and_cell

        feats = feats.reshape(bs*steps, self.hidden_size)

        all_points = self.point_predictor(feats).view(bs, steps, 2, 2)
        all_alphas = self.alpha_predictor(feats).view(bs, steps)
        all_widths = self.width_predictor(feats).view(bs, steps)

        min_width = self.stroke_width[0]
        max_width = self.stroke_width[1]
        all_widths = (max_width - min_width) * all_widths + min_width

        output, scenes = rendering.line_render(all_points, all_widths, all_alphas,
                                        canvas_size=self.imsize)

        # map to [-1, 1]
        output = output*2.0 - 1.0

        return output, scenes


class ChainRNNVectorGenerator(BaseVectorModel):
    """Strokes form a single long chain."""
    def __init__(self, num_strokes=64,
                 zdim=128, width=32, imsize=32,
                 hidden_size=512, dropout=0.9,
                 color_output=False,
                 num_layers=3, stroke_width=None):
        super(ChainRNNVectorGenerator, self).__init__()

        if stroke_width is None:
            self.stroke_width = (0.5, 3.0)
            LOG.warning("Setting default stroke with %s", self.stroke_width)
        else:
            self.stroke_width = stroke_width

        self.num_layers = num_layers
        self.imsize = imsize
        self.num_strokes = num_strokes
        self.hidden_size = hidden_size
        self.zdim = zdim

        self.hidden_cell_predictor = th.nn.Linear(
            zdim, 2*hidden_size*num_layers)

        self.lstm = th.nn.LSTM(
            zdim, hidden_size,
            num_layers=self.num_layers, dropout=dropout,
            batch_first=True)

        # straight lines so n_segments -> n_segments - 1 points
        self.point_predictor = th.nn.Sequential(
            th.nn.Linear(hidden_size, 2),  # 1 point, (x,y)
            th.nn.Tanh()  # bound spatial extent
        )

        self.width_predictor = th.nn.Sequential(
            th.nn.Linear(hidden_size, 1),
            th.nn.Sigmoid()
        )

        self.alpha_predictor = th.nn.Sequential(
            th.nn.Linear(hidden_size, 1),
            th.nn.Sigmoid()
        )

    def _forward(self, z, hidden_and_cell=None):
        steps = self.num_strokes

        # z is passed at each step, duplicate it
        bs = z.shape[0]
        expanded_z = z.unsqueeze(1).repeat(1, steps, 1)

        # First step in the RNN
        if hidden_and_cell is None:
            # Initialize from latent vector
            hidden_and_cell = self.hidden_cell_predictor(th.tanh(z))
            hidden = hidden_and_cell[:, :self.hidden_size*self.num_layers]
            hidden = hidden.view(-1, self.num_layers, self.hidden_size)
            hidden = hidden.permute(1, 0, 2).contiguous()
            cell = hidden_and_cell[:, self.hidden_size*self.num_layers:]
            cell = cell.view(-1, self.num_layers, self.hidden_size)
            cell = cell.permute(1, 0, 2).contiguous()
            hidden_and_cell = (hidden, cell)

        feats, hidden_and_cell = self.lstm(expanded_z, hidden_and_cell)
        hidden, cell = hidden_and_cell

        feats = feats.reshape(bs*steps, self.hidden_size)

        # Construct the chain
        end_points = self.point_predictor(feats).view(bs, steps, 1, 2)
        start_points = th.cat([
            # first point is canvas center
            th.zeros(bs, 1, 1, 2, device=feats.device),
            end_points[:, 1:, :, :]], 1)
        all_points = th.cat([start_points, end_points], 2)

        all_alphas = self.alpha_predictor(feats).view(bs, steps)
        all_widths = self.width_predictor(feats).view(bs, steps)

        min_width = self.stroke_width[0]
        max_width = self.stroke_width[1]
        all_widths = (max_width - min_width) * all_widths + min_width

        output, scenes = rendering.line_render(all_points, all_widths, all_alphas,
                                        canvas_size=self.imsize)

        # map to [-1, 1]
        output = output*2.0 - 1.0

        return output, scenes


class Generator(BaseModel):
    def __init__(self, width=64, imsize=32, zdim=128,
                 stroke_width=None,
                 color_output=False,
                 num_strokes=4):
        super(Generator, self).__init__()
        assert imsize == 32

        self.imsize = imsize
        self.zdim = zdim

        num_in_chans = self.zdim // (2*2)
        num_out_chans = 3 if color_output else 1

        self.net = th.nn.Sequential(
            th.nn.ConvTranspose2d(num_in_chans, width*8, 4, padding=1,
                                  stride=2),
            th.nn.LeakyReLU(0.2, inplace=True),
            th.nn.Conv2d(width*8, width*8, 3, padding=1),
            th.nn.BatchNorm2d(width*8),
            th.nn.LeakyReLU(0.2, inplace=True),
            # 4x4

            th.nn.ConvTranspose2d(8*width, 4*width, 4, padding=1, stride=2),
            th.nn.LeakyReLU(0.2, inplace=True),
            th.nn.Conv2d(4*width, 4*width, 3, padding=1),
            th.nn.BatchNorm2d(width*4),
            th.nn.LeakyReLU(0.2, inplace=True),
            # 8x8

            th.nn.ConvTranspose2d(4*width, 2*width, 4, padding=1, stride=2),
            th.nn.LeakyReLU(0.2, inplace=True),
            th.nn.Conv2d(2*width, 2*width, 3, padding=1),
            th.nn.BatchNorm2d(width*2),
            th.nn.LeakyReLU(0.2, inplace=True),
            # 16x16

            th.nn.ConvTranspose2d(2*width, width, 4, padding=1, stride=2),
            th.nn.LeakyReLU(0.2, inplace=True),
            th.nn.Conv2d(width, width, 3, padding=1),
            th.nn.BatchNorm2d(width),
            th.nn.LeakyReLU(0.2, inplace=True),
            # 32x32

            th.nn.Conv2d(width, width, 3, padding=1),
            th.nn.BatchNorm2d(width),
            th.nn.LeakyReLU(0.2, inplace=True),
            th.nn.Conv2d(width, width, 3, padding=1),
            th.nn.LeakyReLU(0.2, inplace=True),
            th.nn.Conv2d(width, num_out_chans, 1),

            th.nn.Tanh(),
        )

    def forward(self, z):
        bs = z.shape[0]
        num_in_chans = self.zdim // (2*2)
        raster = self.net(z.view(bs, num_in_chans, 2, 2))
        return raster


class Discriminator(th.nn.Module):
    def __init__(self, conditional=False, width=64, color_output=False):
        super(Discriminator, self).__init__()

        self.conditional = conditional

        sn = th.nn.utils.spectral_norm

        num_chan_in = 3 if color_output else 1

        self.net = th.nn.Sequential(
            th.nn.Conv2d(num_chan_in, width, 3, padding=1),
            th.nn.LeakyReLU(0.2, inplace=True),
            th.nn.Conv2d(width, 2*width, 4, padding=1, stride=2),
            th.nn.LeakyReLU(0.2, inplace=True),
            # 16x16

            sn(th.nn.Conv2d(2*width, 2*width, 3, padding=1)),
            th.nn.LeakyReLU(0.2, inplace=True),
            sn(th.nn.Conv2d(2*width, 4*width, 4, padding=1, stride=2)),
            th.nn.LeakyReLU(0.2, inplace=True),
            # 8x8

            sn(th.nn.Conv2d(4*width, 4*width, 3, padding=1)),
            th.nn.LeakyReLU(0.2, inplace=True),
            sn(th.nn.Conv2d(4*width, width*4, 4, padding=1, stride=2)),
            th.nn.LeakyReLU(0.2, inplace=True),
            # 4x4

            sn(th.nn.Conv2d(4*width, 4*width, 3, padding=1)),
            th.nn.LeakyReLU(0.2, inplace=True),
            sn(th.nn.Conv2d(4*width, width*4, 4, padding=1, stride=2)),
            th.nn.LeakyReLU(0.2, inplace=True),
            # 2x2

            modules.Flatten(),
            th.nn.Linear(width*4*2*2, 1),
        )

    def forward(self, x):
        out = self.net(x)
        return out

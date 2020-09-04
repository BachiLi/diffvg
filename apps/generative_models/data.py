import os
import time
import torch as th
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
import imageio

import ttools
import rendering

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
DATA = os.path.join(BASE_DIR, "data")

LOG = ttools.get_logger(__name__)


class QuickDrawImageDataset(th.utils.data.Dataset):
    BASE_DATA_URL = \
        "https://console.cloud.google.com/storage/browser/_details/quickdraw_dataset/full/numpy_bitmap/cat.npy"
    """
    Args:
        spatial_limit(int): maximum spatial extent in pixels.
    """
    def __init__(self, imsize, train=True):
        super(QuickDrawImageDataset, self).__init__()
        file = os.path.join(DATA, "cat.npy")

        self.imsize = imsize

        if not os.path.exists(file):
            msg = "Dataset file %s does not exist, please download"
            " it from %s" % (file, QuickDrawImageDataset.BASE_DATA_URL)
            LOG.error(msg)
            raise RuntimeError(msg)

        self.data = np.load(file, allow_pickle=True, encoding="latin1")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        im = np.reshape(self.data[idx], (1, 1, 28, 28))
        im = th.from_numpy(im).float() / 255.0
        im = th.nn.functional.interpolate(im, size=(self.imsize, self.imsize))

        # Bring it to [-1, 1]
        im = th.clamp(im, 0, 1)
        im -= 0.5
        im /= 0.5

        return im.squeeze(0)


class QuickDrawDataset(th.utils.data.Dataset):
    BASE_DATA_URL = \
        "https://storage.cloud.google.com/quickdraw_dataset/sketchrnn"

    """
    Args:
        spatial_limit(int): maximum spatial extent in pixels.
    """
    def __init__(self, dataset, mode="train",
                 max_seq_length=250,
                 spatial_limit=1000):
        super(QuickDrawDataset, self).__init__()
        file = os.path.join(DATA, "sketchrnn_"+dataset)
        remote = os.path.join(QuickDrawDataset.BASE_DATA_URL, dataset)

        self.max_seq_length = max_seq_length
        self.spatial_limit = spatial_limit

        if mode not in ["train", "test", "valid"]:
            return ValueError("Only allowed data mode are 'train' and 'test',"
                              " 'valid'.")

        if not os.path.exists(file):
            msg = "Dataset file %s does not exist, please download"
            " it from %s" % (file, remote)
            LOG.error(msg)
            raise RuntimeError(msg)

        data = np.load(file, allow_pickle=True, encoding="latin1")[mode]
        data = self.purify(data)
        data = self.normalize(data)

        # Length of longest sequence in the dataset
        self.nmax = max([len(seq) for seq in data])
        self.sketches = data

    def __repr__(self):
        return "Dataset with %d sequences of max length %d" % \
            (len(self.sketches), self.nmax)

    def __len__(self):
        return len(self.sketches)

    def __getitem__(self, idx):
        """Return the idx-th stroke in 5-D format, padded to length (Nmax+2).

        The first and last element of the sequence are fixed to "start-" and
        "end-of-sequence" token.

        dx, dy, + 3 numbers for one-hot encoding of state:
        1 0 0: pen touching paper till next point
        0 1 0: pen lifted from paper after current point
        0 0 1: drawing has ended, next points (including current will not be
            drawn)
        """
        sample_data = self.sketches[idx]

        # Allow two extra slots for start/end of sequence tokens
        sample = np.zeros((self.nmax+2, 5), dtype=np.float32)

        n = sample_data.shape[0]

        # normalize dx, dy
        deltas = sample_data[:, :2]
        # Absolute coordinates
        positions = deltas[..., :2].cumsum(0)
        maxi = np.abs(positions).max() + 1e-8
        deltas = deltas / (1.1 * maxi)  # leave some margin on edges

        # fill in dx, dy coordinates
        sample[1:n+1, :2] = deltas

        # on paper indicator: 0 means touching paper in the 3d format, flip it
        sample[1:n+1, 2] = 1 - sample_data[:, 2]

        # off-paper indicator, complement of previous flag
        sample[1:n+1, 3] = 1 - sample[1:n+1, 2]

        # fill with end of sequence tokens for the remainder
        sample[n+1:, 4] = 1

        # Start of sequence token
        sample[0] = [0, 0, 1, 0, 0]

        return sample

    def purify(self, strokes):
        """removes to small or too long sequences + removes large gaps"""
        data = []
        for seq in strokes:
            if seq.shape[0] <= self.max_seq_length:
                # and seq.shape[0] > 10:

                # Limit large spatial gaps
                seq = np.minimum(seq, self.spatial_limit)
                seq = np.maximum(seq, -self.spatial_limit)
                seq = np.array(seq, dtype=np.float32)
                data.append(seq)
        return data

    def calculate_normalizing_scale_factor(self, strokes):
        """Calculate the normalizing factor explained in appendix of
        sketch-rnn."""
        data = []
        for i, stroke_i in enumerate(strokes):
            for j, pt in enumerate(strokes[i]):
                data.append(pt[0])
                data.append(pt[1])
        data = np.array(data)
        return np.std(data)

    def normalize(self, strokes):
        """Normalize entire dataset (delta_x, delta_y) by the scaling
        factor."""
        data = []
        scale_factor = self.calculate_normalizing_scale_factor(strokes)
        for seq in strokes:
            seq[:, 0:2] /= scale_factor
            data.append(seq)
        return data


class FixedLengthQuickDrawDataset(QuickDrawDataset):
    """A variant of the QuickDraw dataset where the strokes are represented as 
    a fixed-length sequence of triplets (dx, dy, opacity), where opacity = 0, 1.
    """
    def __init__(self, *args, canvas_size=64, **kwargs):
        super(FixedLengthQuickDrawDataset, self).__init__(*args, **kwargs)
        self.canvas_size = canvas_size

    def __getitem__(self, idx):
        sample = super(FixedLengthQuickDrawDataset, self).__getitem__(idx)

        # We construct a stroke opacity variable from the pen down state, dx, dy remain unchanged
        strokes = sample[:, :3]

        im = np.zeros((1, 1))

        # render image
        # start = time.time()
        im = rendering.opacityStroke2diffvg(
            th.from_numpy(strokes).unsqueeze(0), canvas_size=self.canvas_size,
            relative=True, debug=False)
        im = im.squeeze(0).numpy()
        # elapsed = (time.time() - start)*1000
        # print("item %d pipeline gt rendering took %.2fms" % (idx, elapsed))

        return strokes, im


class MNISTDataset(th.utils.data.Dataset):
    def __init__(self, imsize, train=True):
        super(MNISTDataset, self).__init__()
        self.mnist = dset.MNIST(root=os.path.join(DATA, "mnist"),
                                train=train,
                                download=True,
                                transform=transforms.Compose([
                                    transforms.Resize((imsize, imsize)),
                                    transforms.ToTensor(),
                                ]))

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        im, label = self.mnist[idx]

        # make sure data uses [0, 1] range
        im -= im.min()
        im /= im.max() + 1e-8

        # Bring it to [-1, 1]
        im -= 0.5
        im /= 0.5
        return im

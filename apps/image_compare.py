import argparse
import skimage.io
import numpy as np
from matplotlib import cm
import math
from skimage.metrics import structural_similarity as ssim

def normalize(x, min_, max_):
    return (x - min_) / (max_ - min_)

def main(args):
    img1 = skimage.img_as_float(skimage.io.imread(args.img1)).astype(np.float32)
    img2 = skimage.img_as_float(skimage.io.imread(args.img2)).astype(np.float32)
    ref = skimage.img_as_float(skimage.io.imread(args.ref)).astype(np.float32)
    img1 = img1[:, :, :3]
    img2 = img2[:, :, :3]
    ref = ref[:, :, :3]

    diff1 = np.sum(np.abs(img1 - ref), axis = 2)
    diff2 = np.sum(np.abs(img2 - ref), axis = 2)
    min_ = min(np.min(diff1), np.min(diff2))
    max_ = max(np.max(diff1), np.max(diff2)) * 0.5
    diff1 = cm.viridis(normalize(diff1, min_, max_))
    diff2 = cm.viridis(normalize(diff2, min_, max_))

    # MSE
    print('MSE img1:', np.mean(np.power(img1 - ref, 2.0)))
    print('MSE img2:', np.mean(np.power(img2 - ref, 2.0)))
    # PSNR
    print('PSNR img1:', 20 * math.log10(1.0 / math.sqrt(np.mean(np.power(img1 - ref, 2.0)))))
    print('PSNR img2:', 20 * math.log10(1.0 / math.sqrt(np.mean(np.power(img2 - ref, 2.0)))))
    # SSIM
    print('SSIM img1:', ssim(img1, ref, multichannel=True))
    print('SSIM img2:', ssim(img2, ref, multichannel=True))

    skimage.io.imsave('diff1.png', (diff1 * 255).astype(np.uint8))
    skimage.io.imsave('diff2.png', (diff2 * 255).astype(np.uint8))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("img1", help="img1")
    parser.add_argument("img2", help="img2")
    parser.add_argument("ref", help="ref")
    args = parser.parse_args()
    main(args)

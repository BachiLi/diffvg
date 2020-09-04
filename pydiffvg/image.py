import numpy as np
import skimage
import skimage.io
import os

def imwrite(img, filename, gamma = 2.2, normalize = False):
    directory = os.path.dirname(filename)
    if directory != '' and not os.path.exists(directory):
        os.makedirs(directory)

    if not isinstance(img, np.ndarray):
        img = img.data.numpy()
    if normalize:
        img_rng = np.max(img) - np.min(img)
        if img_rng > 0:
            img = (img - np.min(img)) / img_rng
    img = np.clip(img, 0.0, 1.0)
    if img.ndim==2:
        #repeat along the third dimension
        img=np.expand_dims(img,2)
    img[:, :, :3] = np.power(img[:, :, :3], 1.0/gamma)
    skimage.io.imsave(filename, (img * 255).astype(np.uint8))
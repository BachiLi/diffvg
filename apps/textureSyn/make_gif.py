# for gif making
import imageio
import numpy as np
import os
from PIL import Image
from math import floor

def make_gif(savePath, outputPath, frame_every_X_steps=15, repeat_ending=15, total_iter=200):
    number_files = len(os.listdir(savePath)) - 2
    frame_every_X_steps = frame_every_X_steps
    repeat_ending = repeat_ending
    steps = np.arange(floor(number_files / frame_every_X_steps)) * frame_every_X_steps
    steps = steps + (number_files - np.max(steps))

    images = []
    for f in range(total_iter-1):
    # for f in steps:
        filename = savePath + 'iter_' + str(f+1) + '.png'
        images.append(imageio.imread(filename))

    # repeat ending
    for _ in range(repeat_ending):
        filename = savePath + 'final.png'
        # filename = savePath + 'iter_' + str(number_files) + '.png'
        images.append(imageio.imread(filename))
    imageio.mimsave(outputPath, images)
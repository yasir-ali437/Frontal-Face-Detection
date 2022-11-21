import os
import numpy as np
from PIL import Image


if __name__ == '__main__':
    filespath = os.getcwd() + '/image-inference'  # Dataset directory
    pathDir = os.listdir(filespath)  # Images in dataset directory
    num = len(pathDir)  # Here (512512) is the size of each image

    print("Computing mean...")
    data_mean = 0.
    for idx in range(len(pathDir)):
        filename = pathDir[idx]
        img = Image.open(os.path.join(filespath, filename)).convert('L').resize((512, 512))
        img = np.array(img) / 255.0
        data_mean += np.mean(img)  # Take all the data of the first dimension in the three-dimensional matrix
		# As the use of gray images, so calculate a channel on it
    data_mean = data_mean / num

    print("Computing var...")
    data_std = 0.
    for idx in range(len(pathDir)):
        filename = pathDir[idx]
        img = Image.open(os.path.join(filespath, filename)).convert('L').resize((512, 512))
        img = np.array(img) / 255.0
        data_std += np.std(img)

    data_std = data_std / num
    print("mean:{}".format(data_mean))
    print("std:{}".format(data_std))

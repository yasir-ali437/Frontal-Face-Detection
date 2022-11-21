import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('inference1.jpg',0)
cv2.imshow( "FRAME", img)
cv2.waitKey(0)
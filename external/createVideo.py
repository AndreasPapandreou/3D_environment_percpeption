import cv2
import os.path
from os import path
import numpy as np
import glob
 
init_id = 7838
final_id = 11778
ids = []
for i in range(init_id, final_id+20, 20):
	ids.append(i)

img_array = []
for i in ids:
	filename = '/home/andreas/Documents/Caramel/caramelav/res/images/data_2/im_' + str(i) + ".png"
		
	exist = str(path.exists(filename))
	if (exist == "True"):
		img = cv2.imread(filename)
		height, width, layers = img.shape
		size = (width,height)
		img_array.append(img)

out = cv2.VideoWriter('demo.avi',cv2.VideoWriter_fourcc(*'DIVX'), 3, size)
 
for i in range(len(img_array)):
	out.write(img_array[i])
out.release()
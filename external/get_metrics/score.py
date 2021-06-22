import os
import cv2
import os.path
from os import path
from shutil import copyfile
from evalPixelLevelSemanticLabeling import evaluateImgLists

path_to_file = "/home/andreas/Documents/Caramel/caramelav/data/file_res.txt"


# town = "Town01_08_10_2020_19_27_31"

# Town01_07_10_2020_11_52_12
# Town01_08_10_2020_19_27_31
towns = ["Town01_08_10_2020_22_39_41", "Town02_07_10_2020_15_34_22", "Town02_09_10_2020_11_01_17", "Town03_07_10_2020_17_33_03", "Town04_07_10_2020_18_38_51", "Town05_07_10_2020_22_09_37", "Town06_08_10_2020_13_47_36", "Town07_08_10_2020_14_26_01", "Town10HD_08_10_2020_15_56_17"]

# path_to_predicted_images = "/media/andreas/lab/all_thesis_data/" + town + "/res/5/"
# path_to_predicted_processed_images = "/media/andreas/lab/all_thesis_data/" + town + "/evaluation/predicted_processed_images/"

# path_to_gt_images = "/media/andreas/lab/all_thesis_data/" + town + "/seg_carla/"
# path_to_gt_processed_images = "/media/andreas/lab/all_thesis_data/" + town + "/evaluation/gt_processed_images/"

def statistics(predictionImgList_path, groundTruthImgList_path, town):
    groundTruthImgList = os.listdir(groundTruthImgList_path)
    groundTruthImgList.sort()
    predictionImgList = os.listdir(predictionImgList_path)
    predictionImgList.sort()
    evaluateImgLists(predictionImgList, groundTruthImgList, town, "thesis_output")

def crop(image, image_pixel):
	x = 0
	if image_pixel % 2 == 1:
		image_pixel = image_pixel - 1

	w = image.shape[1]
	h = image.shape[0]-image_pixel

	image = image[image_pixel:image_pixel + h, x:x + w]
	return image

# ignore all classes expect from road
def ignore_classes(image):
	for h in range(0, image.shape[0]):
		for w in range(0, image.shape[1]):
			r = image[h, w, 0]
			g = image[h, w, 1]
			b = image[h, w, 2]

			# road
			if r == 128 and g == 64 and b == 128:
				pass
			else:		
				image[h, w, 0] = 0
				image[h, w, 1] = 0
				image[h, w, 2] = 0
	return image

def run():


	for t in towns:	
		print ("town = ", t)

		path_to_predicted_images = "/media/andreas/lab/all_thesis_data/" + t + "/res/5/"
		path_to_predicted_processed_images = "/media/andreas/lab/all_thesis_data/" + t + "/evaluation/predicted_processed_images/"

		path_to_gt_images = "/media/andreas/lab/all_thesis_data/" + t + "/seg_carla/"
		path_to_gt_processed_images = "/media/andreas/lab/all_thesis_data/" + t + "/evaluation/gt_processed_images/"

		# for predicted images => extract pixel only for classes (road, vehicle, walker)
		counter=0
		scale_percent = 50 # percent of original size
		predictedImgList = os.listdir(path_to_predicted_images)
		for im in predictedImgList:
			# if counter > 200:
			# 	break
			# read image
			image = cv2.imread(path_to_predicted_images + im)

			# resize image
			width = int(image.shape[1] * scale_percent / 100)
			height = int(image.shape[0] * scale_percent / 100)
			dim = (width, height)
			resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
			
			# keep only the road
			image_less = ignore_classes(resized)

			# write gt
			cv2.imwrite(path_to_predicted_processed_images + im, image_less)
			counter = counter+1

		# for gt images => extract pixel only for classes (road, vehicle, walker)
		counter=0
		groundTruthImgList = os.listdir(path_to_gt_images)
		for im in groundTruthImgList:
			if (path.exists(path_to_predicted_processed_images+im)):
				# if counter > 200:
					# break
				# read image
				image = cv2.imread(path_to_gt_images + im)

				# resize image
				width = int(image.shape[1] * scale_percent / 100)
				height = int(image.shape[0] * scale_percent / 100)
				dim = (width, height)
				resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

				# keep only the road
				image_less = ignore_classes(resized)

				# write new gt
				cv2.imwrite(path_to_gt_processed_images + im, image_less)
				counter = counter+1

		statistics(path_to_predicted_processed_images, path_to_gt_processed_images, t)

# ----------------------------------------------------
# add footpoints to exclude objects out of road
# ----------------------------------------------------

	# with open(path_to_file) as fp:
	# 	for line in fp:
	# 		line = line.replace("\n", "")
	# 		line = line.split(" ")
	# 		image_name = line[0]
	# 		image_pixel = int(line[1])

	# 		# read image
	# 		image = cv2.imread(path_to_predicted_images + image_name + ".png")

	# 		# crop image
	# 		image_cropped = crop(image, image_pixel)
	# 		image_cropped = ignore_classes(image_cropped, 0)

	# 		# write image
	# 		id = image_name.replace(".png", "")
	# 		# print (id)
	# 		# if (int(id) <= 9998):
	# 			# image_name = "00"+str(id)
	# 		# else:
	# 			# image_name = "0"+str(id)
	# 		cv2.imwrite(path_to_predicted_processed_images + image_name + ".png", image_cropped)

	# 		# fix gt name
	# 		# if (int(id) <= 9998):
	# 		# 	gt_name = "00"+str(id)
	# 		# else:
	# 		# 	gt_name = "0"+str(id)

	# 		# read gt
	# 		gt = cv2.imread(path_to_gt_images + image_name + ".png")

	# 		# crop gt
	# 		gt_cropped = crop(gt, image_pixel)
	# 		gt_cropped = ignore_classes(gt_cropped, 1)

	# 		# write gt
	# 		# cv2.imwrite(path_to_gt_cropped_images + "im_" + gt_name + ".png", gt_cropped)
	# 		cv2.imwrite(path_to_gt_processed_images + image_name + ".png", gt_cropped)

	# statistics(path_to_predicted_processed_images, path_to_gt_processed_images)

def main():
	run()

# call the main method
if __name__ == "__main__":
    main()




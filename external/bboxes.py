import re
import numpy as np
import json
from scipy.spatial.transform import Rotation as R

# !!!!!!!!!!!!!!!!!!!!!
# validate that length of frames in lidar_cam_metadata.txt and
# each json are equal
# !!!!!!!!!!!!!!!!!!!!!

def getEgoPos(path):
	# store ego positions
	path = '/media/andreas/lab/all_thesis_data/Town01_08_10_2020_19_27_31/lidar_cam_metadata.txt'
	ego_pos_x = []
	ego_pos_y = []
	ego_pos_z = []
	with open(path) as file:
		for line in file:
			# get ego x
			res = re.findall(r'x=[+-]?[0-9]+\.[0-9]+', line)
			lidar_x = res[1]
			ego_pos_x.append(lidar_x.replace("x=", ""))

			# get ego y
			res = re.findall(r'y=[+-]?[0-9]+\.[0-9]+', line)
			lidar_y = res[1]
			ego_pos_y.append(lidar_y.replace("y=", ""))

			# get ego z
			res = re.findall(r'z=[+-]?[0-9]+\.[0-9]+', line)
			lidar_z = res[1]
			ego_pos_z.append(lidar_z.replace("z=", ""))
	return ego_pos_x, ego_pos_y, ego_pos_z

# get bounding boxes of vehicles
def getVehicleBoxes(path):
	with open(path) as f:
		data = json.load(f)

		for key, value in data.items():
				binary_name = key.replace("frame_","")
				# print (binary_name)
				index = int(binary_name) - 1

				if (index >= len(ego_pos_x)):
					continue

				# iterate through each bbox
				for v in value:
					# compute center of bbox
					loc = v["location"]
					center = [float(loc[0]) - float(ego_pos_x[index]), 
								float(ego_pos_y[index]) - float(loc[1]),
								float(loc[2]) - float(ego_pos_z[index])]

					# rotate center by box yaw in z-axis
					rotation_degrees = float(v["rotation"][1])
					rotation_radians = np.radians(rotation_degrees)
					rotation_axis = np.array([0, 0, 1])
					rotation_vector = rotation_radians * rotation_axis
					rotation = R.from_rotvec(rotation_vector)
					rotated_vec = rotation.apply(center)

					# compute edges of box
					borders_x = []
					borders_y = []
					borders_z = []
					ex = [float(v["extent"][0]), float(v["extent"][1]), float(v["extent"][2])]
					borders_x.append(rotated_vec[0] - ex[0])
					borders_x.append(rotated_vec[0] - ex[0])
					borders_x.append(rotated_vec[0] - ex[0])
					borders_x.append(rotated_vec[0] - ex[0])
					borders_x.append(rotated_vec[0] + ex[0])
					borders_x.append(rotated_vec[0] + ex[0])
					borders_x.append(rotated_vec[0] + ex[0])
					borders_x.append(rotated_vec[0] + ex[0])

					borders_y.append(rotated_vec[1] - ex[1])
					borders_y.append(rotated_vec[1] + ex[1])
					borders_y.append(rotated_vec[1] + ex[1])
					borders_y.append(rotated_vec[1] - ex[1])
					borders_y.append(rotated_vec[1] - ex[1])
					borders_y.append(rotated_vec[1] + ex[1])
					borders_y.append(rotated_vec[1] + ex[1])
					borders_y.append(rotated_vec[1] - ex[1])

					borders_z.append(rotated_vec[2] + ex[2])
					borders_z.append(rotated_vec[2] + ex[2])
					borders_z.append(rotated_vec[2] - ex[2])
					borders_z.append(rotated_vec[2] - ex[2])
					borders_z.append(rotated_vec[2] + ex[2])
					borders_z.append(rotated_vec[2] + ex[2])
					borders_z.append(rotated_vec[2] - ex[2])
					borders_z.append(rotated_vec[2] - ex[2])

					# rotate back
					rotation_axis = np.array([0, 0, -1])
					rotation_vector = rotation_radians * rotation_axis
					rotation = R.from_rotvec(rotation_vector)
					final_borders = []
					for i in range(0, len(borders_x)):
						vec = [borders_x[i], borders_y[i], borders_z[i]]
						rotated_vec = rotation.apply(vec)
						final_borders.append(rotated_vec)

					# store all edges of each bbox of vehicles of current frame in binary in this form:
					# x y z
					# x y z
					# .....
					with open('/media/andreas/lab/all_thesis_data/Town01_08_10_2020_19_27_31/vehicles/'+str(binary_name)+'.bin', 'a') as file:
						for b in final_borders:
							num = [b[0], b[1], b[2]]
							num = np.array(num).astype(np.float32)
							# print (num)
							num.tofile(file)
						file.close()

# get bounding boxes of vehicles
def getWalkerBoxes(path):
	with open(path) as f:
		data = json.load(f)

		for key, value in data.items():
				binary_name = key.replace("frame_","")
				# print (binary_name)
				index = int(binary_name) - 1

				if (index >= len(ego_pos_x)):
					continue

				# iterate through each bbox
				for v in value:
					# compute center of bbox
					loc = v["location"]
					center = [float(loc[0]) - float(ego_pos_x[index]), 
								float(ego_pos_y[index]) - float(loc[1]),
								float(loc[2]) - float(ego_pos_z[index])]

					# rotate center by box yaw in z-axis
					rotation_degrees = float(v["rotation"][1])
					rotation_radians = np.radians(rotation_degrees)
					rotation_axis = np.array([0, 0, 1])
					rotation_vector = rotation_radians * rotation_axis
					rotation = R.from_rotvec(rotation_vector)
					rotated_vec = rotation.apply(center)

					# compute edges of box
					borders_x = []
					borders_y = []
					borders_z = []
					ex = [float(v["extent"][0]), float(v["extent"][1]), float(v["extent"][2])]
					borders_x.append(rotated_vec[0] - ex[0])
					borders_x.append(rotated_vec[0] - ex[0])
					borders_x.append(rotated_vec[0] - ex[0])
					borders_x.append(rotated_vec[0] - ex[0])
					borders_x.append(rotated_vec[0] + ex[0])
					borders_x.append(rotated_vec[0] + ex[0])
					borders_x.append(rotated_vec[0] + ex[0])
					borders_x.append(rotated_vec[0] + ex[0])

					borders_y.append(rotated_vec[1] - ex[1])
					borders_y.append(rotated_vec[1] + ex[1])
					borders_y.append(rotated_vec[1] + ex[1])
					borders_y.append(rotated_vec[1] - ex[1])
					borders_y.append(rotated_vec[1] - ex[1])
					borders_y.append(rotated_vec[1] + ex[1])
					borders_y.append(rotated_vec[1] + ex[1])
					borders_y.append(rotated_vec[1] - ex[1])

					borders_z.append(rotated_vec[2] + ex[2])
					borders_z.append(rotated_vec[2] + ex[2])
					borders_z.append(rotated_vec[2] - ex[2])
					borders_z.append(rotated_vec[2] - ex[2])
					borders_z.append(rotated_vec[2] + ex[2])
					borders_z.append(rotated_vec[2] + ex[2])
					borders_z.append(rotated_vec[2] - ex[2])
					borders_z.append(rotated_vec[2] - ex[2])

					# rotate back
					rotation_axis = np.array([0, 0, -1])
					rotation_vector = rotation_radians * rotation_axis
					rotation = R.from_rotvec(rotation_vector)
					final_borders = []
					for i in range(0, len(borders_x)):
						vec = [borders_x[i], borders_y[i], borders_z[i]]
						rotated_vec = rotation.apply(vec)
						final_borders.append(rotated_vec)

					# store all edges of each bbox of vehicles of current frame in binary in this form:
					# x y z
					# x y z
					# .....
					with open('/media/andreas/lab/all_thesis_data/Town01_08_10_2020_19_27_31/walkers/'+str(binary_name)+'.bin', 'a') as file:
						for b in final_borders:
							num = [b[0], b[1], b[2]]
							num = np.array(num).astype(np.float32)
							# print (num)
							num.tofile(file)
						file.close()


# Warning! All bounding boxes accessed through carla.World are described in world space. 
# On the contrary, the bounding box of a carla.Vehicle, carla.Walker or carla.Junction, 
# stores its location and rotation relative to the object it is attached to.
# so, the location of road is sescribed in the world coordinate space
# also the lidar sensor is located in (0,0,3) in world coordinate space, so we need to devide it from center point below
# get bounding boxes of roads
def getRoadBoxes(path):
	item=1
	with open(path) as f:
		data = json.load(f)

		for key, value in data.items():
				binary_name = key.replace("frame_","")
				# print (binary_name)
				index = int(binary_name) - 1

				if (index >= len(ego_pos_x)):
					continue

				# iterate through each bbox
				for v in value:
					# compute center of bbox
					loc = v["location"]

					lidar_sensor_height = 3
					center = [float(loc[0]) - float(ego_pos_x[index]), 
								float(ego_pos_y[index]) - float(loc[1]),
								float(loc[2]) - float(ego_pos_z[index]) + lidar_sensor_height]

								# float(loc[2]) - float(ego_pos_z[index]) - lidar_sensor_height]
								# float(loc[2]) - float(ego_pos_z[index]) + lidar_sensor_height]
								# float(loc[2])]

					# center = [float(ego_pos_x[index]), 
								# float(ego_pos_y[index]),
								# float(ego_pos_z[index])-lidar_sensor_height]

					# rotate center by box yaw in z-axis
					rotation_degrees = float(v["rotation"][1])
					rotation_radians = np.radians(rotation_degrees)
					rotation_axis = np.array([0, 0, 1])
					rotation_vector = rotation_radians * rotation_axis
					rotation = R.from_rotvec(rotation_vector)
					rotated_vec = rotation.apply(center)

					# compute edges of box
					borders_x = []
					borders_y = []
					borders_z = []
					ex = [float(v["extent"][0]), float(v["extent"][1]), float(v["extent"][2])]

					borders_x.append(rotated_vec[0] - ex[0])
					borders_x.append(rotated_vec[0] - ex[0])
					borders_x.append(rotated_vec[0] - ex[0])
					borders_x.append(rotated_vec[0] - ex[0])
					borders_x.append(rotated_vec[0] + ex[0])
					borders_x.append(rotated_vec[0] + ex[0])
					borders_x.append(rotated_vec[0] + ex[0])
					borders_x.append(rotated_vec[0] + ex[0])

					borders_y.append(rotated_vec[1] - ex[1])
					borders_y.append(rotated_vec[1] + ex[1])
					borders_y.append(rotated_vec[1] + ex[1])
					borders_y.append(rotated_vec[1] - ex[1])
					borders_y.append(rotated_vec[1] - ex[1])
					borders_y.append(rotated_vec[1] + ex[1])
					borders_y.append(rotated_vec[1] + ex[1])
					borders_y.append(rotated_vec[1] - ex[1])

					borders_z.append(rotated_vec[2] + ex[2])
					borders_z.append(rotated_vec[2] + ex[2])
					borders_z.append(rotated_vec[2] - ex[2])
					borders_z.append(rotated_vec[2] - ex[2])
					borders_z.append(rotated_vec[2] + ex[2])
					borders_z.append(rotated_vec[2] + ex[2])
					borders_z.append(rotated_vec[2] - ex[2])
					borders_z.append(rotated_vec[2] - ex[2])

					# rotate back
					rotation_axis = np.array([0, 0, -1])
					rotation_vector = rotation_radians * rotation_axis
					rotation = R.from_rotvec(rotation_vector)
					final_borders = []
					for i in range(0, len(borders_x)):
						vec = [borders_x[i], borders_y[i], borders_z[i]]
						rotated_vec = rotation.apply(vec)
						final_borders.append(rotated_vec)

					# store all edges of each bbox of vehicles of current frame in binary in this form:
					# x y z
					# x y z
					# .....
					with open('/media/andreas/lab/all_thesis_data/Town01_08_10_2020_19_27_31/roads/'+str(binary_name)+'.bin', 'a') as file:
						for b in final_borders:
							num = [b[0], b[1], b[2]]
							num = np.array(num).astype(np.float32)
							# print (num)
							num.tofile(file)
						file.close()

pathToFile = '/media/andreas/lab/all_thesis_data/Town01_08_10_2020_19_27_31/lidar_cam_metadata.txt'
ego_pos_x, ego_pos_y, ego_pos_z = getEgoPos(pathToFile)

path = '/media/andreas/lab/all_thesis_data/Town01_08_10_2020_19_27_31/vehicles/vehicles.json'
getVehicleBoxes(path)

path = '/media/andreas/lab/all_thesis_data/Town01_08_10_2020_19_27_31/walkers/walkers.json'
getWalkerBoxes(path)

path = '/media/andreas/lab/all_thesis_data/Town01_08_10_2020_19_27_31/roads/roads.json'
getRoadBoxes(path)



import os
import cv2


image_path = "/home/andreas/Desktop/dataset_thesis/Town02_14_09_2020_17_05_28/seg_carla/"
name = "068898"

image = cv2.imread(image_path + name + ".png")

# road : (128, 64, 128)
# sidewalk : (244, 35, 232)
# for each road pixel search its 4 neighbors and if at least one of them belongs to sidewalk => store this pixel

scale_percent = 50
im_width = int(image.shape[1] * scale_percent / 100)
im_height = int(image.shape[0] * scale_percent / 100)
image = cv2.resize(image, (im_width, im_height), interpolation=cv2.INTER_NEAREST)


w_pixels = []
h_pixels = []
# Sidewalk	(244, 35, 232)
for h in range(0, im_height):
    print (str(h) + " / " + str(im_height))
    for w in range(0, im_width):
        r = image[h, w, 2]
        g = image[h, w, 1]
        b = image[h, w, 0]
        if r == 128 and g == 64 and b == 128:
            # check up
            if (h-1) < 0:
                pass
            else:
                r2 = image[h-1, w, 2]
                g2 = image[h-1, w, 1]
                b2 = image[h-1, w, 0]
                if r2 == 244 and g2 == 35 and b2 == 232:
                    w_pixels.append(w)
                    h_pixels.append(h-1)

           # check down
            if (h+1) >= im_height:
                pass
            else:
                r2 = image[h+1, w, 2]
                g2 = image[h+1, w, 1]
                b2 = image[h+1, w, 0]
                if r2 == 244 and g2 == 35 and b2 == 232:
                    w_pixels.append(w)
                    h_pixels.append(h+1)

            # check left
            if (w-1) < 0:
                pass
            else:
                r2 = image[h, w-1, 2]
                g2 = image[h, w-1, 1]
                b2 = image[h, w-1, 0]
                if r2 == 244 and g2 == 35 and b2 == 232:
                    w_pixels.append(w-1)
                    h_pixels.append(h)

            # check right
            if (w+1) >= im_width:
                pass
            else:
                r2 = image[h, w+1, 2]
                g2 = image[h, w+1, 1]
                b2 = image[h, w+1, 0]
                if r2 == 244 and g2 == 35 and b2 == 232:
                    w_pixels.append(w+1)
                    h_pixels.append(h)


print ("drawing")

# draw curbs in image
counter = 0
for counter in range(0, len(h_pixels)):
    print (str(counter) + " / " + str(len(h_pixels)))
    print (counter)
    print (h_pixels[counter])
    print (w_pixels[counter])
    image[h_pixels[counter], w_pixels[counter], 0] = 255
    image[h_pixels[counter], w_pixels[counter], 1] = 255
    image[h_pixels[counter], w_pixels[counter], 2] = 255

name = name + "_curb.png"
cv2.imwrite(image_path + name, image)

print ("finished")

















import os
import json
import cv2
import denoise
import torch
import numpy as np
import torch.nn as nn
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from torch.autograd import Variable
import torchattacks

from advertorch.attacks import LinfPGDAttack, MomentumIterativeAttack
from denoise.train import super_resolution
import sys

root = '/media/beastv2/b776d2f6-4b47-4c15-8055-e488de9e9288/deeplabv3/model'
sys.path.append(root)
from deeplabv3 import DeepLabV3

sys.path.append('./segmentation_model/')
from robustification.techniques import (slq, jpeg_compress, median_filter, denoise_tv_bregman)
from evalPixelLevelSemanticLabeling_old import evaluateImgLists
from AdaFM.codes.interpolate import Adam_smoth

config2 = ConfigProto()
config2.gpu_options.allow_growth = True
session = InteractiveSession(config=config2)

with open('config.json') as f:
    config = json.load(f)


# ----------------------------------------------------------------------------------------------------------------------
# SHAPE OF INPUT DATA TO EACH MODULE
# ----------------------------------------------------------------------------------------------------------------------
# INPUT for segmentation prediction, prediction(x) =>
#                           x as normalized tensor with shape = batch_size x channels x height x width

# INPUT for slq, slq(x) => x as denormalized numpy with shape = height x width x channels
# INPUT for jpeg_compress, jpeg_compress(x) => x as denormalized numpy with shape = height x width x channels
# INPUT for denoise_tv_bregman, denoise_tv_bregman(x) => x as denormalized numpy with shape = height x width x channels
# INPUT for median_filter, median_filter(x) => x as denormalized numpy with shape = height x width x channels

# INPUT for attack, attack(image, prediction, labels) =>
#                           image as normalized tensor with shape = batch_size x rgb x height x width,
#                           prediction as normalized tensor with shape = batch_size x classes x height x width,
#                           labels as denormalized tensor with shape = height x width

# INPUT for denoise, denoise(x) => x as normalized numpy with shape = batch_size x height x width x channels
# ----------------------------------------------------------------------------------------------------------------------

class Resultimages:
    def __init__(self):
        self.image = ""
        self.gt = ""
        self.clean_pred = ""
        self.attacked = ""
        self.noise = ""
        self.attacked_denormalized = ""
        self.attacked_pred = ""
        self.denoised = ""
        self.denoised_pred = ""


class Pipeline:
    def __init__(self, config):
        self.denoise_model_path = config["denoise"]['path']
        self.image_rgb_path = config['image']['rgb_path']
        self.image_gt_path = config['image']['gt_path']

        # load test and ground truth images
        self.images = os.listdir(self.image_rgb_path)
        self.images.sort()
        self.gt = os.listdir(self.image_gt_path)
        self.gt.sort()

        if config["denoise"]["perform"]:
            # load super resolution network
            self.denoise_model = denoise.get_G([1, None, None, 3])
            self.denoise_model.load_weights(os.path.join(self.denoise_model_path, 'g.h5'))
            self.denoise_model.eval()

        # load original segmentation network
        root = '/media/beastv2/b776d2f6-4b47-4c15-8055-e488de9e9288/deeplabv3'
        network = DeepLabV3("eval_val", project_dir=root).cuda()
        checkpoint = torch.load(config["segmentation"]["path"])
        network.load_state_dict(checkpoint['state_dict'])
        self.segmentation_model_original = {"model": network, "loss": torch.nn.CrossEntropyLoss()}
        self.segmentation_model_original["model"].eval()

        checkpoint = torch.load(config["compress"]["path"])
        network.load_state_dict(checkpoint['state_dict'])
        self.segmentation_model_compressed = {"model": network, "loss": torch.nn.CrossEntropyLoss()}
        self.segmentation_model_compressed["model"].eval()

        self.result = Resultimages()

        # create folders to store the results
        if config["simple_run"] == 1:
            # if os.path.exists(str(config["save_data"])):
            # shutil.rmtree(str(config["save_data"]))

            # if folder does not exist
            if os.path.isdir(config["save_data"]) == False:
                os.mkdir(str(config["save_data"]))
                os.mkdir(str(config["save_data"] + "/rgb"))
                os.mkdir(str(config["save_data"] + "/rgb/all_images"))
                os.mkdir(str(config["save_data"] + "/rgb/original"))
                os.mkdir(str(config["save_data"] + "/rgb/attacked"))
                os.mkdir(str(config["save_data"] + "/rgb/compressed_all_levels"))
                for coef in np.arange(0.0, config['denoise']['uper_coef'], config['denoise']['step_coef']):
                    os.mkdir(str(config["save_data"] + "/rgb/denoised" + '{:.2f}'.format(coef)))

                os.mkdir(str(config["save_data"] + "/segmented"))
                os.mkdir(str(config["save_data"] + "/segmented/all_images"))
                os.mkdir(str(config["save_data"] + "/segmented/gt"))
                os.mkdir(str(config["save_data"] + "/segmented/clean_pred"))
                os.mkdir(str(config["save_data"] + "/segmented/attacked_pred"))
                os.mkdir(str(config["save_data"] + "/segmented/compressed_pred_all_levels"))

                for coef in np.arange(0.0, config['denoise']['uper_coef'], config['denoise']['step_coef']):
                    os.mkdir(str(config["save_data"] + "/segmented/denoised_pred" + '{:.2f}'.format(coef)))

                os.mkdir(str(config["save_data"] + "/segmented_raw"))
                os.mkdir(str(config["save_data"] + "/segmented_raw/all_images"))
                os.mkdir(str(config["save_data"] + "/segmented_raw/gt"))
                os.mkdir(str(config["save_data"] + "/segmented_raw/clean_pred"))
                os.mkdir(str(config["save_data"] + "/segmented_raw/attacked_pred"))
                os.mkdir(str(config["save_data"] + "/segmented_raw/compressed_pred_all_levels"))
                for coef in np.arange(0.0, config['denoise']['uper_coef'], config['denoise']['step_coef']):
                    os.mkdir(str(config["save_data"] + "/segmented_raw/denoised_pred" + '{:.2f}'.format(coef)))

    # run the whole pipeline
    def run(self):
        # get attack function
        adversary_attack = self.get_attack_func()

        # process each image
        for count, name in enumerate(self.images):
            print(str(count + 1) + '/' + str(len(self.images)))

            # set original and test image
            self.result.image = cv2.imread(self.image_rgb_path + name)
            self.result.gt = cv2.imread(self.image_gt_path + name)
            self.result.gt = self.result.gt[:, :, 2]  # BGR image, we want the RED channel

            # scale images if needed
            if config['image']['scale']:
                self.result.image = self.scale_image(self.result.image)
                self.result.gt = self.scale_image(self.result.gt)

            if config['image']['crop']:
                # y = int(self.result.image.shape[0] - (3 / 4) * (self.result.image.shape[0]))
                # x = int(self.result.image.shape[1] / 2 - self.result.image.shape[1] / 4)
                y = int(self.result.image.shape[0]*config['image']['crop_factor'])
                if y % 2 ==1:
                    y = y - 1
                x = 0
                # w = config["image"]["crop_width"]
                # h = config["image"]["crop_height"]

                w = self.result.image.shape[1]
                h = self.result.image.shape[0]-y

                self.result.image = self.result.image[y:y + h, x:x + w]
                self.result.gt = self.result.gt[y:y + h, x:x + w]

            image = np.transpose(self.result.image, (2, 0, 1))  # (shape: (3, 512, 1024))
            image = torch.unsqueeze(torch.from_numpy(image) / 255.0, 0).cuda()

            labels = torch.unsqueeze(torch.from_numpy(self.result.gt).type(torch.FloatTensor), 0)
            labels = Variable(labels.type(torch.LongTensor)).cuda()

            if config["attack"]["type"] == 'LinfPGDAttack' or config["attack"]["type"] == 'MomentumIterativeAttack':
                if config["attack"]["targeted"] == 'True':
                    target = torch.ones_like(labels) * config["attack"]["targeted_class"]
                else:
                    target = labels
                adversarial_images = adversary_attack.perturb(image, target)
            else:
                adversarial_images = adversary_attack(image.type(torch.FloatTensor), labels)

            adversarial_images = torch.squeeze(adversarial_images, 0)
            adversarial_images = adversarial_images.data.cpu().numpy()
            adversarial_images = np.transpose(adversarial_images, (1, 2, 0))
            adversarial_images *= 255
            adversarial_images = adversarial_images.astype(np.uint8).clip(0, 255)

            cv2.imwrite(config["save_data"] + '/rgb/original/' + name, self.result.image)
            cv2.imwrite(config["save_data"] + '/rgb/attacked/' + name, adversarial_images)

            seg_image = self.segment_image(self.result.image, 1, self.segmentation_model_original)
            seg_attac = self.segment_image(adversarial_images, 1, self.segmentation_model_original)
            seg_attac, seg_attac_raw = self.prediction_to_colour(seg_attac)
            seg_image, seg_image_raw = self.prediction_to_colour(seg_image)

            cv2.imwrite(config["save_data"] + '/segmented_raw/clean_pred/' + name, seg_image_raw)
            cv2.imwrite(config["save_data"] + '/segmented_raw/attacked_pred/' + name, seg_attac_raw)
            cv2.imwrite(config["save_data"] + '/segmented_raw/gt/' + name, self.result.gt)

            cv2.imwrite(config["save_data"] + '/segmented/clean_pred/' + name, seg_image)
            cv2.imwrite(config["save_data"] + '/segmented/attacked_pred/' + name, seg_attac)
            cv2.imwrite(config["save_data"] + '/segmented/gt/' + name, self.label_img_to_color(self.result.gt))

            if (count >= config["image"]["number_image"] - 1):
                return

    def super_resolution_denoise(self, sigma):
        super_resolution(config["save_data"] + '/rgb/attacked/', config["save_data"] + '/rgb/denoised/', sigma,
                         config["denoise"]["wavelet"])

    def just_segment(self, path_load, path_save, path_raw_save, seg_model):
        image_list = os.listdir(path_load)
        for name in image_list:
            image = cv2.imread(path_load + name)
            res = self.segment_image(image, 1, seg_model)
            res = res.data.cpu().numpy()  # (shape: (batch_size, num_classes, img_h, img_w))
            pred_label_imgs = np.argmax(res, axis=1)  # (shape: (batch_size, img_h, img_w))
            pred_label_imgs = pred_label_imgs[0, :, :]
            pred_label_imgs = pred_label_imgs.astype(np.uint8)

            cv2.imwrite(path_raw_save + '/' + name, pred_label_imgs)
            pred_label_img_color = self.label_img_to_color(pred_label_imgs)
            cv2.imwrite(path_save + '/' + name, pred_label_img_color)

    def segment_image(self, image, need_normalize, seg_model):
        if need_normalize:
            image = self.normalize(image)
            # convert numpy -> torch:
            image = torch.from_numpy(image)
            image = torch.unsqueeze(image, 0)
            image = image.clone().detach().requires_grad_(True).cuda()  # reguires_grad(False) ???

        res = seg_model["model"](image)

        return res

    # return labels for the selected attack
    def get_labels(self, tensor_pr):
        if config["attack"]["type"] == "untargeted":
            label_img = torch.from_numpy(self.result.gt).type(torch.FloatTensor)

        elif config["attack"]["type"] == "rician_image":
            label_img = torch.from_numpy(self.result.gt).type(torch.FloatTensor)

        if config["attack"]["type"] == "iterated":
            label_img = tensor_pr.clone()
            # replace the class of vehicles to the class of road
            label_img[:, 10, :, :] = label_img[:, 7, :, :]

        label_img = label_img.clone().detach().requires_grad_(True).cuda()
        return label_img

    # if numpy shape in = height x width x channels =>
    #    numpy shape out = channels x height x width
    def normalize(self, img):
        img = img / 255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img / np.array([0.229, 0.224, 0.225])
        img = np.transpose(img, (2, 0, 1))
        res = img.astype(np.float32)
        return res

    # if tensor shape in = batch_size x channels x height x width =>
    #    numpy shape out = height x width x channels
    def denormalise(self, img):
        img = torch.squeeze(img, 0)
        img = img.data.cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img *= (0.229, 0.224, 0.225)
        img += (0.485, 0.456, 0.406)
        img *= 255.000
        res = img.astype(np.uint8).clip(0, 255)
        return res

    def save_data(self, name):
        cv2.imwrite(config["save_data"] + '/rgb/original/' + name, self.result.image)
        cv2.imwrite(config["save_data"] + '/rgb/attacked/' + name, self.result.attacked_denormalized)
        cv2.imwrite(config["save_data"] + '/rgb/denoised/' + name, self.result.denoised)

        # torch.cuda.synchronize()

        clean_pred, clean_pred_raw = self.prediction_to_colour(self.result.clean_pred)
        attacked_pred, attacked_pred_raw = self.prediction_to_colour(self.result.attacked_pred)
        denoised_pred, denoised_pred_raw = self.prediction_to_colour(self.result.denoised_pred)
        gt = self.label_img_to_color(self.result.gt)

        cv2.imwrite(config["save_data"] + '/segmented/gt/' + name, gt)
        cv2.imwrite(config["save_data"] + '/segmented/clean_pred/' + name, clean_pred)
        cv2.imwrite(config["save_data"] + '/segmented/attacked_pred/' + name, attacked_pred)
        cv2.imwrite(config["save_data"] + '/segmented/denoised_pred/' + name, denoised_pred)

        cv2.imwrite(config["save_data"] + '/segmented_raw/gt/' + name, self.result.gt)
        cv2.imwrite(config["save_data"] + '/segmented_raw/clean_pred/' + name, clean_pred_raw)
        cv2.imwrite(config["save_data"] + '/segmented_raw/attacked_pred/' + name, attacked_pred_raw)
        cv2.imwrite(config["save_data"] + '/segmented_raw/denoised_pred/' + name, denoised_pred_raw)

    def prediction_to_colour(self, pred):
        pred_raw = torch.argmax(pred[0], 0).cpu().detach().numpy()
        pred = self.label_img_to_color(pred_raw)
        pred = np.array(pred).astype(np.uint8)
        pred_raw = np.array(pred_raw).astype(np.uint8)
        return (pred, pred_raw)

    def label_img_to_color(self, img):
        label_to_color = {
            0: [0, 0, 0],  # None
            1: [70, 70, 70],  # Buildings
            2: [190, 153, 153],  # Fences
            3: [72, 0, 90],  # Other
            4: [220, 20, 60],  # Pedestrians
            5: [153, 153, 153],  # Poles
            6: [157, 234, 50],  # RoadLines
            7: [128, 64, 128],  # Roads
            8: [244, 35, 232],  # Sidewalks
            9: [107, 142, 35],  # Vegetation
            10: [0, 0, 255],  # Vehicles
            11: [102, 102, 156],  # Walls
            12: [220, 220, 0]  # TrafficSigns
        }
        img_height, img_width = img.shape

        img_color = np.zeros((img_height, img_width, 3))
        for row in range(img_height):
            for col in range(img_width):
                label = img[row, col]
                img_color[row, col] = np.array(label_to_color[label])

        return img_color

    def denoise(self, image):
        image = denoise.wavelet_denoise(image, 0.01)
        image = (image / 127.5) - 1  # rescale to ［－1, 1]
        image = np.asarray(image, dtype=np.float32)
        image = image[np.newaxis, :, :, :]

        out = self.denoise_model(image).numpy()
        res = out[0]
        res += 1
        res /= 2
        res *= 255
        return res
        # tl.vis.save_image(out[0], path_save + image_name[count])

    def scale_image(self, image):
        scale_percent = config["image"]["scale_percent"]
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
        return image

    def get_attack_func(self):
        if config["attack"]["type"] == 'LinfPGDAttack':
            adversary_attack = LinfPGDAttack(
                self.segmentation_model_original["model"], loss_fn=nn.CrossEntropyLoss(),
                eps=config["attack"]["eps"] / 255, nb_iter=config["attack"]["iterations"],
                eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
                targeted=config["attack"]["targeted"])
        elif config["attack"]["type"] == 'MomentumIterativeAttack':
            adversary_attack = MomentumIterativeAttack(
                self.segmentation_model_original["model"], loss_fn=None, eps=config["attack"]["eps"] / 255,
                nb_iter=config["attack"]["iterations"], decay_factor=1., eps_iter=0.01, clip_min=0.,
                clip_max=1., targeted=config["attack"]["targeted"], ord=np.inf)
        elif config["attack"]["type"] == 'PGD_attack':
            adversary_attack = torchattacks.PGD(self.segmentation_model_original["model"],
                                                eps=config["attack"]["eps"] / 255,
                                                alpha=config["attack"]["alpha"])
        elif config["attack"]["type"] == 'FGSM_attack':
            adversary_attack = torchattacks.FGSM(self.segmentation_model_original["model"],
                                                 eps=config["attack"]["eps"] / 255)
        elif config["attack"]["type"] == 'BIM_attack':
            adversary_attack = torchattacks.BIM(self.segmentation_model_original["model"],
                                                eps=config["attack"]["eps"] / 255,
                                                alpha=config["attack"]["alpha"] / 255,
                                                steps=config["attack"]["iterations"])
        elif config["attack"]["type"] == 'DeepFool_attack':
            adversary_attack = torchattacks.DeepFool(self.segmentation_model_original["model"],
                                                     steps=config["attack"]["steps"])
        elif config["attack"]["type"] == 'FFGSM_attack':
            adversary_attack = torchattacks.FFGSM(self.segmentation_model_original["model"],
                                                  eps=config["attack"]["eps"] / 255,
                                                  alpha=config["attack"]["alpha"])
        elif config["attack"]["type"] == 'APGD_attack':
            adversary_attack = torchattacks.APGD(self.segmentation_model_original["model"],
                                                 eps=config["attack"]["eps"] / 255,
                                                 alpha=config["attack"]["alpha"] / 255,
                                                 steps=config["attack"]["iterations"],
                                                 sampling=config["attack"]["sampling"])
        return adversary_attack


def statistics(predictionImgList_path, groundTruthImgList_path, type, coef):
    groundTruthImgList = os.listdir(groundTruthImgList_path)
    groundTruthImgList.sort()
    predictionImgList = os.listdir(predictionImgList_path)
    predictionImgList.sort()

    prediction_path = ""
    if type == '/attacked':
        prediction_path = config["save_data"]+config['prediction_attack']
    elif type == '/clean':
        prediction_path = config['save_data']+config['prediction_clean']
    # elif type == '/compressed_all_levels':
    elif type.find("/compressed_all_levels") != -1:
        prediction_path = config['save_data']+config['prediction_compressed_all_levels']
    elif type == '/denoised_pred'+ '{:.2f}'.format(coef):
        prediction_path = config["save_data"]+config['prediction_denoised'] + '{:.2f}'.format(coef) + '/'

    evaluateImgLists(predictionImgList, groundTruthImgList, prediction_path, type)


def main():
    pipeline = Pipeline(config)

    # segment attacked and original images
    if config["simple_run"] == 1:
        pipeline.run()

        # statistic with clean image
        statistics(config['save_data'] + '/segmented_raw/clean_pred', config['save_data'] + '/segmented_raw/gt',
                   '/clean', 0)
        # statistic with attacked image
        statistics(config['save_data'] + '/segmented_raw/attacked_pred', config['save_data'] + '/segmented_raw/gt',
                   '/attacked', 0)

    # denoise
    if config["only_denoise"] == 1:
        # pipeline.super_resolution_denoise(config["denoise"]["sigma"])

        for coef in np.arange(0.0, config['denoise']['uper_coef'], config['denoise']['step_coef']):
            path = config["save_data"] + "/rgb/denoised" + '{:.2f}'.format(coef)

            Adam_smoth(coef, path)

            path_save = config["save_data"] + '/segmented/denoised_pred' + '{:.2f}'.format(coef)
            path_raw_save = config["save_data"] + '/segmented_raw/denoised_pred' + '{:.2f}'.format(coef)
            pipeline.just_segment(path + "/", path_save, path_raw_save, pipeline.segmentation_model_original)

            # statistic with clean image
            statistics(config['save_data'] + '/segmented_raw/denoised_pred' + '{:.2f}'.format(coef),
                       config['save_data'] + '/segmented_raw/gt',
                       '/denoised_pred' + '{:.2f}'.format(coef), coef)

    # compression
    if config["only_compression"] == 1:
        path_to_attacked_rgb = config["save_data"] + "/rgb/attacked"
        attacked_images = os.listdir(path_to_attacked_rgb)
        attacked_images.sort()

        path_rgb_save = config["save_data"] + '/rgb/compressed_all_levels'
        path_seg_save = config["save_data"] + '/segmented/compressed_pred_all_levels'
        path_seg_raw_save = config["save_data"] + '/segmented_raw/compressed_pred_all_levels'

        for count, name in enumerate(attacked_images):
            # print(str(count + 1) + '/' + str(len(attacked_images)))

            im = cv2.imread(path_to_attacked_rgb + "/" + name)

            # apply one preprocessed technique to the attacked image if robust segmentation model is selected
            defense_type = config["compress"]["defense"]
            defense_params = config["compress"]["defense_param"]
            processed_image = ""
            processed_image = {
                "slq": lambda processed_image:
                slq(im, defense_params["slq"]["qualities"], defense_params["slq"]["patch_size"]),
                "jpeg_compress": lambda processed_image:
                jpeg_compress(im, defense_params["jpeg_compress"]["quality"]),
                "denoise_tv_bregman": lambda processed_image:
                denoise_tv_bregman(im, defense_params["denoise_tv_bregman"]["weight"]),
                "median_filter": lambda processed_image:
                median_filter(im, defense_params["median_filter"]["size"])
            }[defense_type](processed_image)
            # store image
            cv2.imwrite(path_rgb_save + '/' + name, processed_image)

        # segment images
        pipeline.just_segment(path_rgb_save + "/", path_seg_save, path_seg_raw_save,
                              pipeline.segmentation_model_compressed)
        # get statistics

        if config["compress"]["defense"] == "slq":
            statistics(config['save_data'] + '/segmented_raw/compressed_pred_all_levels',
                       config['save_data'] + '/segmented_raw/gt',
                       '/compressed_all_levels_patch_'+str(config["compress"]["defense_param"]["slq"]["patch_size"]), 0)
        else:
            statistics(config['save_data'] + '/segmented_raw/compressed_pred_all_levels',
               config['save_data'] + '/segmented_raw/gt',
               '/compressed_all_levels', 0)

if __name__ == "__main__":
    main()

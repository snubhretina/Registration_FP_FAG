import numpy as np
import torch
# from sklearn.metrics import auc, precision_recall_curve, roc_curve
from torch.autograd import Variable
from torchvision import models
from PIL import Image

from registration_code.keypoint_based_registration import regist_SIFT
from registration_code.deformable_registration import regist_BSpline
from registration_code.chamfer_matching import chamfer_matching

from DL_model import FA
from DL_model import FA_SSANet3
from DL_model import fundus

import skimage
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2

def image_load(img_path_list):
    image_list = []
    for img_path in img_path_list:
        image_list.append(np.array(Image.open(img_path).convert('RGB')))
    return image_list
        
def make_FOV(self, imgs):
    FOV_mask = np.bitwise_or(imgs[:, :, 0] >= 10,
                             np.bitwise_or(imgs[:, :, 1] >= 10,
                                           imgs[:, :, 2] >= 10)).astype(np.ubyte) * 255
    return FOV_mask

class FAG_registration(object):
    def __init__(self, DL_model, img_seq, mask, img_size = [1024, 1536]):
        self.orig_frms = img_seq
        self.orig_frms_mask = []
        self.orig_vessel_seg_frms = []

        self.rigid_reg_frms = []
        self.rigid_reg_masks = []
        self.rigid_vessel_seg_frms = []

        self.nonrigid_reg_frms = []
        self.nonrigid_reg_masks = []
        self.nonrigid_vessel_seg_frms = []

        self.continued_max = np.zeros(img_size)
        self.continued_avg = np.zeros(img_size)

        self.DL_model = DL_model
        self.img_size = img_size
        self.main_FOV = mask

    #extraction vessel map from FAG img
    def FAG_vessel_prediction(self, model, FAG_img, mask, size):
        model.eval()
        with torch.no_grad():
            resized_FAG_img = np.array(Image.fromarray(FAG_img).resize(size=(size[1], size[0]), resample=Image.BILINEAR), dtype=np.float32)
            mask = np.around(np.array(Image.fromarray(mask).resize(size=(size[1], size[0]), resample=Image.BILINEAR), dtype=np.float32)/255).astype(np.ubyte)*255
            if len(resized_FAG_img.shape) == 2:
                resized_FAG_img = np.array(Image.fromarray(resized_FAG_img).convert('RGB'))
    
            mean = np.mean(resized_FAG_img[mask!=0])
            std = np.std(resized_FAG_img[mask != 0])
            img = (resized_FAG_img-mean)
    
            data = torch.from_numpy(img).permute(2,0,1)
            data = torch.unsqueeze(data, 0)
            FAG_data = Variable(data.cuda(0))
            out_FAG = model(FAG_data)
            pred_FAG = out_FAG.data.cpu()[0][0].numpy()
    
        pred_FAG = np.array(Image.fromarray((pred_FAG*255).astype(np.ubyte)).resize([FAG_img.shape[1], FAG_img.shape[0]], Image.BILINEAR))
        pred_FAG = pred_FAG.astype(np.float32)/255.
        return pred_FAG

    #extraction vessel map using CNN for trained FAG
    def Frms_vessel_extraction(self):
        for image in self.orig_frms:
            vessel = self.FAG_vessel_prediction(self.DL_model, image, self.main_FOV, size = self.img_size)
            self.orig_vessel_seg_frms.append(vessel)

    #registration image using SURF
    def rigid_registration(self):
        self.rigid_reg_frms.append(self.orig_frms[0])
        self.rigid_reg_masks.append(self.main_FOV)
        self.rigid_vessel_seg_frms.append(self.orig_vessel_seg_frms[0])
        for idx in range(1, self.orig_frms.__len__()):
            cur_img = self.orig_frms[idx]
            cur_FOV = self.make_FOV(cur_img)
            cur_vessel = self.orig_vessel_seg_frms[idx]
            sift = regist_SIFT(cur_img, self.rigid_reg_frms[-1], cur_FOV, self.rigid_reg_masks[-1])
            bFail, [regi_img1, h, sift_kpt_imgs, matching_img, kpt, des, pts, new_matching_img, new_matching_img_row] = sift.do_registration()
            if bFail == False:
                continue
            self.rigid_reg_frms.append(regi_img1)
            cur_reg_mask = cv2.warpPerspective(cur_FOV, h, (cur_FOV.shape[1], cur_FOV.shape[0]))
            cur_reg_vessel_mask = cv2.warpPerspective(cur_vessel, h, (cur_FOV.shape[1], cur_FOV.shape[0]))
            self.rigid_reg_masks.append(cur_reg_mask)
            self.rigid_vessel_seg_frms.append(cur_reg_vessel_mask)

    #registration iamge using b-spline
    def non_rigid_registration(self):
        self.nonrigid_reg_frms.append(self.rigid_reg_frms[0])
        self.nonrigid_vessel_seg_frms.append(self.rigid_vessel_seg_frms[0])
        self.nonrigid_reg_masks.append(self.rigid_reg_masks[0])


        for idx in range(1, self.orig_frms.__len__()):
            # pixel-wise maximum vessel probaility map.
            self.continued_max = np.concatenate([self.continued_max.reshape([1, self.continued_max.shape[0], self.continued_max.shape[1]])
                                                    , self.nonrigid_vessel_seg_frms[-1].reshape([1, self.continued_max.shape[0], self.continued_max.shape[1]])], 0)
            self.continued_max = np.max(self.continued_max, 0).astype(np.float32)

            # pixel-wise average vessel probaility map.
            self.continued_avg = np.concatenate(
                [self.continued_avg.reshape([1, self.continued_avg.shape[0], self.continued_avg.shape[1]])
                    ,  self.nonrigid_vessel_seg_frms[-1].reshape([1, self.continued_avg.shape[0], self.continued_avg.shape[1]])], 0)
            self.continued_avg = np.mean(self.continued_avg, 0).astype(np.float32)


            cur_img = self.rigid_reg_frms[idx]
            cur_mask = self.rigid_reg_masks[idx]
            cur_vessel = self.rigid_vessel_seg_frms[idx]

            bsp = regist_BSpline(self.continued_max, cur_vessel*255.)
            bsp_vessel_map = bsp.do_registration()
            self.nonrigid_reg_frms.append(bsp.registrationFromMatrix(cur_img[:,:,0].astype(np.float32)))
            self.nonrigid_reg_masks.append(bsp.registrationFromMatrix(cur_mask.astype(np.float32)))
            self.nonrigid_vessel_seg_frms.append(bsp_vessel_map)


    #aggregated image
    def aggregated_frames(self):
        vessel_seg_frms = np.array(self.nonrigid_vessel_seg_frms)/255.
        # aggregated FA by pixwel-wise average image and maximum image.
        avg_FAG = np.average(vessel_seg_frms, 0)
        max_FAG = np.max(vessel_seg_frms, 0)
        return avg_FAG, max_FAG


    #forward FAG registration series
    def all_registration(self):
        self.Frms_vessel_extraction()
        self.rigid_registration()
        self.non_rigid_registration()
        avg_FAG, max_FAG = self.aggregated_frames()
        return avg_FAG, max_FAG


class FP_registration(object):
    def __init__(self, DL_model, FP_img, FAagg_vessel_mean_map, FAagg_vessel_max_map, mask, img_size = [1024, 1536]):
        self.FP_img = FP_img
        self.FP_vessel_map = 0
        self.FAagg_vessel_mean_map = FAagg_vessel_mean_map
        self.FAagg_vessel_max_map = FAagg_vessel_max_map
        self.rigid_FAG_map = 0
        self.img_size = img_size
        self.DL_model = DL_model
        self.main_FOV = mask
        self.bsp = 0

    #extraction vessel map from FP img
    def FP_vessel_prediction(self, model, FP_img, mask, size):
        model.eval()
        with torch.no_grad():
            img = np.array(Image.fromarray(FP_img).resize((size[1], size[0]), Image.BILINEAR), dtype=np.float32)
            mean = [np.mean(img[mask[:, :] != 0, 0]), np.mean(img[mask[:, :] != 0, 1]), np.mean(img[mask[:, :] != 0, 2])]
            img[:, :, 0] = (img[:, :, 0] - mean[0])
            img[:, :, 1] = (img[:, :, 1] - mean[1])
            img[:, :, 2] = (img[:, :, 2] - mean[2])

            data = torch.from_numpy(img).permute(2,0,1).float()
            data = torch.unsqueeze(data,0)
            fundus_data = Variable(data.cuda(1))
            out_fundus = model(fundus_data)
            pred_fundus = out_fundus.data.cpu()[0][0].numpy()

        pred_fundus = np.array(
            Image.fromarray((pred_fundus * 255).astype(np.ubyte)).resize([FP_img.shape[1], FP_img.shape[0]], Image.BILINEAR))
        pred_fundus = pred_fundus.astype(np.float32) / 255.

        return pred_fundus
    
    #extraction vessel map using CNN for trained FP
    def vessel_extraction(self):
        self.FP_vessel_map = self.FP_vessel_prediction(self.DL_model, self.FP_img, self.main_FOV, self.img_size)

    #registration image using chamfer matching
    def chamfer_matching(self):
        # Chamfer Matcging.
        translated_FAG, t, angle = chamfer_matching(self.FAagg_vessel_max_map, self.FP_vessel_map)

        # apply Chamfer matching result.
        translation_matrix = np.float32([[1, 0, t[1]], [0, 1, t[0]]])
        img_translation = cv2.warpAffine(self.FAagg_vessel_mean_map, translation_matrix, (translated_FAG.shape[1], translated_FAG.shape[0]))
        self.rigid_FAG_map = skimage.transform.rotate(img_translation, angle, resize=False)


    #registration iamge using b-spline
    def non_rigid_registration(self):
        # defomable registration for FP-FA with BSpline.
        self.bsp  = regist_BSpline(self.FP_vessel_map * 255, self.rigid_FAG_map * 255.)
        bsp_fpfag_img = self.bsp.do_registration()

        # apply registration.
        registrated_avg_FAG = self.bsp.registrationFromMatrix(self.FAagg_vessel_mean_map*255)
        return bsp_fpfag_img

    def bsp_frommatrix(self, img_list):
        for i in range(img_list):
            img_list[i] = self.bsp.registrationFromMatrix(img_list[i]*255)
        return img_list


    def all_registration(self):
        self.vessel_extraction()
        self.chamfer_matching()
        bsp_fpfag_img = self.non_rigid_registration()
        return bsp_fpfag_img


def get_model(FP_model_weight_path, FA_model_weight_path):
    pretraind_model1 = models.resnet34(pretrained=True)
    FA_model = FA.SSA(pretraind_model1)
    FA_model = FA_model.cuda('cuda:0')
    FA_model.load_state_dict(torch.load(FA_model_weight_path, map_location='cuda:0'))
    

    pretraind_model2 = models.resnet34(pretrained=True)
    fundus_model = fundus.SSA(pretraind_model2)
    fundus_model = fundus_model.cuda('cuda:1')
    fundus_model.load_state_dict(torch.load(FP_model_weight_path, map_location='cuda:1'))

    return FA_model, fundus_model



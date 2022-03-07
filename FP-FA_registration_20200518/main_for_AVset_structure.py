from __future__ import print_function

import os
import torch
import utils
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
from _registration_code.keypoint_based_registration import regist_SIFT
from registration_code.deformable_registration import regist_BSpline
from registration_code.chamfer_matching import chamfer_matching
import skimage
from skimage import morphology
import cv2
import csv
import warnings
from skimage.filters import frangi
from datetime import datetime
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
warnings.filterwarnings("ignore")

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class SSANet3(nn.Module):
    def __init__(self, pretraind_model):
        super(SSANet3, self).__init__()
        self.scale_factor = 2
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 1 , 3, bias=False)
        self.conv1.load_state_dict(pretraind_model.conv1.state_dict().copy())

        self.bn1 = nn.BatchNorm2d(64)
        self.bn1.load_state_dict(pretraind_model.bn1.state_dict().copy())
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = self._make_layer(BasicBlock, 64, 3, stride=2)
        # self.conv2.load_state_dict(pretraind_model.layer1.state_dict().copy())
        self.conv3 = self._make_layer(BasicBlock, 128, 4, stride=2)
        self.conv3.load_state_dict(pretraind_model.layer2.state_dict().copy())
        self.conv4 = self._make_layer(BasicBlock, 256, 6, stride=2)
        self.conv4.load_state_dict(pretraind_model.layer3.state_dict().copy())
        self.conv5 = self._make_layer(BasicBlock, 512, 3, stride=2)
        self.conv5.load_state_dict(pretraind_model.layer4.state_dict().copy())

        self.compressed_conif = [64, 64, 128, 256, 512]
        self.compressed = []
        for i in range(5):
            layer = nn.Sequential(
            nn.Conv2d(self.compressed_conif[i], 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True))
            self.compressed.append(layer.cuda())

        self.output = self.make_infer(2, 5 * 8)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def make_infer(self, n_infer, n_in_feat):
        infer_layers = []
        for i in range(n_infer - 1):
            if i == 0:
                conv = nn.Sequential(
                    nn.Conv2d(n_in_feat, 8, 3, 1, 1),
                    nn.BatchNorm2d(8),
                    nn.ReLU(True)
                )
            else:
                conv = nn.Sequential(
                    nn.Conv2d(8, 8, 3, 1, 1),
                    nn.BatchNorm2d(8),
                    nn.ReLU(True)
                )
            infer_layers.append(conv)

        if n_infer == 1:
            infer_layers.append(nn.Sequential(nn.Conv2d(n_in_feat, 1, 1)))
        else:
            infer_layers.append(nn.Sequential(nn.Conv2d(8, 8, 3, 1, 1), nn.ReLU(True)))
            infer_layers.append(nn.Sequential(nn.Conv2d(8, 1, 1)))

        return nn.Sequential(*infer_layers)

    def forward(self, x):
        origin_size = (x.size(2), x.size(3))
        fixed_down_size = (x.size(2)//self.scale_factor, x.size(3)//self.scale_factor)
        c1 = self.conv1(x)
        c1 = self.bn1(c1)
        c1 = self.relu(c1)
        sp1 = self.compressed[0](c1)

        c2 = self.conv2(c1)
        sp2 = F.interpolate(self.compressed[1](c2), size=origin_size, mode='bilinear', align_corners=True)
        c2 = F.interpolate(c2, size=fixed_down_size, mode='bilinear', align_corners=True)

        c3 = self.conv3(c2)
        sp3 = F.interpolate(self.compressed[2](c3), size=origin_size, mode='bilinear', align_corners=True)
        c3 = F.interpolate(c3, size=fixed_down_size, mode='bilinear', align_corners=True)

        c4 = self.conv4(c3)
        sp4 = F.interpolate(self.compressed[3](c4), size=origin_size, mode='bilinear', align_corners=True)
        c4 = F.interpolate(c4, size=fixed_down_size, mode='bilinear', align_corners=True)

        c5 = self.conv5(c4)
        sp5 = F.interpolate(self.compressed[4](c5), size=origin_size, mode='bilinear', align_corners=True)

        cat = torch.cat([sp1, sp2, sp3, sp4, sp5], 1)
        out = self.output(cat)

        return F.sigmoid(out)

class AV_net(nn.Module):
    def __init__(self):
        super(AV_net, self).__init__()

        self.net = SSANet3(models.resnet34())

    def forward(self, x):

        out = self.net(x)
        return out

# make directory from path(same os.makedirs(path, exist_ok=True) on python 3.x)
def mkdir(path):
    if os.path.exists(path) == False:
        os.mkdir(path)

# fixed vessel segmentation image size
HRF_size = [1024, 1536]
# set image path
DB_path = '../DB/test/'
# set out path
save_path = DB_path
mkdir(save_path)

# create deep learning model & load to saved data
FP_model_weight_path = './DL_model/fundus/model_8000_iter_loss_84949.1797.pth.tar'
# FA_model_weight_path = './DL_model/FA/sub_mean_model_8400_iter_loss_35412.4364.pth.tar'
FA_model_weight_path = '/mnt/hdd/code/8_Registration/011_miccai20_retraining_version/FP-FA_registration_20200518/DL_model/FA/AV_set_retraining_model_50000_iter_loss_65253.8840_acc_0.9315_vessel_acc_0.9952.pth.tar'
FAG_model, FP_model = utils.get_model(FP_model_weight_path, FA_model_weight_path)
# FAG_model = AV_net()
# FAG_model.load_state_dict(torch.load(FA_model_weight_path))
# FAG_model.cuda(0)
# FAG_model.eval()

# save directory name list
intermedia_result_method_name = ['/1.SIFT/', '/2.VesselProb/', '/3.BSP_FAG/', '/4.Aggregation/', '/5.Chamfer/', '/6.BSP_FP-FAG/', '/7.Final/']

def data_load(path):
    all_raw_dir_list = {}
    for dir_name in sorted(os.listdir(path)):
        cur_dir_list = []
        dir_path = path + dir_name + '/'
        for file_name in sorted(os.listdir(dir_path)):
            file_path = dir_path + file_name
            if os.path.isfile(file_path) and (file_name[-3:] == 'jpg' or file_name[-3:] == 'png'):
                cur_dir_list.append(file_path)
        all_raw_dir_list[dir_name] = cur_dir_list

    all_dir_list = {}
    for key, item in all_raw_dir_list.items():
        all_dir_list[key] = {'fundus':[], 'FOV':[], 'FAGs':[]}
        for cur_path in item:
            cur_file_name = cur_path.split('/')[-1]
            if cur_file_name.find('1_origin.png') != -1:
                all_dir_list[key]['fundus'].append(cur_path)
            elif cur_file_name.find('6_FOV_mask.png') != -1:
                all_dir_list[key]['FOV'].append(cur_path)
            elif cur_file_name.find('FAG_FAG2FP.png') != -1:
                all_dir_list[key]['FAGs'].append(cur_path)

    return all_dir_list

def make_output_directory(cur_dir_path):
    # only for showing SIFT registration result.
    SIFT_result_save_path = cur_dir_path + intermedia_result_method_name[0]
    mkdir(SIFT_result_save_path)

    # only for showing Vessel Probability map with deep learning.
    VesselProb_result_save_path = cur_dir_path + intermedia_result_method_name[1]
    mkdir(VesselProb_result_save_path)

    # only for showing BSpline registration result.
    BSpline_result_save_path = cur_dir_path + intermedia_result_method_name[2]
    mkdir(BSpline_result_save_path)

    return SIFT_result_save_path, VesselProb_result_save_path, BSpline_result_save_path

def read_FOV(path, size):
    FOV_mask = np.array(Image.open(path).convert('L').resize(size=size, resample=Image.BILINEAR))
    return FOV_mask

def read_fundus_and_vessel_extraction(path, size, FOV_mask, FP_data, VesselProb_result_save_path):
    fundus_image = np.array(Image.open(path).convert('RGB').resize(size=size,
                                                                 resample=Image.BILINEAR))

    FP_data['path'] = path
    FP_data['fname'] = path.split('/')[-1]
    FP_data['FP_img'] = fundus_image

    FP_VessProbMap = utils.FP_vessel_prediction2(FP_model, FP_data['FP_img'], FOV_mask, size=size[::-1])
    FP_data['FPVPmap'] = FP_VessProbMap

    # write Vessel Probability map.
    Image.fromarray((FP_data['FPVPmap']*255).astype(np.ubyte)).save(
        VesselProb_result_save_path + FP_data['fname'][:-4] + '_VesselProb.png')


def read_FAGs_and_vessel_extraction(path_list, size, FOV_mask, FAG_data, SIFT_result_save_path, VesselProb_result_save_path, BSpline_result_save_path, model):
    continued_max = np.zeros(size[::-1])
    continued_avg = np.zeros(size[::-1])

    model.eval()
    # only FAG image
    for file_path in path_list:

        # read image and convert to numpy array.
        img = Image.open(file_path)
        img = img.convert('RGB').resize(size=(HRF_size[1], HRF_size[0]), resample=Image.BILINEAR)
        img = np.array(img, dtype=np.ubyte)
        img[FOV_mask == 0] = 0

        # load FAG image
        FAG_data.append({})
        FAG_data[-1]['path'] = file_path
        FAG_data[-1]['fname'] = file_path.split('/')[-1]
        FAG_data[-1]['FAG_img'] = img


        # first FAG image is do not registration.
        # registration based on SIFT decriptor matching method is operated second FAG image(moving image) with before image(fixed image).
        if len(FAG_data) > 1:
            # SIFT registration
            sift = regist_SIFT(FAG_data[-2]['SIFT'], FAG_data[-1]['FAG_img'], FAG_data[-2]['FOV'], FOV_mask,
                               2)  # get SIFT registration Class instance.
            # do registration based on SIFT detection, descriptor matching, and RANSAC.
            SIFT_st = datetime.now()
            bFail, [regi_img1, h, sift_kpt_imgs, matching_img, kpt, des, pts, new_matching_img,
                    new_matching_img_row] = sift.do_registration()
            if bFail == False:
                if (FAG_data) <=2:
                    FAG_data.pop(-2)
                    FAG_data[-1]['SIFT'] = FAG_data[-1]['FAG_img']
                    cur_FAG_FOV_mask = FOV_mask.copy()
                else:
                    FAG_data.pop(-1)
                    continue
            if len(FAG_data) >= 2:
                SIFT_ed = datetime.now()
                # SIFT_t_list.append((SIFT_ed - SIFT_st).total_seconds())
                cur_FAG_FOV_mask = cv2.warpPerspective(FOV_mask, h, (FOV_mask.shape[1], FOV_mask.shape[0]))
                FAG_data[-1]['SIFT'] = regi_img1
        elif len(FAG_data) == 1:
            FAG_data[-1]['SIFT'] = FAG_data[-1]['FAG_img']
            cur_FAG_FOV_mask = FOV_mask.copy()

        FAG_data[-1]['FOV'] = cur_FAG_FOV_mask.copy()

        # write SIFT registration result image.
        Image.fromarray(FAG_data[-1]['SIFT']).save(SIFT_result_save_path + FAG_data[-1]['fname'][:-4] + '_SIFT.png')
        Image.fromarray(cur_FAG_FOV_mask).save(
            SIFT_result_save_path + FAG_data[-1]['fname'][:-4] + '_SIFT_mask.png')

        if len(FAG_data) > 1:
            Image.fromarray(sift_kpt_imgs[0]).save(
                SIFT_result_save_path + FAG_data[-1]['fname'][:-4] + '_SIFT_target_detect_key.png')
            Image.fromarray(sift_kpt_imgs[1]).save(
                SIFT_result_save_path + FAG_data[-1]['fname'][:-4] + '_SIFT_source_detect_key.png')
            Image.fromarray(matching_img).save(
                SIFT_result_save_path + FAG_data[-1]['fname'][:-4] + '_SIFT_key_matching.png')
            Image.fromarray(new_matching_img).save(
                SIFT_result_save_path + FAG_data[-1]['fname'][:-4] + '_SIFT_key_matching_new.png')
            Image.fromarray(new_matching_img_row).save(
                SIFT_result_save_path + FAG_data[-1]['fname'][:-4] + '_SIFT_key_matching_new_row.png')

        # get vessel probability map of FAG image with deep learning.
        # AS-IS model
        # FAG_VessProbMap = utils.FAG_vessel_prediction3(FAG_model, FAG_data[-1]['SIFT'], FOV_mask, size=HRF_size)
        FAG_VessProbMap = utils.FAG_vessel_prediction_retrain_AVset(model, FAG_data[-1]['SIFT'], FOV_mask, size=HRF_size)

        FAG_data[-1]['FAGVPmap'] = FAG_VessProbMap

        # write Vessel Probability map.
        Image.fromarray((FAG_data[-1]['FAGVPmap'] * 255).astype(np.ubyte)).save(
            VesselProb_result_save_path + FAG_data[-1]['fname'][:-4] + '_VesselProb.png')


        # first FAG image is target frame, so just fixed it.
        if len(FAG_data) > 1:
            # deformable registration with BSpline
            BSP_st = datetime.now()
            bsp = regist_BSpline(continued_max, FAG_data[-1]['FAGVPmap'] * 255.)
            regi_img2 = bsp.do_registration()
            BSP_ed = datetime.now()
            # BSP_t_list.append((BSP_ed - BSP_st).total_seconds())
            dens_disp, draw_disp = bsp.get_displacement_vector_field()

            Image.fromarray(draw_disp.astype(np.ubyte)).save(
                BSpline_result_save_path + FAG_data[-1]['fname'][:-4] + '_displacement_vector_field.png')

            FAG_data[-1]['BSP'] = regi_img2.astype(np.ubyte)
        elif len(FAG_data) == 1:
            # first frame(fixed)
            FAG_data[-1]['BSP'] = (FAG_data[-1]['FAGVPmap'] * 255).astype(np.ubyte)

        # write Vessel Probability map.
        Image.fromarray((FAG_data[-1]['BSP']).astype(np.ubyte)).save(
            BSpline_result_save_path + FAG_data[-1]['fname'][:-4] + '_BSP.png')

        # pixel-wise maximum vessel probaility map.
        continued_max = np.concatenate([continued_max.reshape([1, continued_max.shape[0], continued_max.shape[1]])
                                           , FAG_data[-1]['BSP'].reshape(
                [1, continued_max.shape[0], continued_max.shape[1]])], 0)
        continued_max = np.max(continued_max, 0).astype(np.float32)

        # pixel-wise average vessel probaility map.
        continued_avg = np.concatenate(
            [continued_avg.reshape([1, continued_avg.shape[0], continued_avg.shape[1]])
                , FAG_data[-1]['BSP'].reshape([1, continued_avg.shape[0], continued_avg.shape[1]])], 0)
        continued_avg = np.mean(continued_avg, 0).astype(np.float32)

        # write maximum Vessel Probability map.
        Image.fromarray(continued_max.astype(np.ubyte)).save(
            BSpline_result_save_path + FAG_data[-1]['fname'][:-4] + '_BSP_continued_max.png')

        # write average Vessel Probability map.
        Image.fromarray(continued_avg.astype(np.ubyte)).save(
            BSpline_result_save_path + FAG_data[-1]['fname'][:-4] + '_BSP_continued_avg.png')

        # applied registration result to original image and write.
        if len(FAG_data) > 1:
            FAG_data[-1]['FAG_ORIGIN_BSP'] = bsp.registrationFromMatrix(
                FAG_data[-1]['SIFT'][:, :, 0].astype(np.float32))
            FAG_data[-1]['FOV'] = bsp.registrationFromMatrix(FAG_data[-1]['FOV'].astype(np.float32))
        else:
            FAG_data[-1]['FAG_ORIGIN_BSP'] = FAG_data[-1]['SIFT'][:, :, 0]

        origin_FAG_img_save_path = BSpline_result_save_path + 'Origin_FAG_domain/'
        mkdir(origin_FAG_img_save_path)
        # write Vessel Probability map.
        Image.fromarray((FAG_data[-1]['FAG_ORIGIN_BSP']).astype(np.ubyte)).save(
            origin_FAG_img_save_path + FAG_data[-1]['fname'][:-4] + '_BSP_FAG_ORIGIN.png')

        Image.fromarray((FAG_data[-1]['FOV']).astype(np.ubyte)).save(
            BSpline_result_save_path + FAG_data[-1]['fname'][:-4] + '_BSP_FOV.png')
        # except:
        #     FAG_data.pop(-1)


def post_processing(FP_data, FAG_data, FOV_mask, root_save_path):
    # put FOV mask into dict.
    FP_data['FOV'] = FOV_mask.copy()
    FP_data['FPVPmap'] = np.array(Image.fromarray((FP_data['FPVPmap'] * 255).astype(np.ubyte)).resize( \
        [FAG_data[0]['BSP'].shape[1], FAG_data[0]['BSP'].shape[0]], Image.BILINEAR)).astype(np.float32) / 255.

    # convert list type to numpy array type
    registrated_FAG_set = []
    for cur_FAG_data in FAG_data:
        registrated_FAG_set.append(cur_FAG_data['BSP'].astype(np.float32))
    registrated_FAG_set = np.array(registrated_FAG_set).astype(np.float32) / 255.
    # registrated_FAG_set_range1 = np.array(registrated_FAG_set[:int(len(registrated_FAG_set)*0.5+0.5)]).astype(np.float32)/255.
    # registrated_FAG_set_range2 = np.array(registrated_FAG_set[:int(len(registrated_FAG_set)*0.7+0.5)]).astype(np.float32)/255.

    # aggregated FA by pixwel-wise average image and maximum image.
    avg_FAG = np.average(registrated_FAG_set, 0)
    max_FAG = np.max(registrated_FAG_set, 0)

    # subtaction(just see difference between both images)
    sub_max2avg = max_FAG - avg_FAG

    # only for showing average and maximum image.
    avg_max_result_save_path = root_save_path + intermedia_result_method_name[3]
    mkdir(avg_max_result_save_path)

    # write average image.
    Image.fromarray((avg_FAG * 255).astype(np.ubyte)).save(
        avg_max_result_save_path + FAG_data[-1]['fname'][:-7] + '_avg.png')

    # write maximum image.
    Image.fromarray((max_FAG * 255).astype(np.ubyte)).save(
        avg_max_result_save_path + FAG_data[-1]['fname'][:-7] + '_max.png')

    # write maximum image.
    Image.fromarray((sub_max2avg * 255).astype(np.ubyte)).save(
        avg_max_result_save_path + FAG_data[-1]['fname'][:-7] + '_sub.png')

    # Chamfer Matcging.
    Chamfer_st = datetime.now()
    translated_FAG, t, angle = chamfer_matching(max_FAG, FP_data['FPVPmap'])
    Chamfer_ed = datetime.now()
    # Chamfer_t_list.append((Chamfer_ed - Chamfer_st).total_seconds())

    # apply Chamfer matching result.
    translation_matrix = np.float32([[1, 0, t[1]], [0, 1, t[0]]])
    img_translation = cv2.warpAffine(avg_FAG, translation_matrix, (translated_FAG.shape[1], translated_FAG.shape[0]))
    avg_FAG = skimage.transform.rotate(img_translation, angle, resize=False)

    # only for showing Chamfer matching result image.
    Chamfer_result_save_path = root_save_path + intermedia_result_method_name[4]
    mkdir(Chamfer_result_save_path)

    # write Chamfer Matching result image.
    Image.fromarray((translated_FAG * 255).astype(np.ubyte)).save(
        Chamfer_result_save_path + FAG_data[-1]['fname'][:-7] + '_Chamfer.png')

    # applied registration result to original image and write.
    for cur_FAG_data in FAG_data:
        FAG_img = cur_FAG_data['FAG_ORIGIN_BSP'].astype(np.ubyte)
        img_translation = cv2.warpAffine(FAG_img, translation_matrix,
                                         (translated_FAG.shape[1], translated_FAG.shape[0]))
        rigid_registration_FAG = skimage.transform.rotate(img_translation, angle, resize=False)

        FAG_VP_img = cur_FAG_data['BSP'].astype(np.ubyte)
        img_translation = cv2.warpAffine(FAG_VP_img, translation_matrix,
                                         (translated_FAG.shape[1], translated_FAG.shape[0]))
        cur_FAG_data['global_FAGVP'] = skimage.transform.rotate(img_translation, angle, resize=False)

        img_translation = cv2.warpAffine(cur_FAG_data['FOV'].astype(np.ubyte), translation_matrix,
                                         (translated_FAG.shape[1], translated_FAG.shape[0]))
        cur_FAG_data['FOV'] = skimage.transform.rotate(img_translation, angle, resize=False)

        origin_FAG_img_save_path = Chamfer_result_save_path + 'Origin_FAG_domain/'
        mkdir(origin_FAG_img_save_path)

        Image.fromarray((rigid_registration_FAG * 255).astype(np.ubyte)).save(
            origin_FAG_img_save_path + cur_FAG_data['fname'][:-4] + '_Chamfer.png')

        cur_FAG_data['FAG_ORIGIN_Chamfer'] = rigid_registration_FAG
        'Origin_FAG_domain'

    # defomable registration for FP-FA with BSpline.
    BSP_FPFA_st = datetime.now()
    bsp = regist_BSpline(FP_data['FPVPmap'] * 255, translated_FAG * 255.)
    regi_FAG2FP = bsp.do_registration()
    BSP_FPFA_ed = datetime.now()
    # BSP_FPFA_t_list.append((BSP_FPFA_ed - BSP_FPFA_st).total_seconds())
    regi_FAG2FP = np.array(Image.fromarray(regi_FAG2FP).resize([FP_data['FP_img'].shape[1], FP_data['FP_img'].shape[0]],
                                                               Image.BILINEAR)).copy()
    dens_disp, draw_disp = bsp.get_displacement_vector_field()

    # apply registration.
    registrated_avg_FAG = bsp.registrationFromMatrix(avg_FAG * 255)
    registrated_avg_FAG = np.array(
        Image.fromarray(registrated_avg_FAG).resize([FP_data['FP_img'].shape[1], FP_data['FP_img'].shape[0]])).astype(
        np.ubyte)

    resized_FP_VPmap = np.array(
        Image.fromarray((FP_data['FPVPmap'] * 255).astype(np.ubyte)).resize(
            [FP_data['FP_img'].shape[1], FP_data['FP_img'].shape[0]], Image.BILINEAR))

    # only for showing FP-FAG BSpline registration result image.
    FAG2FP_BSpline_result_save_path = root_save_path + intermedia_result_method_name[5]
    mkdir(FAG2FP_BSpline_result_save_path)

    # write FP-FAG registration result image.
    Image.fromarray((regi_FAG2FP).astype(np.ubyte)).save(
        FAG2FP_BSpline_result_save_path + FAG_data[-1]['fname'][:-7] + '_FAG2FP.png')

    # write FP VPmap result image.
    Image.fromarray(resized_FP_VPmap).save(
        FAG2FP_BSpline_result_save_path + FAG_data[-1]['fname'][:-7] + '_FP_VPmap.png')

    # write FP-FAG registration result image.
    Image.fromarray(registrated_avg_FAG).save(
        FAG2FP_BSpline_result_save_path + FAG_data[-1]['fname'][:-7] + '_FAG2FP_avg.png')

    Image.fromarray(draw_disp).save(
        FAG2FP_BSpline_result_save_path + FAG_data[-1]['fname'][:-7] + '_displacement_vector_field.png')

    # aggregated FOV mask
    # and applied registration result to original image and write.
    registrated_FAG_FOV_set = []
    for cur_FAG_data in FAG_data:
        FAG_img = cur_FAG_data['FAG_ORIGIN_Chamfer']
        FAG_img = bsp.registrationFromMatrix(FAG_img)
        cur_FAG_data['FAG_ORIGIN_FAG2FP'] = FAG_img
        FAG_VP_img = cur_FAG_data['global_FAGVP']
        cur_FAG_data['global_FAGVP'] = bsp.registrationFromMatrix(FAG_VP_img)
        registrated_FAG_FOV_set.append(bsp.registrationFromMatrix(cur_FAG_data['FOV']).astype(np.ubyte) * 255)

        origin_FAG_img_save_path = FAG2FP_BSpline_result_save_path + 'Origin_FAG_domain/'
        mkdir(origin_FAG_img_save_path)

        Image.fromarray((FAG_img * 255).astype(np.ubyte)).save(
            origin_FAG_img_save_path + cur_FAG_data['fname'][:-4] + '_FAG2FP.png')

    FAG_FOV_aggregation = np.max(np.array(registrated_FAG_FOV_set), axis=0)
    FP_FAG_FOV = np.bitwise_and(FAG_FOV_aggregation, FP_data['FOV'])

    ## post-processing ##
    Post_st = datetime.now()
    Origin_FAG = []
    Origin_FAG_minmax = []
    global_FAGVP = []
    for cur_FAG_data in FAG_data:
        FAG_img = cur_FAG_data['FAG_ORIGIN_FAG2FP']
        FAG_img[FP_FAG_FOV == 0] = 0
        Origin_FAG.append(FAG_img)
        FAG_img = FAG_img.astype(np.float32)
        Origin_FAG_minmax.append((FAG_img - FAG_img.min()) / (FAG_img.max() - FAG_img.min()))
        global_FAGVP.append(cur_FAG_data['global_FAGVP'])

    Origin_FAG = np.array(Origin_FAG)
    Origin_FAG_minmax = np.array(Origin_FAG_minmax)

    # search maximum enhanced FA frame
    max_bright_idx = 0
    max_bright = 0
    for j in range(len(Origin_FAG)):
        if max_bright < Origin_FAG[j].mean():
            max_bright = Origin_FAG[j].mean()
            max_bright_idx = j

    # compute reverse frangi called vally detection in our papre.
    gray_FP = np.array(Image.fromarray(FP_data['FP_img']).convert('L'))
    gray_FP = gray_FP.astype(np.float32)
    gray_FP -= gray_FP.min()
    gray_FP /= gray_FP.max()
    gray_FP = (gray_FP * 255).astype(np.ubyte)
    reverse_frangi_FP = frangi(255 - gray_FP, scale_range=[1, 3], scale_step=1)
    reverse_frangi_FP = (reverse_frangi_FP - reverse_frangi_FP.min()) / ((1e-5) - reverse_frangi_FP.min())
    FAG = np.array(Image.fromarray((Origin_FAG[max_bright_idx] * 255).astype(np.ubyte)).convert('L')).astype(np.float32)
    FAG -= FAG.min()
    FAG /= FAG.max()
    FAG = (FAG * 255).astype(np.ubyte)
    reverse_frangi_FAG = frangi(FAG, scale_range=[1, 3], scale_step=1)
    reverse_frangi_FAG = (reverse_frangi_FAG - reverse_frangi_FAG.min()) / ((5e-5) - reverse_frangi_FAG.min())

    registrated_avg_FAG[FP_FAG_FOV == 0] = 0
    regi_FAG2FP[FP_FAG_FOV == 0] = 0

    # binary image with fixed threshold value.
    bn_regi_FAG2FP = regi_FAG2FP >= (0.6 * 255.)

    # hysteresis with skeletonization method.
    tmp = regi_FAG2FP >= (0.1 * 255.)
    dt = cv2.distanceTransform((tmp).astype(np.ubyte), cv2.DIST_L2, 3)
    dt2 = cv2.distanceTransform((bn_regi_FAG2FP >= 0.75 * 255.).astype(np.ubyte), cv2.DIST_L2, 3)
    rapidly_increase = (np.abs(dt - dt2) > 2)
    thin = skimage.morphology.skeletonize(tmp).astype(np.float32)

    # merge large vessel and thin vessel
    FAG_thin = np.bitwise_and(dt <= 3, thin == 1)
    FAG_combine = np.bitwise_or(bn_regi_FAG2FP, thin == 1).astype(np.ubyte) * 255

    # prevent very closed vessel using vally detection result.
    bn_frangi = (reverse_frangi_FAG > 0.3).astype(np.ubyte) * 255
    bn_frangi[regi_FAG2FP >= 254] = 0
    bn_res_frangi = FAG_combine.copy()
    bn_res_frangi[bn_frangi == 255] = 0

    # remove very small region.
    cca = cv2.connectedComponentsWithStats(bn_res_frangi, connectivity=8)
    remove_noise_img = np.zeros(bn_res_frangi.shape, dtype=np.ubyte)
    for n in range(1, cca[0] + 1):
        cur_label_img = cca[1] == n
        n_pixel = cur_label_img.sum()
        if n_pixel > 50:
            remove_noise_img += cur_label_img
    remove_noise_img = remove_noise_img * 255

    Post_ed = datetime.now()
    # Post_t_list.append((Post_ed - Post_st).total_seconds())

    ## write final results ##
    Final_result_save_path = root_save_path + intermedia_result_method_name[6]
    mkdir(Final_result_save_path)

    Image.fromarray((regi_FAG2FP).astype(np.ubyte)).save(
        Final_result_save_path + FAG_data[-1]['fname'][:-7] + '_05_prob.png')

    Image.fromarray((FAG_combine).astype(np.ubyte)).save(
        Final_result_save_path + FAG_data[-1]['fname'][:-7] + '_04_binary.png')

    overlay = FP_data['FP_img'].copy()
    overlay[FAG_combine != 0, 2] = 255
    overlay[FP_FAG_FOV == 0] = 0

    Image.fromarray((overlay).astype(np.ubyte)).save(
        Final_result_save_path + FAG_data[-1]['fname'][:-7] + '_03_overlay.png')

    Image.fromarray(FP_data['FP_img'].astype(np.ubyte)).save(
        Final_result_save_path + FAG_data[-1]['fname'][:-7] + '_02_.png')

    Image.fromarray((FP_FAG_FOV).astype(np.ubyte)).save(
        Final_result_save_path + FAG_data[-1]['fname'][:-7] + '_01_FOV.png')

    Image.fromarray((reverse_frangi_FP * 255).astype(np.ubyte)).save(
        Final_result_save_path + FAG_data[-1]['fname'][:-7] + '_10_reverse_frangi_FP.png')

    Image.fromarray((reverse_frangi_FAG * 255).astype(np.ubyte)).save(
        Final_result_save_path + FAG_data[-1]['fname'][:-7] + '_11_reverse_frangi_FAG.png')

    Image.fromarray((bn_res_frangi).astype(np.ubyte)).save(
        Final_result_save_path + FAG_data[-1]['fname'][:-7] + '_6_binary_reverse_frnagi_.png')

    overlay = FP_data['FP_img'].copy()
    overlay[bn_res_frangi != 0, 2] = 255
    overlay[FP_FAG_FOV == 0] = 0

    Image.fromarray((overlay).astype(np.ubyte)).save(
        Final_result_save_path + FAG_data[-1]['fname'][:-7] + '_07_binary_reverse_frnagi_overlay.png')

    Image.fromarray((remove_noise_img).astype(np.ubyte)).save(
        Final_result_save_path + FAG_data[-1]['fname'][:-7] + '_08_binary_remove_noise(CCA).png')

    overlay = FP_data['FP_img'].copy()
    overlay[remove_noise_img != 0, 2] = 255
    overlay[FP_FAG_FOV == 0] = 0

    Image.fromarray((overlay).astype(np.ubyte)).save(
        Final_result_save_path + FAG_data[-1]['fname'][:-7] + '_09_binary_remove_noise(CCA)_overlay.png')

    for cur_FAG_data in FAG_data:
        FAG_img = cur_FAG_data['FAG_ORIGIN_FAG2FP']
        FAG_img[FP_FAG_FOV == 0] = 0

        origin_FAG_img_save_path = Final_result_save_path + 'Origin_FAG_domain/'
        mkdir(origin_FAG_img_save_path)

        Image.fromarray((FAG_img * 255).astype(np.ubyte)).save(
            origin_FAG_img_save_path + cur_FAG_data['fname'][:-4] + '_FAG2FP.png')

        FAG_img_32f = FAG_img.astype(np.float32)

    # # recode running time.
    # et = datetime.now()
    # cur_seq_t = (et - st).total_seconds()
    # mean_sift_t = np.mean(np.array(SIFT_t_list))
    # mean_BSP_t = np.mean(np.array(BSP_t_list))
    # mean_Chamfer_t = Chamfer_t_list[0]
    # mean_BSP_FPFA_t = BSP_FPFA_t_list[0]
    # mean_Post_t = Post_t_list[0]
    #
    # csv_writer.writerow(
    #     [seq_dir + '_' + laterality_dir, cur_seq_t, mean_sift_t, mean_BSP_t, mean_Chamfer_t, mean_BSP_FPFA_t,
    #      mean_Post_t])
    # f.flush()

#main
all_dir_list = data_load(DB_path)
for key, cur_dir_data in all_dir_list.items():
    cur_dir_path = DB_path + key + '/'
    fundus_path = cur_dir_data['fundus'][0]
    FOV_path = cur_dir_data['FOV'][0]
    FAG_path_list = cur_dir_data['FAGs']

    SIFT_result_save_path, VesselProb_result_save_path, BSpline_result_save_path = \
        make_output_directory(cur_dir_path
                              )
    # declare for each recording space.
    FP_data = {}  # Fundus Photo data.
    FAG_data = []  # FAG data.

    FOV_mask = read_FOV(FOV_path, (HRF_size[1], HRF_size[0]))
    read_fundus_and_vessel_extraction(fundus_path, (HRF_size[1], HRF_size[0]), FOV_mask, FP_data, VesselProb_result_save_path)
    read_FAGs_and_vessel_extraction(FAG_path_list, (HRF_size[1], HRF_size[0]), FOV_mask, FAG_data, SIFT_result_save_path, VesselProb_result_save_path, BSpline_result_save_path, FAG_model)
    post_processing(FP_data,FAG_data,FOV_mask,cur_dir_path)






# #### main ####
# with open('time_info.csv', 'w') as f: # for time recode.
#     csv_writer = csv.writer(f)
#
#     all_seq_set = []
#     for seq_idx, seq_dir in enumerate(sorted(os.listdir(DB_path))): # loop for image sequence directory.
#         cur_seq_path = DB_path + seq_dir + '/'
#
#         if os.path.isdir(cur_seq_path) == False:
#             continue
#
#         save_cur_cur_seq_path = save_path + seq_dir + '/'
#         mkdir(save_cur_cur_seq_path)
#
#         st = datetime.now()
#         SIFT_t_list = []
#         BSP_t_list = []
#         Chamfer_t_list = []
#         BSP_FPFA_t_list = []
#         Post_t_list = []
#
#
#         # declare for each recording space.
#         FP_data = {} # Fundus Photo data.
#         FAG_data = [] # FAG data.
#
#         # only for showing SIFT registration result.
#         SIFT_result_save_path = save_cur_laterality_path + intermedia_result_method_name[0]
#         mkdir(SIFT_result_save_path)
#
#         # only for showing Vessel Probability map with deep learning.
#         VesselProb_result_save_path = save_cur_laterality_path + intermedia_result_method_name[1]
#         mkdir(VesselProb_result_save_path)
#
#         # only for showing BSpline registration result.
#         BSpline_result_save_path = save_cur_laterality_path + intermedia_result_method_name[2]
#         mkdir(BSpline_result_save_path)
#
#         # first, read mask image
#         b_read_FOV_mask = False
#         for fname in sorted(os.listdir(cur_laterality_path)):
#             file_path = cur_laterality_path + fname
#             # prevent for directory.
#             if os.path.isdir(file_path):
#                 continue
#             # check existing mask image in file name.
#             is_mask = (fname.find('mask')!= -1 or fname.find('FOV')!= -1)
#
#             # read mask image
#             if is_mask:
#                 img = Image.open(file_path)
#                 img = img.convert('L').resize(size=(HRF_size[1], HRF_size[0]), resample=Image.BILINEAR)
#                 img = np.around(np.array(img, dtype=np.ubyte).copy())
#                 FOV_mask = img.copy()
#                 b_read_FOV_mask = True
#
#         # second, read fundus photo image
#         for fname in sorted(os.listdir(cur_laterality_path)):
#             file_path = cur_laterality_path + fname
#             # prevent for directory.
#             if os.path.isdir(file_path):
#                 continue
#
#             # check existing fundus photo in file name.
#             is_fundus_rgb = fname.find('RGB') != -1
#
#             # read fundus_photo image
#             if is_fundus_rgb:
#                 img = Image.open(file_path)
#                 img = img.convert('RGB').resize(size=(HRF_size[1], HRF_size[0]), resample=Image.BILINEAR)
#                 img = np.array(img, dtype=np.ubyte).copy()
#
#                 FP_data['path'] = file_path
#                 FP_data['fname'] = fname
#                 FP_data['FP_img'] = img
#
#                 # if there is no existing FOV mask, make naive FOV mask
#                 if not b_read_FOV_mask:
#                     FOV_mask = np.bitwise_or(FP_data['FP_img'][:, :, 0] >= 10,
#                                              np.bitwise_or(FP_data['FP_img'][:, :, 1] >= 10,
#                                                            FP_data['FP_img'][:, :, 2] >= 10)).astype(np.ubyte) * 255
#
#                 # get vessel probability map of Fundus Photo image with deep learning.
#                 FP_VessProbMap = utils.FP_vessel_prediction2(FP_model, FP_data['FP_img'], FOV_mask, size=HRF_size)
#                 FP_data['FPVPmap'] = FP_VessProbMap
#
#                 # write Vessel Probability map.
#                 Image.fromarray((FP_data['FPVPmap'] * 255).astype(np.ubyte)).save(
#                     VesselProb_result_save_path + FP_data['fname'][:-4] + '_VesselProb.png')
#
#         if len(FP_data) == 0:
#             continue
#         continued_max = np.zeros(FP_data['FP_img'].shape[:-1])
#         continued_avg = np.zeros(FP_data['FP_img'].shape[:-1])
#
#         # third, read FAG image
#         for fname in sorted(os.listdir(cur_laterality_path)):
#             file_path = cur_laterality_path + fname
#             # prevent for directory.
#             if os.path.isdir(file_path):
#                 continue
#             # check fundus imaeg in file name.
#             # is_fundus_rgb = fname.find('fundus_RGB')
#             is_fundus_rgb = fname.find('RGB') == -1
#             is_fundus_gray = fname.find('GRAY')== -1 and fname.find('gray')== -1
#             is_mask = fname.find('mask')== -1
#             is_csv = fname.find('csv')== -1
#             is_FAG = fname.find('FAG') != -1
#
#             try:
#                 # only FAG image
#                 if is_fundus_rgb and is_mask and is_fundus_gray and is_csv and is_FAG:
#                     # read image and convert to numpy array.
#                     img = Image.open(file_path)
#                     img = img.convert('RGB').resize(size=(HRF_size[1], HRF_size[0]), resample=Image.BILINEAR)
#                     img = np.array(img, dtype=np.ubyte)
#                     img[FOV_mask==0] = 0
#
#                     # load FAG image
#                     FAG_data.append({})
#                     FAG_data[-1]['path'] = file_path
#                     FAG_data[-1]['fname'] = fname
#                     FAG_data[-1]['FAG_img'] = img
#
#                     if os.path.exists(SIFT_result_save_path+FAG_data[-1]['fname'][:-4]+'_SIFT.png'):
#                         regi_img1 = np.array(Image.open(SIFT_result_save_path+FAG_data[-1]['fname'][:-4]+'_SIFT.png'))
#                         FAG_data[-1]['SIFT'] = regi_img1
#                         FAG_data[-1]['FOV'] = np.array(Image.open(SIFT_result_save_path+FAG_data[-1]['fname'][:-4]+'_SIFT_mask.png'), dtype=np.float32)
#                     else:
#                         # first FAG image is do not registration.
#                         # registration based on SIFT decriptor matching method is operated second FAG image(moving image) with before image(fixed image).
#                         if len(FAG_data) > 1:
#                             # SIFT registration
#                             sift = regist_SIFT(FAG_data[-2]['SIFT'], FAG_data[-1]['FAG_img'], FAG_data[-2]['FOV'], FOV_mask, 2) # get SIFT registration Class instance.
#                             # do registration based on SIFT detection, descriptor matching, and RANSAC.
#                             SIFT_st = datetime.now()
#                             bFail, [regi_img1, h, sift_kpt_imgs, matching_img, kpt, des, pts, new_matching_img, new_matching_img_row] = sift.do_registration()
#                             if bFail == False:
#                                 FAG_data.pop(-1)
#                                 continue
#                             SIFT_ed = datetime.now()
#                             SIFT_t_list.append((SIFT_ed - SIFT_st).total_seconds())
#                             cur_FAG_FOV_mask = cv2.warpPerspective(FOV_mask, h, (FOV_mask.shape[1], FOV_mask.shape[0]))
#                             FAG_data[-1]['SIFT'] = regi_img1
#                         elif len(FAG_data) == 1:
#                             FAG_data[-1]['SIFT'] = FAG_data[-1]['FAG_img']
#                             cur_FAG_FOV_mask = FOV_mask.copy()
#
#                         FAG_data[-1]['FOV'] = cur_FAG_FOV_mask.copy()
#
#                         # write SIFT registration result image.
#                         Image.fromarray(FAG_data[-1]['SIFT']).save(SIFT_result_save_path+FAG_data[-1]['fname'][:-4]+'_SIFT.png')
#                         Image.fromarray(cur_FAG_FOV_mask).save(SIFT_result_save_path + FAG_data[-1]['fname'][:-4] + '_SIFT_mask.png')
#
#                         if len(FAG_data) > 1:
#                             Image.fromarray(sift_kpt_imgs[0]).save(SIFT_result_save_path + FAG_data[-1]['fname'][:-4] + '_SIFT_target_detect_key.png')
#                             Image.fromarray(sift_kpt_imgs[1]).save(SIFT_result_save_path + FAG_data[-1]['fname'][:-4] + '_SIFT_source_detect_key.png')
#                             Image.fromarray(matching_img).save(SIFT_result_save_path + FAG_data[-1]['fname'][:-4] + '_SIFT_key_matching.png')
#                             Image.fromarray(new_matching_img).save(
#                                 SIFT_result_save_path + FAG_data[-1]['fname'][:-4] + '_SIFT_key_matching_new.png')
#                             Image.fromarray(new_matching_img_row).save(
#                                 SIFT_result_save_path + FAG_data[-1]['fname'][:-4] + '_SIFT_key_matching_new_row.png')
#
#                     # get vessel probability map of FAG image with deep learning.
#                     FAG_VessProbMap = utils.FAG_vessel_prediction3(FAG_model, FAG_data[-1]['SIFT'], FOV_mask, size=HRF_size)
#                     FAG_data[-1]['FAGVPmap'] = FAG_VessProbMap
#
#                     # write Vessel Probability map.
#                     Image.fromarray((FAG_data[-1]['FAGVPmap'] * 255).astype(np.ubyte)).save(
#                         VesselProb_result_save_path + FAG_data[-1]['fname'][:-4] + '_VesselProb.png')
#
#                     if os.path.exists(BSpline_result_save_path + FAG_data[-1]['fname'][:-4] + '_BSP.png'):
#                         regi_img2 = np.array(Image.open(BSpline_result_save_path + FAG_data[-1]['fname'][:-4] + '_BSP.png'))
#                         FAG_data[-1]['BSP'] = regi_img2.astype(np.ubyte)
#                         continued_max = np.array(Image.open(BSpline_result_save_path + FAG_data[-1]['fname'][:-4] + '_BSP_continued_max.png'), dtype=np.float32)
#                         continued_avg = np.array(Image.open(BSpline_result_save_path + FAG_data[-1]['fname'][:-4] + '_BSP_continued_avg.png'), dtype=np.float32)
#                         origin_FAG_img_save_path = BSpline_result_save_path + 'Origin_FAG_domain/'
#                         FAG_data[-1]['FAG_ORIGIN_BSP'] = np.array(Image.open(origin_FAG_img_save_path + FAG_data[-1]['fname'][:-4] + '_BSP_FAG_ORIGIN.png')).astype(np.ubyte)
#                         FAG_data[-1]['FOV'] = np.array(Image.open(BSpline_result_save_path + FAG_data[-1]['fname'][:-4] + '_BSP_FOV.png')).astype(np.float32)
#                     else:
#                         # first FAG image is target frame, so just fixed it.
#                         if len(FAG_data) > 1:
#                             # deformable registration with BSpline
#                             BSP_st = datetime.now()
#                             bsp = regist_BSpline(continued_max, FAG_data[-1]['FAGVPmap']*255.)
#                             regi_img2 = bsp.do_registration()
#                             BSP_ed = datetime.now()
#                             BSP_t_list.append((BSP_ed-BSP_st).total_seconds())
#                             dens_disp, draw_disp = bsp.get_displacement_vector_field()
#
#                             Image.fromarray(draw_disp.astype(np.ubyte)).save(
#                                 BSpline_result_save_path + FAG_data[-1]['fname'][:-4] + '_displacement_vector_field.png')
#
#                             FAG_data[-1]['BSP'] = regi_img2.astype(np.ubyte)
#                         elif len(FAG_data) == 1:
#                             # first frame(fixed)
#                             FAG_data[-1]['BSP'] = (FAG_data[-1]['FAGVPmap'] * 255).astype(np.ubyte)
#
#                         # write Vessel Probability map.
#                         Image.fromarray((FAG_data[-1]['BSP']).astype(np.ubyte)).save(
#                             BSpline_result_save_path + FAG_data[-1]['fname'][:-4] + '_BSP.png')
#
#                         # pixel-wise maximum vessel probaility map.
#                         continued_max = np.concatenate([continued_max.reshape([1, continued_max.shape[0], continued_max.shape[1]])
#                                                , FAG_data[-1]['BSP'].reshape([1, continued_max.shape[0], continued_max.shape[1]])], 0)
#                         continued_max = np.max(continued_max, 0).astype(np.float32)
#
#                         # pixel-wise average vessel probaility map.
#                         continued_avg = np.concatenate(
#                             [continued_avg.reshape([1, continued_avg.shape[0], continued_avg.shape[1]])
#                                 , FAG_data[-1]['BSP'].reshape([1, continued_avg.shape[0], continued_avg.shape[1]])], 0)
#                         continued_avg = np.mean(continued_avg, 0).astype(np.float32)
#
#                         # write maximum Vessel Probability map.
#                         Image.fromarray(continued_max.astype(np.ubyte)).save(
#                             BSpline_result_save_path + FAG_data[-1]['fname'][:-4] + '_BSP_continued_max.png')
#
#                         # write average Vessel Probability map.
#                         Image.fromarray(continued_avg.astype(np.ubyte)).save(
#                             BSpline_result_save_path + FAG_data[-1]['fname'][:-4] + '_BSP_continued_avg.png')
#
#                         # applied registration result to original image and write.
#                         if len(FAG_data) > 1:
#                             FAG_data[-1]['FAG_ORIGIN_BSP'] = bsp.registrationFromMatrix(FAG_data[-1]['SIFT'][:,:,0].astype(np.float32))
#                             FAG_data[-1]['FOV'] = bsp.registrationFromMatrix(FAG_data[-1]['FOV'].astype(np.float32))
#                         else:
#                             FAG_data[-1]['FAG_ORIGIN_BSP'] = FAG_data[-1]['SIFT'][:,:,0]
#
#                         origin_FAG_img_save_path = BSpline_result_save_path + 'Origin_FAG_domain/'
#                         mkdir(BSpline_result_save_path)
#                         # write Vessel Probability map.
#                         Image.fromarray((FAG_data[-1]['FAG_ORIGIN_BSP']).astype(np.ubyte)).save(
#                             origin_FAG_img_save_path + FAG_data[-1]['fname'][:-4] + '_BSP_FAG_ORIGIN.png')
#
#                         Image.fromarray((FAG_data[-1]['FOV']).astype(np.ubyte)).save(
#                             BSpline_result_save_path + FAG_data[-1]['fname'][:-4] + '_BSP_FOV.png')
#             except:
#                 FAG_data.pop(-1)
#                 print("Fail!!")
#                 continue
#
#         if len(FAG_data) == 0:
#             continue
#
#         # put FOV mask into dict.
#         FP_data['FOV'] = FOV_mask.copy()
#         FP_data['FPVPmap'] = np.array(Image.fromarray((FP_data['FPVPmap']*255).astype(np.ubyte)).resize(\
#             [FAG_data[0]['BSP'].shape[1], FAG_data[0]['BSP'].shape[0]], Image.BILINEAR)).astype(np.float32)/255.
#
#         # convert list type to numpy array type
#         registrated_FAG_set = []
#         for cur_FAG_data in FAG_data:
#             registrated_FAG_set.append(cur_FAG_data['BSP'].astype(np.float32))
#         registrated_FAG_set = np.array(registrated_FAG_set).astype(np.float32)/255.
#         # registrated_FAG_set_range1 = np.array(registrated_FAG_set[:int(len(registrated_FAG_set)*0.5+0.5)]).astype(np.float32)/255.
#         # registrated_FAG_set_range2 = np.array(registrated_FAG_set[:int(len(registrated_FAG_set)*0.7+0.5)]).astype(np.float32)/255.
#
#         # aggregated FA by pixwel-wise average image and maximum image.
#         avg_FAG = np.average(registrated_FAG_set,0)
#         max_FAG = np.max(registrated_FAG_set,0)
#
#         # subtaction(just see difference between both images)
#         sub_max2avg = max_FAG - avg_FAG
#
#         # only for showing average and maximum image.
#         avg_max_result_save_path = save_cur_laterality_path + intermedia_result_method_name[3]
#         mkdir(avg_max_result_save_path)
#
#         # write average image.
#         Image.fromarray((avg_FAG * 255).astype(np.ubyte)).save(
#             avg_max_result_save_path + FAG_data[-1]['fname'][:-7] + '_avg.png')
#
#         # write maximum image.
#         Image.fromarray((max_FAG * 255).astype(np.ubyte)).save(
#             avg_max_result_save_path + FAG_data[-1]['fname'][:-7] + '_max.png')
#
#         # write maximum image.
#         Image.fromarray((sub_max2avg * 255).astype(np.ubyte)).save(
#             avg_max_result_save_path + FAG_data[-1]['fname'][:-7] + '_sub.png')
#
#         # Chamfer Matcging.
#         Chamfer_st = datetime.now()
#         translated_FAG, t, angle = chamfer_matching(max_FAG, FP_data['FPVPmap'])
#         Chamfer_ed = datetime.now()
#         Chamfer_t_list.append((Chamfer_ed-Chamfer_st).total_seconds())
#
#         # apply Chamfer matching result.
#         translation_matrix = np.float32([[1, 0, t[1]], [0, 1, t[0]]])
#         img_translation = cv2.warpAffine(avg_FAG, translation_matrix, (translated_FAG.shape[1], translated_FAG.shape[0]))
#         avg_FAG = skimage.transform.rotate(img_translation, angle, resize=False)
#
#         # only for showing Chamfer matching result image.
#         Chamfer_result_save_path = save_cur_laterality_path + intermedia_result_method_name[4]
#         mkdir(Chamfer_result_save_path)
#
#         # write Chamfer Matching result image.
#         Image.fromarray((translated_FAG * 255).astype(np.ubyte)).save(
#             Chamfer_result_save_path + FAG_data[-1]['fname'][:-7] + '_Chamfer.png')
#
#         # applied registration result to original image and write.
#         for cur_FAG_data in FAG_data:
#             FAG_img = cur_FAG_data['FAG_ORIGIN_BSP'].astype(np.ubyte)
#             img_translation = cv2.warpAffine(FAG_img, translation_matrix,
#                                              (translated_FAG.shape[1], translated_FAG.shape[0]))
#             rigid_registration_FAG = skimage.transform.rotate(img_translation, angle, resize=False)
#
#             FAG_VP_img = cur_FAG_data['BSP'].astype(np.ubyte)
#             img_translation = cv2.warpAffine(FAG_VP_img, translation_matrix,
#                                              (translated_FAG.shape[1], translated_FAG.shape[0]))
#             cur_FAG_data['global_FAGVP'] = skimage.transform.rotate(img_translation, angle, resize=False)
#
#             img_translation = cv2.warpAffine(cur_FAG_data['FOV'].astype(np.ubyte), translation_matrix,
#                                              (translated_FAG.shape[1], translated_FAG.shape[0]))
#             cur_FAG_data['FOV'] = skimage.transform.rotate(img_translation, angle, resize=False)
#
#             origin_FAG_img_save_path = Chamfer_result_save_path + 'Origin_FAG_domain/'
#             mkdir(origin_FAG_img_save_path)
#
#             Image.fromarray((rigid_registration_FAG*255).astype(np.ubyte)).save(
#                 origin_FAG_img_save_path + cur_FAG_data['fname'][:-4] + '_Chamfer.png')
#
#             cur_FAG_data['FAG_ORIGIN_Chamfer'] = rigid_registration_FAG
#             'Origin_FAG_domain'
#
#         # defomable registration for FP-FA with BSpline.
#         BSP_FPFA_st = datetime.now()
#         bsp = regist_BSpline(FP_data['FPVPmap'] * 255, translated_FAG * 255.)
#         regi_FAG2FP = bsp.do_registration()
#         BSP_FPFA_ed = datetime.now()
#         BSP_FPFA_t_list.append((BSP_FPFA_ed - BSP_FPFA_st).total_seconds())
#         regi_FAG2FP = np.array(Image.fromarray(regi_FAG2FP).resize([FP_data['FP_img'].shape[1], FP_data['FP_img'].shape[0]], Image.BILINEAR)).copy()
#         dens_disp, draw_disp = bsp.get_displacement_vector_field()
#
#         # apply registration.
#         registrated_avg_FAG = bsp.registrationFromMatrix(avg_FAG*255)
#         registrated_avg_FAG = np.array(Image.fromarray(registrated_avg_FAG).resize([FP_data['FP_img'].shape[1], FP_data['FP_img'].shape[0]])).astype(np.ubyte)
#
#         resized_FP_VPmap = np.array(
#             Image.fromarray((FP_data['FPVPmap'] * 255).astype(np.ubyte)).resize([FP_data['FP_img'].shape[1], FP_data['FP_img'].shape[0]], Image.BILINEAR))
#
#         # only for showing FP-FAG BSpline registration result image.
#         FAG2FP_BSpline_result_save_path = save_cur_laterality_path + intermedia_result_method_name[5]
#         mkdir(FAG2FP_BSpline_result_save_path)
#
#         # write FP-FAG registration result image.
#         Image.fromarray((regi_FAG2FP).astype(np.ubyte)).save(
#             FAG2FP_BSpline_result_save_path + FAG_data[-1]['fname'][:-7] + '_FAG2FP.png')
#
#         # write FP VPmap result image.
#         Image.fromarray(resized_FP_VPmap).save(
#             FAG2FP_BSpline_result_save_path + FAG_data[-1]['fname'][:-7] + '_FP_VPmap.png')
#
#         # write FP-FAG registration result image.
#         Image.fromarray(registrated_avg_FAG).save(
#             FAG2FP_BSpline_result_save_path + FAG_data[-1]['fname'][:-7] + '_FAG2FP_avg.png')
#
#         Image.fromarray(draw_disp).save(
#             FAG2FP_BSpline_result_save_path + FAG_data[-1]['fname'][:-7] + '_displacement_vector_field.png')
#
#         # aggregated FOV mask
#         # and applied registration result to original image and write.
#         registrated_FAG_FOV_set = []
#         for cur_FAG_data in FAG_data:
#             FAG_img = cur_FAG_data['FAG_ORIGIN_Chamfer']
#             FAG_img = bsp.registrationFromMatrix(FAG_img)
#             cur_FAG_data['FAG_ORIGIN_FAG2FP'] = FAG_img
#             FAG_VP_img = cur_FAG_data['global_FAGVP']
#             cur_FAG_data['global_FAGVP'] = bsp.registrationFromMatrix(FAG_VP_img)
#             registrated_FAG_FOV_set.append(bsp.registrationFromMatrix(cur_FAG_data['FOV']).astype(np.ubyte)*255)
#
#             origin_FAG_img_save_path = FAG2FP_BSpline_result_save_path + 'Origin_FAG_domain/'
#             mkdir(origin_FAG_img_save_path)
#
#             Image.fromarray((FAG_img*255).astype(np.ubyte)).save(
#                 origin_FAG_img_save_path + cur_FAG_data['fname'][:-4] + '_FAG2FP.png')
#
#         FAG_FOV_aggregation = np.max(np.array(registrated_FAG_FOV_set), axis=0)
#         FP_FAG_FOV = np.bitwise_and(FAG_FOV_aggregation, FP_data['FOV'])
#
#         ## post-processing ##
#         Post_st = datetime.now()
#         Origin_FAG = []
#         Origin_FAG_minmax = []
#         global_FAGVP = []
#         for cur_FAG_data in FAG_data:
#             FAG_img = cur_FAG_data['FAG_ORIGIN_FAG2FP']
#             FAG_img[FP_FAG_FOV == 0] = 0
#             Origin_FAG.append(FAG_img)
#             FAG_img = FAG_img.astype(np.float32)
#             Origin_FAG_minmax.append((FAG_img - FAG_img.min()) / (FAG_img.max() - FAG_img.min()))
#             global_FAGVP.append(cur_FAG_data['global_FAGVP'])
#
#         Origin_FAG = np.array(Origin_FAG)
#         Origin_FAG_minmax = np.array(Origin_FAG_minmax)
#
#         # search maximum enhanced FA frame
#         max_bright_idx = 0
#         max_bright = 0
#         for j in range(len(Origin_FAG)):
#             if max_bright < Origin_FAG[j].mean():
#                 max_bright = Origin_FAG[j].mean()
#                 max_bright_idx = j
#
#         # compute reverse frangi called vally detection in our papre.
#         gray_FP = np.array(Image.fromarray(FP_data['FP_img']).convert('L'))
#         gray_FP = gray_FP.astype(np.float32)
#         gray_FP -= gray_FP.min()
#         gray_FP /= gray_FP.max()
#         gray_FP = (gray_FP * 255).astype(np.ubyte)
#         reverse_frangi_FP = frangi(255 - gray_FP, scale_range=[1, 3], scale_step=1)
#         reverse_frangi_FP = (reverse_frangi_FP - reverse_frangi_FP.min()) / ((1e-5) - reverse_frangi_FP.min())
#         FAG = np.array(Image.fromarray((Origin_FAG[max_bright_idx]*255).astype(np.ubyte)).convert('L')).astype(np.float32)
#         FAG -= FAG.min()
#         FAG /= FAG.max()
#         FAG = (FAG*255).astype(np.ubyte)
#         reverse_frangi_FAG = frangi(FAG, scale_range=[1, 3], scale_step=1)
#         reverse_frangi_FAG = (reverse_frangi_FAG - reverse_frangi_FAG.min()) / ((5e-5) - reverse_frangi_FAG.min())
#
#         registrated_avg_FAG[FP_FAG_FOV == 0] = 0
#         regi_FAG2FP[FP_FAG_FOV == 0] = 0
#
#         # binary image with fixed threshold value.
#         bn_regi_FAG2FP = regi_FAG2FP >= (0.6*255.)
#
#         # hysteresis with skeletonization method.
#         tmp = regi_FAG2FP >= (0.1 * 255.)
#         dt = cv2.distanceTransform((tmp).astype(np.ubyte), cv2.DIST_L2, 3)
#         dt2 = cv2.distanceTransform((bn_regi_FAG2FP >= 0.75*255.).astype(np.ubyte), cv2.DIST_L2, 3)
#         rapidly_increase = (np.abs(dt-dt2) > 2)
#         thin = skimage.morphology.skeletonize(tmp).astype(np.float32)
#
#         # merge large vessel and thin vessel
#         FAG_thin = np.bitwise_and(dt <= 3, thin == 1)
#         FAG_combine = np.bitwise_or(bn_regi_FAG2FP, thin==1).astype(np.ubyte)*255
#
#         # prevent very closed vessel using vally detection result.
#         bn_frangi = (reverse_frangi_FAG > 0.3).astype(np.ubyte) * 255
#         bn_frangi[regi_FAG2FP >= 254] = 0
#         bn_res_frangi = FAG_combine.copy()
#         bn_res_frangi[bn_frangi == 255] = 0
#
#         # remove very small region.
#         cca = cv2.connectedComponentsWithStats(bn_res_frangi, connectivity=8)
#         remove_noise_img = np.zeros(bn_res_frangi.shape, dtype=np.ubyte)
#         for n in range(1, cca[0] + 1):
#             cur_label_img = cca[1] == n
#             n_pixel = cur_label_img.sum()
#             if n_pixel > 50:
#                 remove_noise_img += cur_label_img
#         remove_noise_img = remove_noise_img*255
#
#         Post_ed = datetime.now()
#         Post_t_list.append((Post_ed - Post_st).total_seconds())
#
#
#         ## write final results ##
#         Final_result_save_path = save_cur_laterality_path + intermedia_result_method_name[6]
#         mkdir(Final_result_save_path)
#
#         Image.fromarray((regi_FAG2FP).astype(np.ubyte)).save(
#             Final_result_save_path + FAG_data[-1]['fname'][:-7] + '_05_prob.png')
#
#         Image.fromarray((FAG_combine).astype(np.ubyte)).save(
#             Final_result_save_path + FAG_data[-1]['fname'][:-7] + '_04_binary.png')
#
#         overlay = FP_data['FP_img'].copy()
#         overlay[FAG_combine!=0,2] = 255
#         overlay[FP_FAG_FOV==0] = 0
#
#         Image.fromarray((overlay).astype(np.ubyte)).save(
#             Final_result_save_path + FAG_data[-1]['fname'][:-7] + '_03_overlay.png')
#
#         Image.fromarray(FP_data['FP_img'].astype(np.ubyte)).save(
#             Final_result_save_path + FAG_data[-1]['fname'][:-7] + '_02_.png')
#
#         Image.fromarray((FP_FAG_FOV).astype(np.ubyte)).save(
#             Final_result_save_path + FAG_data[-1]['fname'][:-7] + '_01_FOV.png')
#
#         Image.fromarray((reverse_frangi_FP*255).astype(np.ubyte)).save(
#             Final_result_save_path + FAG_data[-1]['fname'][:-7] + '_10_reverse_frangi_FP.png')
#
#         Image.fromarray((reverse_frangi_FAG * 255).astype(np.ubyte)).save(
#             Final_result_save_path + FAG_data[-1]['fname'][:-7] + '_11_reverse_frangi_FAG.png')
#
#         Image.fromarray((bn_res_frangi).astype(np.ubyte)).save(
#             Final_result_save_path + FAG_data[-1]['fname'][:-7] + '_6_binary_reverse_frnagi_.png')
#
#         overlay = FP_data['FP_img'].copy()
#         overlay[bn_res_frangi != 0, 2] = 255
#         overlay[FP_FAG_FOV == 0] = 0
#
#         Image.fromarray((overlay).astype(np.ubyte)).save(
#             Final_result_save_path + FAG_data[-1]['fname'][:-7] + '_07_binary_reverse_frnagi_overlay.png')
#
#         Image.fromarray((remove_noise_img).astype(np.ubyte)).save(
#             Final_result_save_path + FAG_data[-1]['fname'][:-7] + '_08_binary_remove_noise(CCA).png')
#
#         overlay = FP_data['FP_img'].copy()
#         overlay[remove_noise_img != 0, 2] = 255
#         overlay[FP_FAG_FOV == 0] = 0
#
#         Image.fromarray((overlay).astype(np.ubyte)).save(
#             Final_result_save_path + FAG_data[-1]['fname'][:-7] + '_09_binary_remove_noise(CCA)_overlay.png')
#
#         for cur_FAG_data in FAG_data:
#             FAG_img = cur_FAG_data['FAG_ORIGIN_FAG2FP']
#             FAG_img[FP_FAG_FOV==0] = 0
#
#             origin_FAG_img_save_path = Final_result_save_path + 'Origin_FAG_domain/'
#             mkdir(origin_FAG_img_save_path)
#
#             Image.fromarray((FAG_img*255).astype(np.ubyte)).save(
#                 origin_FAG_img_save_path + cur_FAG_data['fname'][:-4] + '_FAG2FP.png')
#
#             FAG_img_32f = FAG_img.astype(np.float32)
#
#         # recode running time.
#         et = datetime.now()
#         cur_seq_t = (et - st).total_seconds()
#         mean_sift_t = np.mean(np.array(SIFT_t_list))
#         mean_BSP_t = np.mean(np.array(BSP_t_list))
#         mean_Chamfer_t = Chamfer_t_list[0]
#         mean_BSP_FPFA_t = BSP_FPFA_t_list[0]
#         mean_Post_t = Post_t_list[0]
#
#         csv_writer.writerow([seq_dir+'_'+laterality_dir, cur_seq_t, mean_sift_t, mean_BSP_t, mean_Chamfer_t, mean_BSP_FPFA_t, mean_Post_t])
#         f.flush()
#
#     # end loop for image sequence directory.

from __future__ import print_function

import os
import torch
import utils_220208
import argparse
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt

import skimage
from skimage import morphology
import cv2
import csv, glob
import warnings
from skimage.filters import frangi
from datetime import datetime
warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
# make directory from path(same os.makedirs(path, exist_ok=True) on python 3.x)

parser = argparse.ArgumentParser(description='PyTorch Multimodal registration of fundus images With fluorescein angiography for fine-scale vessel segmentation')
parser.add_argument('--data_path', type=str, default='./data',
                    help='input path(default : ./data)')
parser.add_argument('--res_path', type=str, default='',
                    help='input path(default : ./res)')
parser.add_argument('--fp_model_path', type=str, default='',
                    help='fp model path(default : none)')
parser.add_argument('--fag_model_path', type=str, default='',
                    help='fp model path(default : none)')
args = parser.parse_args()
def mkdir(path):
    if os.path.exists(path) == False:
        os.mkdir(path)

# fixed vessel segmentation image size
HRF_size = [1024, 1536]
# set image path
# DB_path = '../DB/test/'
DB_path = args.data_path
result_path = args.res_path
os.makedirs(result_path, exist_ok = True)
# set out path
save_path = DB_path
mkdir(save_path)

# create deep learning model & load to saved data
# FP_model_weight_path = './DL_model/fundus/model_8000_iter_loss_84949.1797.pth.tar'
# FA_model_weight_path = './DL_model/FA/sub_mean_model_8400_iter_loss_35412.4364.pth.tar'
FP_model_weight_path = args.fp_model_path
FA_model_weight_path = args.fag_model_path
FAG_model, FP_model = utils_220208.get_model(FP_model_weight_path, FA_model_weight_path)

# save directory name list
intermedia_result_method_name = ['/1.SIFT/', '/2.VesselProb/', '/3.BSP_FAG/', '/4.Aggregation/', '/5.Chamfer/', '/6.BSP_FP-FAG/', '/7.Final/']


#### main ####
for seq_idx, seq_dir in enumerate(sorted(os.listdir(DB_path))): # loop for image sequence directory.
    cur_seq_path = DB_path + seq_dir + '/'

    if os.path.isdir(cur_seq_path) == False:
        continue

    img_path = DB_path + seq_dir
    # declare for each recording space.
    FAG_list_path = glob.glob(img_path + "/*FAG2FP.png") # FAG data.
    FAG_list = utils_220208.image_load(FAG_list_path)
    FP_path = glob.glob(img_path + "/*origin.png") # Fundus Photo data.
    FP = utils_220208.image_load(FP_path)[0]

    FOV_mask = utils_220208.make_FOV(FP)

    #FAG registration
    FAG_reg_class = utils_220208.FAG_registration(FAG_model, FAG_list, FOV_mask)
    FAagg_vessel_mean_map, FAagg_vessel_max_map = FAG_reg_class.all_registration()
    
    #FP registration
    FP_reg_class = utils_220208.FP_registration(FP_model, FP, FAagg_vessel_mean_map, FAagg_vessel_max_map, FOV_mask)
    Registrated_map = FP_reg_class.all_registration()

    #Mask between FP - FAG
    FAG_FOV_list = FAG_reg_class.nonrigid_reg_masks
    FAG_FOV_aggregation = np.max(np.array(FAG_FOV_list), axis=0)
    FP_FAG_FOV = np.bitwise_and(FAG_FOV_aggregation, FP_reg_class.main_FOV)

    Registrated_map[FP_FAG_FOV == 0] = 0
    result_path = os.makedirs(result_path + seq_dir + '/', exist_ok=True)
    Image.fromarray(Registrated_map).save(result_path + "result.png")

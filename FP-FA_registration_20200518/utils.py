import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from torch.autograd import Variable
from torchvision import models
from PIL import Image
from DL_model import FA
from DL_model import FA_SSANet3
from DL_model import fundus
from scipy import ndimage
import matplotlib.pyplot as plt

def get_pr_auc(gt, pred):
    precision, recall, _ = precision_recall_curve(gt, pred)
    pr_auc = auc(recall, precision)
    return precision,recall,pr_auc
def get_roc_auc(gt, pred):
    fpr, tpr, _ = roc_curve(gt, pred, pos_label=1.)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def FP_vessel_prediction2(model, FP_img, mask, size):
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

def FAG_vessel_prediction3(model, FAG_img, mask, size):
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
        FAG_data = Variable(data.cuda())
        out_FAG = model(FAG_data)
        pred_FAG = out_FAG.data.cpu()[0][0].numpy()

    pred_FAG = np.array(Image.fromarray((pred_FAG*255).astype(np.ubyte)).resize([FAG_img.shape[1], FAG_img.shape[0]], Image.BILINEAR))
    pred_FAG = pred_FAG.astype(np.float32)/255.
    return pred_FAG

def FAG_vessel_prediction_retrain_AVset(model, FAG_img, mask, size):
    model.eval()
    with torch.no_grad():
        # resized_FAG_img = np.array(Image.fromarray(FAG_img).resize(size=(size[1], size[0]), resample=Image.BILINEAR), dtype=np.float32)
        resized_FAG_img = FAG_img.astype(np.float32).copy()
        mask = np.around(np.array(Image.fromarray(mask).resize(size=(size[1], size[0]), resample=Image.BILINEAR), dtype=np.float32)/255).astype(np.ubyte)*255
        if len(resized_FAG_img.shape) == 2:
            resized_FAG_img = np.array(Image.fromarray(resized_FAG_img).convert('RGB'))

        resized_FAG_img = resized_FAG_img.transpose(2,0,1).copy()
        # mean = np.mean(resized_FAG_img)

        resized_FAG_img[0] -= np.mean(resized_FAG_img[0])
        resized_FAG_img[1] -= np.mean(resized_FAG_img[1])
        resized_FAG_img[2] -= np.mean(resized_FAG_img[2])

        data = torch.FloatTensor(1,3,size[0], size[1])
        data[0] = torch.from_numpy(resized_FAG_img)
        # data = torch.unsqueeze(data, 0)
        data = Variable(data.cuda(0))
        out_FAG = model(data)
        pred_FAG = out_FAG.data.cpu()[0][0].numpy()

    pred_FAG = np.array(Image.fromarray((pred_FAG*255).astype(np.ubyte)).resize([FAG_img.shape[1], FAG_img.shape[0]], Image.BILINEAR))
    pred_FAG = pred_FAG.astype(np.float32)/255.
    return pred_FAG

# get deep learning model
def get_model(FP_model_weight_path, FA_model_weight_path):
    pretraind_model1 = models.resnet34(pretrained=True)

    ## AS-IS model
    # FA_model = FA.SSA(pretraind_model1)
    # FA_model = FA_model.cuda(0)
    # FA_model.load_state_dict(torch.load(FA_model_weight_path))

    ## retrain on AV DB set
    # FA_model = FA.FA_SSANet3_version2(pretraind_model1)
    # FA_model = FA_SSANet3.AV_net()
    # FA_model = FA_model.cuda(0)
    # FA_model.load_state_dict(torch.load(FA_model_weight_path))
    FA_model = torch.load('/mnt/hdd/code/15_A-V/010_vessel_learning/001_/model_50000_iter_loss_65253.8840_acc_0.9315_vessel_acc_0.9952.pth.tar')
    FA_model = FA_model.cuda(0)

    pretraind_model2 = models.resnet34(pretrained=True)
    fundus_model = fundus.SSA(pretraind_model2)
    fundus_model = fundus_model.cuda(1)
    fundus_model.load_state_dict(torch.load(FP_model_weight_path, map_location='cuda:1'))

    return FA_model, fundus_model

def normalization(img):
    resized_img = np.array(Image.fromarray(img).resize([img.shape[1]/2, img.shape[0]/2], Image.BILINEAR))
    n = 101
    nh = int(np.floor(n / 2))
    bg_th = 4
    dbl_img = resized_img[:,:,0].astype(np.float32)
    img_h = dbl_img.shape[0]
    img_w = dbl_img.shape[1]
    nhood = np.ones([n, n])
    mean_img = ndimage.convolve(dbl_img, nhood / (n*n))
    # mean_img = mean_img[nh:img_h + nh, nh:img_w + nh]

    d_img = np.abs(dbl_img - mean_img)
    bg_lbl_img = d_img < bg_th

    nsum = np.zeros([img_h, img_w])
    nstd = np.ones([img_h, img_w])
    ncount = np.ones([img_h, img_w])
    nmean = np.zeros([img_h, img_w])
    dbl_img_sqr = dbl_img * dbl_img

    for y  in range(img_h):
        for x in range(img_w):
            p_bg = bg_lbl_img[y - nh:y + nh, x - nh:x + nh].copy()
            p_img = dbl_img[y - nh:y + nh, x - nh:x + nh].copy()
            # p_img_sqr = dbl_img_sqr[y - nh:y + nh, x - nh:x + nh]
            nsum[y, x] = np.multiply(p_bg, p_img).sum()
            ncount[y, x] = p_bg.sum()
            if ncount[y, x] ==0:
                nmean[y, x] = 0
            else:
                nmean[y, x] = nsum[y, x] / ncount[y, x]

    nstd[nstd == 0] = 1

    # padding = np.zeros([img_h, img_w])
    # padding[nh:(img_h - nh), nh:(img_w - nh)] = dbl_img[nh:(img_h - nh), nh:(img_w - nh)].copy()
    norm_img = (dbl_img - nmean)
    norm_img[norm_img < 0] = 0
    # norm_img_minmax = (norm_img - min(min(norm_img))) / (max(max(norm_img)) - min(min(norm_img))) * 255

    norm_img = np.array(Image.fromarray((norm_img).astype(np.ubyte)).resize([img.shape[1], img.shape[0]], Image.BILINEAR))
    return norm_img

import numpy as np
import cv2
import skimage

def chamfer_matching(img1, img2, which_img=0):
    # cost = np.ones([img1.shape[0], img1.shape[0], 5]) * 10000.
    all_cost = []
    index = []
    b_img2 = (img2 > 0.4).astype(np.ubyte)
    b_img2 = skimage.morphology.thin(b_img2.astype(np.ubyte)).astype(np.ubyte)
    dist_img2 = cv2.distanceTransform(1 - b_img2, cv2.DIST_L2, 3)

    for angle in range(-5, 5):
        b_img1 = (skimage.transform.rotate(img1, angle, resize=False) > 0.4).astype(np.ubyte)
        b_img1 = skimage.morphology.thin(b_img1.astype(np.ubyte)).astype(np.ubyte)
        dist_img1 = cv2.distanceTransform(1 - b_img1,cv2.DIST_L2,3)
        offset = [(img2.shape[0] - int((img2.shape[0] * 0.6))) // 2, (img2.shape[1] - int((img2.shape[1] * 0.6))) // 2]
        crop_dist_img2 = dist_img2[offset[0]:img2.shape[0] - offset[0], offset[1]:img2.shape[1] - offset[1]]

        # cv2.TM_SQDIFF
        # cv2.TM_SQDIFF_NORMED
        # res = cv2.matchTemplate(dist_img1.astype(np.float32), crop_dist_img2.astype(np.float32), cv2.TM_SQDIFF)
        # res = np.ones([dist_img1.shape[0], dist_img1.shape[1]])*10000
        # for y in range(0, dist_img1.shape[0]-crop_dist_img2.shape[0]):
        #     for x in range(0, dist_img1.shape[1]-crop_dist_img2.shape[1]):
        #         cur_dist_img1 = dist_img1[y:y+crop_dist_img2.shape[0], x:x+crop_dist_img2.shape[1]]
        #         res[y+crop_dist_img2.shape[0]/2, x+crop_dist_img2.shape[1]/2] = cur_dist_img1[crop_dist_img2==0].mean()

        res = cv2.filter2D(dist_img1, -1, (crop_dist_img2==0).astype(np.float32))

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        all_cost.append(min_val)
        index.append([min_loc[1],min_loc[0], angle])

    min_cost_pt = index[np.argmin(all_cost)]
    # t = [(offset[0]-min_cost_pt[0]), (offset[1]-min_cost_pt[1])]
    if which_img == 0:
        t = [dist_img2.shape[0]//2-min_cost_pt[0], dist_img2.shape[1]//2-min_cost_pt[1]]

        translation_matrix = np.float32([[1, 0, t[1]], [0, 1, t[0]]])
        img_translation = cv2.warpAffine(img1, translation_matrix, (img1.shape[1], img1.shape[0]))

        draw = skimage.transform.rotate(img_translation, min_cost_pt[2], resize=False)
        return draw, t, min_cost_pt[2]
    else:
        t = [-(dist_img2.shape[0] // 2 - min_cost_pt[0]), -(dist_img2.shape[1] // 2 - min_cost_pt[1])]

        translation_matrix = np.float32([[1, 0, t[1]], [0, 1, t[0]]])
        img_translation = cv2.warpAffine(img2, translation_matrix, (img2.shape[1], img2.shape[0]))

        draw = skimage.transform.rotate(img_translation, -min_cost_pt[2], resize=False)
        return draw, t, -min_cost_pt[2]
import numpy as np
import skimage
import skimage.transform
import skimage.morphology
import skimage.filters
import cv2
import matplotlib.pyplot as plt

class regist_SIFT():
    def __init__(self, img1, img2, mask1, mask2, hess_thresh=3):
        # super(regist_SIFT, self).__init__()
        self.img1 = img1
        self.img2 = img2
        # self.max_point_distance = 200 # set maximum distance of matching pair points.
        # self.im_out, self.h, self.sift_kpt_imgs, self.matching_img, self.kpt, self.des, self.pts \
        #     = self.do_registration(img1, img2, self.max_point_distance, False)
        #
        # return self.im_out, self.h, self.sift_kpt_imgs, self.matching_img, self.kpt, self.des, self.pts
        self.hess_thresh = hess_thresh
        self.mask1 = mask1
        self.mask2 = mask2
    def get_sift_kpt(self, img, mask, is_VPmap=False):
        if len(img.shape) == 3:
            if img.shape[2] == 3:
                gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray_img = img.reshape(img.shape[:-1])
        else:
            gray_img = img.copy()

        if is_VPmap == False:
            mask = skimage.morphology.erosion(mask).astype(np.ubyte)

        gray_img_norm = ((gray_img - gray_img.min()) / float(gray_img.max() - gray_img.min()) * 255.).astype(np.ubyte)
        # sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers=100)
        # sift = cv2.AKAZE_create(nOctaves = self.hess_thresh, nOctaveLayers = 10)
        # sift = cv2.AKAZE_create(nOctaveLayers = self.hess_thresh)
        sift = cv2.xfeatures2d.SURF_create(nOctaveLayers=self.hess_thresh)
        kps, descs = sift.detectAndCompute(gray_img_norm, mask=mask)
        sift_kpt_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

        cv2.drawKeypoints(gray_img_norm, kps, sift_kpt_img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.imwrite('sift_elecs1.jpg', sift_kpt_img)

        return kps, descs, sift_kpt_img

    def matching(self, kps1, descs1, kps2, descs2, mask1, mask2, max_pt_dist=500):
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descs1, descs2, k=2)
        # Apply ratio test
        good = []
        for m, n in matches:
            pt_dist = np.sqrt(
                (kps1[m.queryIdx].pt[0] - kps2[m.trainIdx].pt[0]) * (kps1[m.queryIdx].pt[0] - kps2[m.trainIdx].pt[0]) + \
                (kps1[m.queryIdx].pt[1] - kps2[m.trainIdx].pt[1]) * (kps1[m.queryIdx].pt[1] - kps2[m.trainIdx].pt[1]))
            is_mask1 = mask1[int(kps1[m.queryIdx].pt[1]), int(kps1[m.queryIdx].pt[0])]
            is_mask2 = mask2[int(kps1[m.queryIdx].pt[1]), int(kps1[m.queryIdx].pt[0])]
            # if m.distance < 0.7 * n.distance and pt_dist < max_pt_dist \
            #         and is_mask1 == 1 and is_mask2 == 1:
            if m.distance < 0.7 * n.distance and is_mask1 != 0 and is_mask2 != 0 and pt_dist < max_pt_dist:
                good.append([m])

        return good

    def do_registration(self, is_VPmap=False, max_pt_dist=300, intermedia_out=False):
        mask1 = self.mask1.copy()
        mask2 = self.mask2.copy()
        kps1, descs1, sift_kpt_img1 = self.get_sift_kpt(self.img1, mask1, is_VPmap)
        kps2, descs2, sift_kpt_img2 = self.get_sift_kpt(self.img2, mask2, is_VPmap)

        # if is_VPmap == False:
        #     mask1 = skimage.morphology.erosion((self.img1[:, :, 0] > 5), skimage.morphology.square(11)).astype(np.float32)
        #     mask2 = skimage.morphology.erosion((self.img2[:, :, 0] > 5), skimage.morphology.square(11)).astype(np.float32)
        # else:
        #     mask1 = np.ones([self.img1.shape[0], self.img1.shape[1]])
        #     mask2 = np.ones([self.img1.shape[0], self.img1.shape[1]])

        if intermedia_out == True:
            cv2.imwrite('sift_kpt_img1.jpg', sift_kpt_img1)
            cv2.imwrite('sift_kpt_img2.jpg', sift_kpt_img2)

        good = self.matching(kps1, descs1, kps2, descs2, mask1, mask2, max_pt_dist)

        # cv2.drawMatchesKnn expects list of lists as matches.
        matching_img = cv2.drawMatchesKnn(self.img1, kps1, self.img2, kps2, good, flags=2, outImg=None)
        if intermedia_out == True:
            cv2.imwrite('matching.jpg', matching_img)

        # print('number of matching points : %d' % (len(good)))
        pts_src = []
        pts_dst = []

        new_matching_img = np.zeros([self.img1.shape[0]*2, self.img1.shape[1], 3], dtype=np.ubyte)
        new_matching_img[0:self.img1.shape[0], :] = self.img1.copy()
        new_matching_img[self.img1.shape[0]:, :] = self.img2.copy()
        for i, match in enumerate(good):
            pts_src.append([kps1[match[0].queryIdx].pt[0], kps1[match[0].queryIdx].pt[1]])
            pts_dst.append([kps2[match[0].trainIdx].pt[0], kps2[match[0].trainIdx].pt[1]])

            if i < 10:
                rand_color = np.random.random_sample(3)
                rand_color[0] = int(rand_color[0] * 255)
                rand_color[1] = int(rand_color[1] * 255)
                rand_color[2] = int(rand_color[2] * 255)
                # rand_color = rand_color.astype(np.int)
                pt1 = (int(np.around(pts_src[-1][0])), int(np.around(pts_src[-1][1])))
                pt2 = (int(np.around(pts_dst[-1][0])), int(self.img1.shape[0]+np.around(pts_dst[-1][1])))
                cv2.line(new_matching_img, pt1, pt2, tuple(rand_color), 2)

        new_matching_img_row = cv2.drawMatchesKnn(self.img1, kps1, self.img2, kps2, good, flags=2, outImg=None)

        pts_src = np.array(pts_src)
        pts_dst = np.array(pts_dst)

        if len(pts_src) < 30 or len(pts_dst) < 30:
            return False, [[],[],[],[],[],[],[],[],[]]

        try:
            h, status = cv2.findHomography(pts_dst, pts_src, cv2.RANSAC)
            im_out = cv2.warpPerspective(cv2.cvtColor(self.img2, cv2.COLOR_RGB2BGR), h,
                                         (self.img1.shape[1], self.img1.shape[0]))
        except:
            return False, [[], [], [], [], [], [], [], [], []]

        # move = np.matmul(h, np.array([int(self.img1.shape[1]/2), int(self.img1.shape[0]/2), 1])).astype(np.int16)

        return True, [im_out, h, [sift_kpt_img1, sift_kpt_img2], matching_img, [kps1, kps2], [descs1, descs2], [pts_src,
                                                                                                         pts_dst], new_matching_img, new_matching_img_row]


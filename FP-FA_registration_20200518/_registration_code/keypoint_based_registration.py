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


## just made function before
    # def get_file_path(root_path):
    #     path_list = []
    #     dir_list = []
    #     for dir in sorted(os.listdir(root_path)):
    #         cur_seq_fundus_path = ''
    #         cur_seq_fundus_name = ''
    #         cur_seq_FA_path_list = []
    #         cur_seq_FA_file_name_list = []
    #         cur_seq_FA_dir_name_list = []
    #         for fname in os.listdir(root_path+dir+'/origin/'):
    #             if fname.find('RGB')!=-1:
    #                 cur_seq_fundus_path = root_path+dir+'/origin/'+fname
    #                 cur_seq_fundus_name = fname
    #             elif fname.find('fundus')==-1:
    #                 cur_seq_FA_path_list.append(root_path+dir+'/origin/'+fname)
    #                 cur_seq_FA_file_name_list.append(fname)
    #                 cur_seq_FA_dir_name_list.append(dir)
    #
    #         cur_seq_FA_path_list = np.sort(cur_seq_FA_path_list)
    #         cur_seq_FA_file_name_list = np.sort(cur_seq_FA_file_name_list)
    #
    #         path_list.append([cur_seq_fundus_path, cur_seq_fundus_name, cur_seq_FA_path_list, \
    #                           cur_seq_FA_file_name_list, cur_seq_FA_dir_name_list])
    #
    #     return path_list
    #
    # def read_image(path_list):
    #     img_array = []
    #     for i in path_list:
    #         cur_seq_img_array = []
    #
    #         for j in i[2]:
    #             img = Image.open(j)
    #             cur_seq_img_array.append(np.asarray(img, dtype=np.ubyte))
    #         cur_seq_img_array = np.array(cur_seq_img_array)
    #         cur_seq_img_array[:, :140, -400:, :] = 0  # sample1
    #
    #         # fundus
    #         fundus_img = Image.open(i[0])
    #         fundus_img = (np.asarray(fundus_img, dtype=np.ubyte))
    #         if fundus_img.shape[0] != cur_seq_img_array[0].shape[0] or fundus_img.shape[1] != cur_seq_img_array[0].shape[1]:
    #             fundus_img = Image.fromarray(fundus_img)
    #             fundus_img = fundus_img.resize((cur_seq_img_array[0].shape[1], cur_seq_img_array[0].shape[0]))
    #             fundus_img = np.asarray(fundus_img)
    #         img_array.append([fundus_img, cur_seq_img_array])
    #
    #     # img_array[:, :140, -350:, :] = 0 # sample2
    #
    #     return img_array
    #
    # def compute_score(data):
    #     rot_crop_tar = rotate(data[1], angle=data[2], axes=(0, 1), reshape=False, order=0)
    #     score = rot_crop_tar[data[0] == 1].sum()
    #     return score
    # def chamfer_matching(src, tar):
    #     tmp = np.zeros(src.shape)
    #     nz = np.nonzero(src)
    #     LT = [nz[1].min()+300, nz[0].min()+300]
    #     RB = [nz[1].max()-300, nz[0].max()-300]
    #     center = [LT[0]+(RB[0]-LT[0])/2, LT[1]+(RB[1]-LT[1])/2,]
    #     crop = src[LT[1]:RB[1], LT[0]:RB[0]].copy()
    #
    #     n_core = 10
    #     pool = Pool(processes=n_core)
    #
    #     best_score = 0
    #     t = [0, 0]
    #     max_angle = 3
    #     angles = range(-max_angle, max_angle, 1)
    #     score_map = np.zeros([tar.shape[0], tar.shape[1], angles.__len__()])
    #
    #     for y in range(LT[1]-100,LT[1]+100,2):
    #         data = []
    #         # for x in range(tar.shape[1] - crop.shape[1]):
    #         for x in range(LT[0]-50,LT[0]+50,2):
    #             crop_tar = tar[y:y + crop.shape[0], x:x + crop.shape[1], 0].copy()
    #             for r, angle in enumerate(angles):
    #                 data.append([crop.copy(), crop_tar.copy(), angle, y, x, r])
    #                 # rot_crop_tar = rotate(crop_tar, angle=angle, axes=(0, 1), reshape=False, order=0)
    #                 # score = rot_crop_tar[crop==1].sum()
    #         score = pool.map(compute_score, data)
    #         for i in range(data.__len__()):
    #             score_map[data[i][-3], data[i][-2], data[i][-1]] = score[i]
    #
    #
    #
    #                 # if best_score<score:
    #                 #     best_score = score
    #                 #     t = [y, x, angle]
    #
    #     best_score = np.max(score_map)
    #     idx = np.unravel_index(score_map.argmax(), score_map.shape)
    #     t = [idx[0], idx[1], angles[idx[2]]]
    #     t = [t[0]-LT[1], t[1]-LT[0], angle]
    #
    #     nz2 = [nz[0]+t[0], nz[1]+t[1] ]
    #
    #     tmp = np.zeros([tar.shape[0], tar.shape[1]], dtype=np.ubyte)
    #     for i in range(len(nz2[0])):
    #         if nz2[0][i]>0 and nz2[1][i]>0 and nz2[0][i] <tar.shape[0]and nz2[1][i] <tar.shape[1]:
    #             tmp[nz2[0][i], nz2[1][i]] = 255
    #     rot_tmp = np.round(rotate(tmp, angle=-t[2], axes=(0, 1), reshape=False))
    #     tmp = tar.copy()
    #     tmp[rot_tmp>=100, 0] = 255
    #     result = Image.fromarray(tmp)
    #     result.save('out.bmp')
    #
    #     t_tar = np.zeros([tar.shape[0], tar.shape[1]], dtype=np.ubyte)
    #     for y in range(t_tar.shape[0]):
    #         for x in range(t_tar.shape[1]):
    #             # print(y-t[0], x-t[1])
    #             if y+t[0] >=0 and x+t[1] >=0 and y+t[0] <tar.shape[0] and x+t[1]<tar.shape[1]:
    #                 t_tar[y, x] = tar[y+t[0], x+t[1],0]
    #     t_tar = rotate(t_tar, angle=t[2], axes=(0, 1), reshape=False)
    #     result = Image.fromarray(t_tar)
    #     result.save('out2.bmp')
    #
    #     pool.close()
    #     pool.join()
    #
    #     return t
    #
    # def ncc(arr_img, patch_sz):
    #     n_img = arr_img.shape[0]
    #     w = arr_img.shape[2]
    #     h = arr_img.shape[1]
    #
    #     score = np.zeros([arr_img.shape[0]-1, arr_img.shape[1], arr_img.shape[2]])
    #     for y in range(patch_sz/2, h-patch_sz/2-1, 4):
    #         for x in range(patch_sz/2, w-patch_sz/2-1, 4):
    #             for i in range(n_img-1):
    #                 crop1 = arr_img[i,y-patch_sz/2:y+patch_sz/2, x-patch_sz/2:x+patch_sz/2, 0]
    #                 crop2 = arr_img[i+1,y-patch_sz/2:y+patch_sz/2, x-patch_sz/2:x+patch_sz/2, 0]
    #
    #                 a = patch_sz*patch_sz*(crop1*crop2).sum()-float(crop1.sum()*crop2.sum())
    #                 b = np.sqrt((patch_sz*patch_sz*(crop1*crop1).sum()-float(crop1.sum()*crop1.sum()))*\
    #                             (patch_sz*patch_sz*(crop2*crop2).sum()-float(crop2.sum()*crop2.sum())))
    #                 if b == 0:
    #                     ncc_score = 0.
    #                 else:
    #                     ncc_score = a / b
    #
    #                 score[i, y, x] = ncc_score
    #
    #     for i,img in enumerate(score):
    #         eq_img = ((img-img.min())/(img.max()-img.min())*255).astype(np.ubyte)
    #         cv2.imwrite('test%02d.png'%(i), eq_img)
    #
    # def kmenas(arr_img):
    #     n_img = arr_img.shape[0]
    #     w = arr_img.shape[2]
    #     h = arr_img.shape[1]
    #
    #     norm_arr_img = arr_img.copy()
    #     for i in range(n_img):
    #         norm_arr_img[i] = (arr_img[i]-arr_img[i].min())/(arr_img[i].max()-arr_img[i].min())*255.
    #
    #     feat = np.zeros([arr_img.shape[1]*arr_img.shape[2], n_img], dtype=np.float32)
    #     pts = np.zeros([arr_img.shape[1]*arr_img.shape[2], n_img*2, ], dtype=np.int)
    #
    #     # plt.subplot('131')
    #     # plt.plot(range(norm_arr_img.shape[0]), norm_arr_img[:, 892, 1237, 0])
    #     # plt.ylim([0,255])
    #     # plt.subplot('132')
    #     # plt.plot(range(norm_arr_img.shape[0]), norm_arr_img[:, 760, 1141, 0])
    #     # plt.ylim([0, 255])
    #     # plt.subplot('133')
    #     # plt.plot(range(norm_arr_img.shape[0]), norm_arr_img[:, 242, 708, 0])
    #     # plt.ylim([0, 255])
    #     # plt.show()
    #     for y in range(0,h):
    #         for x in range(0,w):
    #             # for i in range(n_img):
    #             feat[y*w+x] = arr_img[:, y, x, 0]
    #             pts[y*w+x,0] = y
    #             pts[y * w + x, 1] = x
    #     # define criteria and apply kmeans()
    #     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100000, 1.0)
    #     ret, label, center = cv2.kmeans(feat, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    #
    #     draw = np.zeros([h,w,3], dtype=np.ubyte)
    #     rand_color = np.random.uniform(0,1,[5,3])*255.
    #     for i, la in enumerate(label):
    #         y=pts[i, 0]
    #         x=pts[i, 1]
    #         a = rand_color[int(la[0])]
    #         draw[y,x,0] = a[0]
    #         draw[y, x, 1] = a[1]
    #         draw[y, x, 2] = a[2]
    #     cv2.imwrite('label.png', draw)
    #     a= 0
    #
    # def do_distortion(imgs):
    #     distortion_imgs = []
    #     for cur_fundus, fa_imgs in imgs:
    #
    #         distCoeff = np.zeros((4, 1), np.float64)
    #
    #         # TODO: add your coefficients here!
    #         k1 = -9 * 1e-8;  # negative to remove barrel distortion
    #         # k1 = 0.0;  # negative to remove barrel distortion
    #         k2 = 0.0;
    #         p1 = 0.0;
    #         p2 = 0.0;
    #
    #         distCoeff[0, 0] = k1;
    #         distCoeff[1, 0] = k2;
    #         distCoeff[2, 0] = p1;
    #         distCoeff[3, 0] = p2;
    #
    #         # assume unit matrix for camera
    #         cam = np.eye(3, dtype=np.float32)
    #
    #         cam[0, 2] = fa_imgs.shape[2] / 2.0 # define center x
    #         cam[1, 2] = fa_imgs.shape[1] / 2.0 # define center y
    #         cam[0, 0] = 1.  # define focal length x
    #         cam[1, 1] = 1.  # define focal length y
    #
    #         n_img = fa_imgs.shape[0]
    #
    #         distortion_FA_imgs = []
    #
    #         distortion_fundus_img = cv2.undistort(cur_fundus, cam, distCoeff)
    #
    #         for cur_img in fa_imgs:
    #             distored = cv2.undistort(cur_img, cam, distCoeff)
    #             distortion_FA_imgs.append(distored)
    #
    #         distortion_FA_imgs = np.asarray(distortion_FA_imgs)
    #         distortion_imgs.append([distortion_fundus_img, distortion_FA_imgs])
    #     return distortion_imgs

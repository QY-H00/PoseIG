import numpy as np
import torchvision.transforms as transforms
import cv2, torch
from torch.utils.data import Dataset
import math,os,json

class BaseDataset(Dataset):
    def __init__(self, path_name, ds_name, mode, input_size, output_size, bbs_scale, output_full, aug):
        super(BaseDataset, self).__init__()
        assert mode in ['training','evaluation'], 'mode should be training/evaluation'
        self.mode=mode
        self.output_size = output_size
        self.input_size = input_size
        self.path_name = path_name
        self.output_full = output_full
        self.bbs_scale = bbs_scale
        self.ds_name = ds_name
        self.aug = aug
        self.joint_num = 21
        self.image_trans = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),])
        if self.aug:
            self.image_trans =transforms.Compose([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                                     transforms.RandomErasing(p=0.75, scale=(0.02, 0.1)),
                                                     ])

    def get_rand(self, scale=(0.8,1.0), trans = (-5.,5), rotate = 0.05):
        randScaleImage = np.random.uniform(low=scale[0], high=scale[1])
        randAngle = rotate * 2 * math.pi * np.random.rand(1)[0]
        randTransX = np.random.uniform(low=trans[0], high=trans[1])
        randTransY = np.random.uniform(low=trans[0], high=trans[1])
        return randScaleImage,randAngle,randTransX,randTransY

    def augmentation(self, image, hand_mask, uv, pose3d_normed):
        randScaleImage, randAngle, randTransX, randTransY = self.get_rand()

        rot_mat_input = cv2.getRotationMatrix2D((self.input_size/2, self.input_size/2), -180.0 * randAngle/ math.pi, randScaleImage)
        rot_mat_input[0,2] += randTransX*self.input_size/self.output_size
        rot_mat_input[1,2] += randTransY*self.input_size/self.output_size

        rot_mat_output = cv2.getRotationMatrix2D((self.output_size/2, self.output_size/2), -180.0 * randAngle/ math.pi, randScaleImage)
        rot_mat_output[0,2] += randTransX
        rot_mat_output[1,2] += randTransY

        image_aug = image.copy()
        mask_aug = hand_mask.copy()
        uv_aug = np.ones([uv.shape[0], uv.shape[1] + 1])
        uv_aug[:, :2] = uv

        image_aug = cv2.warpAffine(image_aug, rot_mat_input, (self.input_size, self.input_size), flags=cv2.INTER_NEAREST, borderValue=0.0)
        mask_aug = cv2.warpAffine(mask_aug, rot_mat_output, (self.output_size, self.output_size), flags=cv2.INTER_NEAREST, borderValue=0.0)
        uv_aug = np.dot(rot_mat_output, uv_aug.T).T

        # get rot_mat_inv
        rot_mat_inv = np.eye(3)
        rot_mat_inv[:2, :] = rot_mat_output
        rot_mat_inv = np.linalg.inv(rot_mat_inv.T)
        rot_mat_inv = rot_mat_inv[:, :2]

        (pose3d_normed[:, 0], pose3d_normed[:, 1]) = \
            self.rotate((0, 0), (pose3d_normed[:, 0], pose3d_normed[:, 1]), randAngle)

        # how to reverse uv_aug to uv_original
        # uv_aug_reverse = np.ones([uv.shape[0], uv.shape[1] + 1])
        # uv_aug_reverse[:, :2] = uv_aug
        # uv_aug_reverse = np.dot(uv_aug_reverse,rot_mat_inv)

        return image_aug, mask_aug, uv_aug, rot_mat_output, rot_mat_inv, pose3d_normed

    def imcrop(self, img, center, crop_size):
        x1 = int(center[0] - crop_size)
        y1 = int(center[1] - crop_size)
        x2 = int(center[0] + crop_size)
        y2 = int(center[1] + crop_size)

        if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
            img, x1, x2, y1, y2 = self.pad_img_to_fit_bbox(img, x1, x2, y1, y2)

        if img.ndim < 3:
            img_crop = img[y1:y2, x1:x2]
        else:
            img_crop = img[y1:y2, x1:x2, :]

        return img_crop

    def pad_img_to_fit_bbox(self, img, x1, x2, y1, y2):
        if img.ndim < 3:  # for depth
            borderValue = [0]
        else:  # for rgb
            borderValue = [127, 127, 127]

        img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0),
                                 -min(0, x1), max(x2 - img.shape[1], 0), cv2.BORDER_CONSTANT, value=borderValue)
        y2 += -min(0, y1)
        y1 += -min(0, y1)
        x2 += -min(0, x1)
        x1 += -min(0, x1)
        return img, x1, x2, y1, y2

    def creat_refine_uv(self, kp_coord_uv, crop_center, crop_size, crop_scale):
        keypoint_uv21_u = (kp_coord_uv[:, 0] - crop_center[0]) * crop_scale + crop_size // 2
        keypoint_uv21_v = (kp_coord_uv[:, 1] - crop_center[1]) * crop_scale + crop_size // 2
        coords_uv = np.stack([keypoint_uv21_u, keypoint_uv21_v], 1)
        return coords_uv

    def rotate(self, origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.
        The angle should be given in radians.
        """
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

        return qx, qy

    def depth2mask(self, depth, threshold):
        l, r = threshold[0], threshold[1]
        maskSingleHand = np.ones_like(depth).astype(np.float)

        maskSingleHand[depth > r] = 0
        maskSingleHand[depth < l] = 0

        return maskSingleHand

    def getTemplate_pose(self, joint_gt):
        device = joint_gt.device
        OriManotempJ = joint_gt.clone().reshape(-1, 21, 3).to(device)
        manotempJ = OriManotempJ.clone()
        manoidx = [2, 3, 17, 5, 6, 18, 8, 9, 20, 11, 12, 19, 14, 15, 16]
        manopdx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        manoppx = [0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]
        for idx in range(len(manoidx)):
            ci = manoidx[idx]
            pi = manopdx[idx]
            ppi = manoppx[idx]
            dp = torch.norm(OriManotempJ[:, ci] - OriManotempJ[:, pi], dim=-1, keepdim=True)
            dm = torch.norm(OriManotempJ[:, pi] - OriManotempJ[:, ppi], dim=-1, keepdim=True)
            manotempJ[:, ci] = manotempJ[:, pi] + (manotempJ[:, pi] - manotempJ[:, ppi]) / dm * dp

        return manotempJ

    def getTemplateFromBoneLen(self, boneLen, manoTemplate):
        if (torch.is_tensor(boneLen)): boneLen = boneLen.clone().detach().cpu().numpy().copy()
        if (torch.is_tensor(manoTemplate)): manoTemplate = manoTemplate.clone().detach().cpu().numpy().copy()

        boneLen = boneLen.copy().reshape(20)
        boneLen = np.concatenate([[0], boneLen], axis=0).reshape(21)
        OriManotempJ = manoTemplate.reshape(21, 3)
        manotempJ = OriManotempJ.copy()

        boneidxpalm = [1, 5, 9, 13, 17]
        manopalm = [1, 4, 7, 10, 13]
        for i in range(5):
            ci = manopalm[i]
            dm = np.linalg.norm(OriManotempJ[ci] - OriManotempJ[0]) + 1e-8
            manotempJ[ci] = manotempJ[0] + (OriManotempJ[ci] - OriManotempJ[0]) / dm * boneLen[boneidxpalm[i]]

        manoidx = [2, 3, 17, 5, 6, 18, 8, 9, 20, 11, 12, 19, 14, 15, 16]
        manopdx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        manoppx = [0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]
        boneidxfinger = [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]

        for idx in range(len(manoidx)):
            ci = manoidx[idx]
            pi = manopdx[idx]
            ppi = manoppx[idx]
            dp = boneLen[boneidxfinger[idx]] + 1e-8
            dm = np.linalg.norm(OriManotempJ[pi] - OriManotempJ[ppi]) + 1e-8
            manotempJ[ci] = manotempJ[pi] + (manotempJ[pi] - manotempJ[ppi]) / dm * dp

        return manotempJ

    def segment_hand(self, *args, **kwargs):
        raise NotImplementedError

    def generate_target(self, joints, joints_vis, heatmap_size, sigma, image_size):
        """Generate heatamap for joints.
        Args:
            joints: (K, 2)
            joints_vis: (K, 1)
            heatmap_size: W, H
            sigma:
            image_size:
        Returns:
        """
        num_joints = joints.shape[0]
        target_weight = np.ones((num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        target = np.zeros((num_joints,
                           heatmap_size[1],
                           heatmap_size[0]),
                          dtype=np.float32)

        tmp_size = sigma * 3
        image_size = np.array(image_size)
        heatmap_size = np.array(heatmap_size)

        for joint_id in range(num_joints):
            feat_stride = image_size / heatmap_size
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if mu_x >= heatmap_size[0] or mu_y >= heatmap_size[1] \
                    or mu_x < 0 or mu_y < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue

            # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight

    def keypoint2d_to_3d(self, keypoint2d: np.ndarray, intrinsic_matrix: np.ndarray, Zc: np.ndarray):
        """Convert 2D keypoints to 3D keypoints"""
        uv1 = np.concatenate([np.copy(keypoint2d), np.ones((keypoint2d.shape[0], 1))],
                             axis=1).T * Zc  # 3 x NUM_KEYPOINTS
        xyz = np.matmul(np.linalg.inv(intrinsic_matrix), uv1).T  # NUM_KEYPOINTS x 3
        return xyz

    def keypoint3d_to_2d(self, keypoint3d: np.ndarray, intrinsic_matrix: np.ndarray):
        """Convert 3D keypoints to 2D keypoints"""
        keypoint2d = np.matmul(intrinsic_matrix, keypoint3d.T).T  # NUM_KEYPOINTS x 3
        keypoint2d = keypoint2d[:, :2] / keypoint2d[:, 2:3]  # NUM_KEYPOINTS x 2
        return keypoint2d

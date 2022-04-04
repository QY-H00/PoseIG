import os
import numpy as np
import cv2
import pickle
import torch
import imageio
from PIL import Image
from dataset.util import RHD2Bighand_skeidx, root_idx, wrist_idx, norm_idx
from dataset.base_dataset import BaseDataset

class RHDDateset(BaseDataset):
    def __init__(self, path_name, mode, input_size = 64, output_size = 64, bbs_scale = 1.1, output_full = False, aug = False):
        super().__init__(path_name, 'rhd', mode, input_size, output_size, bbs_scale, output_full, aug)

        with open(os.path.join(path_name + self.mode, 'anno_%s.pickle' % self.mode), 'rb') as fi:
            self.anno_all = pickle.load(fi)
            self.num_samples = len(self.anno_all)

    def __len__(self):
        return int(self.num_samples)

    def __getitem__(self, idx):
        if (idx == 20500 or idx == 28140): idx = 0

        anno = self.anno_all[idx]
        image = imageio.imread(os.path.join(self.path_name, self.mode, 'color', '%.5d.png' % idx))
        mask = imageio.imread(os.path.join(self.path_name, self.mode, 'mask', '%.5d.png' % idx))
        uv = anno['uv_vis'][:, :2]
        kp_visible = anno['uv_vis'][:, 2] == 1
        pose3d = anno['xyz']
        camera = anno['K']

        hand_image, hand_mask, new_uv, crop_center, crop_size, hand_side, uv_vis, \
        relative_depth, pose3d_root, bone_length, pose3d, pose_uv_all, pose3d_normed\
            = self.segment_hand(image, mask, uv, kp_visible, pose3d)
        image = np.array(image, dtype=np.float)

        rot_mat_inv = np.eye(3, 2)
        rot_mat = np.eye(2, 3)
        hand_image_aug, hand_mask_aug, new_uv_aug, pose3d_normed_aug \
            = hand_image.copy(), hand_mask.copy(),new_uv.copy(), pose3d_normed.copy()

        if self.aug:
            hand_image_aug, hand_mask_aug, new_uv_aug, rot_mat, rot_mat_inv, pose3d_normed_aug \
                = self.augmentation(hand_image, hand_mask, new_uv, pose3d_normed)

        hand_image_tensor = Image.fromarray(hand_image_aug)
        hand_image_tensor = self.image_trans(hand_image_tensor).float()

        target = {}
        target['new_uv'] = torch.from_numpy(new_uv_aug).view(21,2).float()
        target['uv_vis'] = torch.from_numpy(uv_vis).view(21).float()
        target['relative_depth'] = torch.from_numpy(relative_depth).view(21,1).float()
        target['mask'] = torch.as_tensor(hand_mask_aug).view(1,self.output_size,self.output_size).float()
        target['confidence'] = torch.ones(1).view(1).float()

        if self.output_full:
            target['camera_type'] = 'perspective'  # orthographic
            target['pose3d_normed'] = torch.from_numpy(pose3d_normed_aug).float()
            target['rot_mat_inv'] = torch.from_numpy(rot_mat_inv).float()
            target['rot_mat'] = torch.from_numpy(rot_mat).float()
            target['pose3d'] = torch.from_numpy(pose3d).float()
            target['pose_uv_all'] = torch.from_numpy(pose_uv_all).float()
            target['crop_center'], target['crop_size'], target['hand_side'], target['pose3d_root'], target['bone_length'], target['camera'] = \
                torch.from_numpy(crop_center).float(),torch.as_tensor(crop_size).float(),\
                torch.as_tensor(hand_side).bool(),torch.as_tensor(pose3d_root).float(),torch.as_tensor(bone_length).float(),\
                torch.from_numpy(camera).float()

        return self.ds_name, idx, hand_image_tensor, target

    def segment_hand(self, image, mask, kp_coord_uv, kp_visible, pose3d):
        cond_l = np.logical_and(mask > 1, mask < 18)
        cond_r = mask > 17
        num_px_left_hand = np.sum(cond_l)
        num_px_right_hand = np.sum(cond_r)
        hand_side = num_px_left_hand > num_px_right_hand

        ### crop image
        if hand_side:
            pose_uv_all = kp_coord_uv[:21, :]
            uv_vis = kp_visible[:21]
            pose3d = pose3d[:21]
        else:
            pose_uv_all = kp_coord_uv[-21:, :]
            uv_vis = kp_visible[-21:]
            pose3d = pose3d[-21:]

        if hand_side:
            maskSingleHand = cond_l.astype(np.float)
        else:
            maskSingleHand = cond_r.astype(np.float)

        ### RHD2bighand idx
        pose3d = pose3d[RHD2Bighand_skeidx,:]
        pose_uv_all = pose_uv_all[RHD2Bighand_skeidx,:]
        uv_vis = uv_vis[RHD2Bighand_skeidx]

        crop_center = pose_uv_all[root_idx, :]
        crop_center = np.reshape(crop_center, 2)

        pose_uv_vis = pose_uv_all[uv_vis, :]
        crop_size = np.max(np.absolute(pose_uv_vis - crop_center))
        crop_size = np.minimum(np.maximum(crop_size, 25.0), 200.0)
        crop_size = np.ceil(crop_size) * self.bbs_scale

        # print crop_size
        image_crop = self.imcrop(image, crop_center, crop_size)
        mask_crop = self.imcrop(maskSingleHand, crop_center, crop_size)

        # resize image to the input size of hourglass
        height, width, _ = image.shape
        refine_size = self.output_size
        image_crop = cv2.resize(image_crop, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
        mask_crop = cv2.resize(mask_crop, (refine_size, refine_size), interpolation=cv2.INTER_NEAREST)

        crop_scale = refine_size / (crop_size * 2)
        new_uv = self.creat_refine_uv(pose_uv_all, crop_center, refine_size, crop_scale)


        ### get relative depth
        pose3d_root = pose3d[root_idx, :]  # this is the root coord
        pose3d_rel = pose3d - pose3d_root  # relative coords in metric coords
        bone_length = np.sqrt(np.sum(np.square(pose3d_rel[root_idx, :] - pose3d_rel[norm_idx, :])))
        pose3d_normed = pose3d_rel / bone_length
        relative_depth = pose3d_normed[:,2]

        if hand_side:  # transfer left hands to right hand
            image_crop = cv2.flip(image_crop, 1)
            mask_crop = cv2.flip(mask_crop,1)
            new_uv[:, 0] = refine_size - new_uv[:, 0]
            pose3d_normed[:, 0] = -pose3d_normed[:, 0]

        return image_crop, mask_crop, new_uv, crop_center, crop_size, hand_side, uv_vis, \
               relative_depth, pose3d_root, bone_length, pose3d, pose_uv_all, pose3d_normed

if __name__ == '__main__':
    import os,torch
    os.chdir('/home/yangl/MultiTaskHand/')

    train_dataset = RHDDateset('../dataset/RHD_published_v2/', 'training',output_full=True)
    Dataloader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=1,
                                            shuffle=True,
                                            drop_last=True,
                                            num_workers=0)

    ds_name,  index,  hand_image_tensor, target = train_dataset.__getitem__(11111)

    hand_image = hand_image_tensor.permute(1, 2, 0).numpy()
    hand_image = (hand_image + 1) / 2

    pose3d = target['pose3d'].numpy()
    if target['hand_side']:
        pose3d[:, 0] = -pose3d[:, 0]

    # mask = target['mask'].squeeze().cpu().numpy()
    #
    # print(target['new_uv'])
    #
    # print(target['relative_depth'])

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from vis.skeleton import plot_pose3d
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.imshow(hand_image)
    ax1.axis('off')
    plt.show()
    # plt.savefig('rhd.png',bbox_inches='tight')
    plt.close()

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    plot_pose3d(pose3d, '.-', ax1, azim=-90.0, elev=180.0)
    ax1.axis('off')
    plt.show()
    # plt.savefig('rhd.png',bbox_inches='tight')
    plt.close()

    # import matplotlib.pyplot as plt
    # from vis.skeleton import mask_cmap
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.imshow(mask,cmap=mask_cmap)
    # ax1.axis('off')
    # plt.savefig('rhd_mask.png',bbox_inches='tight')
    # plt.close()
import os.path as osp
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import cv2
from kornia.geometry.dsnt import spatial_softmax_2d, spatial_softargmax_2d


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def regress25d(heatmaps, beta):
    # Figure out what is beta
    bs = heatmaps.shape[0]
    betas = beta.clone().view(1, 21, 1).repeat(bs, 1, 1)
    uv_heatmaps = spatial_softmax_2d(heatmaps, betas)
    coord_out = spatial_softargmax_2d(uv_heatmaps, normalized_coordinates=False)
    coord_out_final = coord_out.clone()

    return coord_out_final.view(bs, 21, 2)


def MeanEPE(x, y):
    return torch.mean(F.pairwise_distance(x.permute(0, 2, 1), y.permute(0, 2, 1), p=2))


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v, norm
    return v / norm, norm


def is_on_skeleton(idx, res, direction, length, first_point, sigma=1):
    vertical_direction = np.array([direction[1], -direction[0]])
    weit = np.zeros([res, res])
    pred = np.zeros([res, res])
    if length < 1:
        for x in range(res):
            for y in range(res):
                pred[x, y] = np.sum(np.abs(np.array([x, y]) - first_point)) <= sigma
                if pred[x, y]:
                    # Check that any part of the gaussian is in-bounds
                    ul = [int(x - 2 * sigma), int(y - 2 * sigma)]
                    br = [int(x + 2 * sigma + 1), int(y + 2 * sigma + 1)]
                    if (
                            ul[0] >= res
                            or ul[1] >= res
                            or br[0] < 0
                            or br[1] < 0
                    ):
                        # If not, just return the image as is
                        continue
                    # Generate gaussian
                    size = 4 * sigma + 1
                    x_c = np.arange(0, size, 1, float)
                    y_c = x_c[:, np.newaxis]
                    x0 = y0 = size // 2
                    g = np.exp(- ((x_c - x0) ** 2 + (y_c - y0) ** 2) / (2 * sigma ** 2))
                    # Usable gaussian range
                    g_x = max(0, -ul[0]), min(br[0], res) - ul[0]
                    g_y = max(0, -ul[1]), min(br[1], res) - ul[1]
                    # Image range
                    img_x = max(0, ul[0]), min(br[0], res)
                    img_y = max(0, ul[1]), min(br[1], res)

                    pred[img_x[0]:img_x[1], img_y[0]:img_y[1]] = np.maximum(g[g_x[0]:g_x[1], g_y[0]:g_y[1]], pred[img_x[0]:img_x[1], img_y[0]:img_y[1]])
        count = 0
        for x in range(res):
            for y in range(res):
                if pred[x, y] > 0:
                    count += 1
        count = count / (res * res)
        count_vers = 1 - count
        for x in range(res):
            for y in range(res):
                weit[x, y] = count_vers if pred[x, y] > 0 else count
        return pred, weit
    for x in range(res):
        for y in range(res):
            pred_1 = 0 <= np.dot(direction, np.array([x, y]) - first_point) and np.dot(direction, np.array(
                [x, y]) - first_point) <= length
            pred_2 = np.abs(np.dot(vertical_direction, np.array([x, y]) - first_point)) <= sigma
            # pred[x, y] = pred_1 and pred_2
            if pred_1 and pred_2:
                # Check that any part of the gaussian is in-bounds
                ul = [int(x - 2 * sigma), int(y - 2 * sigma)]
                br = [int(x + 2 * sigma + 1), int(y + 2 * sigma + 1)]
                if (
                        ul[0] >= res
                        or ul[1] >= res
                        or br[0] < 0
                        or br[1] < 0
                ):
                    # If not, just return the image as is
                    continue
                # Generate gaussian
                size = 4 * sigma + 1
                x_c = np.arange(0, size, 1, float)
                y_c = x_c[:, np.newaxis]
                x0 = y0 = size // 2
                g = np.exp(- ((x_c - x0) ** 2 + (y_c - y0) ** 2) / (2 * sigma ** 2))
                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], res) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], res) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], res)
                img_y = max(0, ul[1]), min(br[1], res)
                # print("g_x", g_x)
                # print("g_y", g_y)
                # print("img_x", img_x)
                # print("img_y", img_y)
                # print("gaussian", g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
                # print("pred", pred[img_y[0]:img_y[1], img_x[0]:img_x[1]])
                pred[img_x[0]:img_x[1], img_y[0]:img_y[1]] = np.maximum(g[g_x[0]:g_x[1], g_y[0]:g_y[1]], pred[img_x[0]:img_x[1], img_y[0]:img_y[1]])
    count = 0
    for x in range(res):
        for y in range(res):
            if pred[x, y] > 0:
                count += 1
    count = count / (res * res)
    count_vers = 1 - count
    for x in range(res):
        for y in range(res):
            weit[x, y] = count_vers if pred[x, y] > 0 else count
    return pred, weit


def get_direction_from_flux(flux, mask):
    direction = np.copy(flux)
    direction[:, :, 0] = flux[:, :, 1] * mask[:, :, 0]
    direction[:, :, 1] = -flux[:, :, 0] * mask[:, :, 0]
    return direction


def maps_to_kp(flux_map, dis_map, res=64):
    locs = np.zeros([res, res, 2])
    for y in range(res):
        locs[:, y, 0] = np.array(range(res))
        locs[:, y, 1] = y
    kp = np.zeros([21, 2])

    # predicts the wrist keypoint 0
    kp[0] = 0
    for i in range(5):
        flux = flux_map[4 * i, :, :, :2]
        mask = flux_map[4 * i, :, :, 2, None]
        direction = get_direction_from_flux(flux, mask)
        count = 5
        if np.sum(np.abs(mask)) != 0:
            kp[0] += np.sum(np.sum(np.abs(mask) * (
                    direction * dis_map[4 * i, :, :, 0, None] + locs), axis=0), axis=0) / np.sum(np.abs(mask))
        # else:
        #     print("flag")
    kp[0] = kp[0] / count
    # print("kp[0] = ", kp[0])

    for i in range(5):
        for j in range(4):
            as_second_point_flux = flux_map[4 * i + j, :, :, :2]
            as_second_point_mask = flux_map[4 * i + j, :, :, 2, None]  # (64, 64, 1)
            as_second_point_direction = get_direction_from_flux(as_second_point_flux, as_second_point_mask)
            as_second_point_votes = np.abs(as_second_point_mask) * (
                    -as_second_point_direction * dis_map[4 * i + j, :, :, 1, None] + locs)
            as_second_point_pred = as_second_point_votes / np.sum(np.abs(as_second_point_mask))
            if j != 3:
                # predict the location of keypoint (4, 3, 2), (8, 7, 6), (12, 11, 10), (16, 15, 14), (20, 19, 18)
                as_first_point_flux = flux_map[4 * i + j + 1, :, :, :2]
                as_first_point_mask = flux_map[4 * i + j + 1, :, :, 2, None]
                as_first_point_direction = get_direction_from_flux(as_first_point_flux, as_first_point_mask)
                as_first_point_votes = np.abs(as_first_point_mask) * (
                        as_first_point_direction * dis_map[4 * i + j + 1, :, :, 0, None] + locs)
                as_first_point_pred = as_first_point_votes / np.sum(np.abs(as_first_point_mask))
                kp[4 * i + 4 - j] = np.sum(np.sum(as_second_point_pred + as_first_point_pred, axis=0), axis=0) / 2
            else:
                # predicts the location of fingertips 1, 5, 9, 13, 17
                kp[4 * i + 4 - j] = np.sum(np.sum(as_second_point_pred, axis=0), axis=0)
            # print("kp[", 4 * i + 4 - j, "] = ", kp[4 * i + 4 - j])

    return kp


def fill_maps(flux_map, dis_map, ske_mask, phalanx_idx, first_point, second_point,
              res=64, sigma=1):
    phalanx = second_point - first_point
    direction, length = normalize(phalanx)
    counter_clockwise_flux = np.array([-direction[1], direction[0]])
    clockwise_flux = np.array([direction[1], -direction[0]])
    ske_mask[phalanx_idx, :, :, 0] = is_on_skeleton(res, direction, length, first_point, sigma)

    for x in range(res):
        for y in range(res):
            if np.cross(first_point - np.array([x, y]), direction) < 0 and ske_mask[phalanx_idx, x, y, 0]:
                flux_map[phalanx_idx, x, y, :2] = counter_clockwise_flux
                flux_map[phalanx_idx, x, y, 2] = -1
                dis_map[phalanx_idx, x, y, 0] = normalize(second_point - np.array([x, y]))[1]
                dis_map[phalanx_idx, x, y, 1] = normalize(np.array([x, y]) - first_point)[1]

            elif np.cross(first_point - np.array([x, y]), direction) >= 0 and ske_mask[phalanx_idx, x, y, 0]:
                flux_map[phalanx_idx, x, y, :2] = clockwise_flux
                flux_map[phalanx_idx, x, y, 2] = 1
                dis_map[phalanx_idx, x, y, 0] = normalize(second_point - np.array([x, y]))[1]
                dis_map[phalanx_idx, x, y, 1] = normalize(np.array([x, y]) - first_point)[1]

            else:
                flux_map[phalanx_idx, x, y, :] = 0
                dis_map[phalanx_idx, x, y, :] = 0
    if ske_mask[phalanx_idx, :, :, 0].any() != True:
        flux_map[phalanx_idx, int(first_point[0]), int(first_point[1]), 2] = 1
        flux_map[phalanx_idx, int(second_point[0]), int(second_point[1]), 2] = -1


# Use vertical vector to phalanges
def kp_to_maps(kp, res=64, sigma=1):
    ske_mask = np.zeros([20, res, res, 1])

    # maps
    flux_map = np.zeros([20, res, res, 3])
    dis_map = np.zeros([20, res, res, 2])

    wrist_coor = kp[0]
    # 0-4 4-3 3-2 2-1
    for j in range(5):

        # generates the maps of phalanges connected to wrist
        second_point = kp[4 * j + 4]
        first_point = wrist_coor
        fill_maps(flux_map, dis_map, ske_mask, 4 * j, first_point, second_point, res, sigma)

        # generates other 3 phalanges of each finger:
        # example: phalange 1 connects kp 4 and kp 3
        for i in range(3):
            curr = 4 * j + 4 - i
            first_point = kp[curr]
            second_point = kp[curr - 1]
            fill_maps(flux_map, dis_map, ske_mask, 4 * j + i + 1, first_point, second_point, res, sigma)

    return flux_map, dis_map


# Here the idx is the batch number
def draw_maps(flux, dis, idx, mode):
    direction = np.zeros([flux.shape[1], flux.shape[2]])
    magnitude = np.zeros([flux.shape[1], flux.shape[2]])
    dis_to_first = np.zeros([dis.shape[1], dis.shape[2]])
    dis_to_second = np.zeros([dis.shape[1], dis.shape[2]])
    for bone in range(flux.shape[0]):
        for x in range(flux.shape[1]):
            for y in range(flux.shape[2]):
                if flux[bone, x, y, 0] != 0:
                    dis_to_first[y, x] = dis[bone, y, x, 0]
                    dis_to_second[y, x] = dis[bone, x, y, 1]
                    direction[y, x] = flux[bone, x, y, 1] / flux[bone, x, y, 0]
                    magnitude[y, x] = flux[bone, x, y, 2]
                    if flux[bone, x, y, 0] > 0:
                        direction[y, x] = np.degrees(np.arctan(direction[y, x]))
                    if flux[bone, x, y, 0] < 0 < flux[bone, x, y, 1]:
                        direction[y, x] = np.degrees(np.arctan(direction[y, x])) + 180
                    if flux[bone, x, y, 1] < 0 and flux[bone, x, y, 0] < 0:
                        direction[y, x] = np.degrees(np.arctan(direction[y, x])) - 180

    # 作图阶段
    fig = plt.figure()

    ax_1 = fig.add_subplot(221)
    plt.sca(ax_1)
    ax_1.set_title("flux_map_direction")
    sns.heatmap(data=direction,
                cmap=sns.diverging_palette(250, 15, sep=12, n=20, center="dark"),  # 区分度显著色盘：sns.diverging_palette()使用
                vmax=180,
                vmin=-180
                )

    ax_2 = fig.add_subplot(222)
    plt.sca(ax_2)
    ax_2.set_title("flux_map_magnitude")
    sns.heatmap(data=magnitude)

    ax_3 = fig.add_subplot(223)
    plt.sca(ax_3)
    ax_3.set_title("dis_map_to_the_first_keypoint")
    sns.heatmap(data=dis_to_first)

    ax_4 = fig.add_subplot(224)
    plt.sca(ax_4)
    ax_4.set_title("dis_map_to_the_second_keypoint")
    sns.heatmap(data=dis_to_second)

    maps_save_path = osp.join("sample", f"sample_{idx}", f"{mode}_maps_{idx}.jpg")
    plt.savefig(maps_save_path)
    plt.close()


# Here the idx is the batch number
def draw_maps_with_one_bone(flux, dis, idx, mode):
    for bone in range(20):
        direction = np.zeros([flux.shape[1], flux.shape[2]])
        magnitude = np.zeros([flux.shape[1], flux.shape[2]])
        dis_to_first = np.zeros([dis.shape[1], dis.shape[2]])
        dis_to_second = np.zeros([dis.shape[1], dis.shape[2]])
        for x in range(flux.shape[1]):
            for y in range(flux.shape[2]):
                if flux[bone, x, y, 0] != 0:
                    dis_to_first[y, x] = dis[bone, x, y, 0]
                    dis_to_second[y, x] = dis[bone, x, y, 1]
                    direction[y, x] = flux[bone, x, y, 1] / flux[bone, x, y, 0]
                    magnitude[y, x] = flux[bone, x, y, 2]
                    if flux[bone, x, y, 0] > 0:
                        direction[y, x] = np.degrees(np.arctan(direction[y, x]))
                    if flux[bone, x, y, 0] < 0 < flux[bone, x, y, 1]:
                        direction[y, x] = np.degrees(np.arctan(direction[y, x])) + 180
                    if flux[bone, x, y, 1] < 0 and flux[bone, x, y, 0] < 0:
                        direction[y, x] = np.degrees(np.arctan(direction[y, x])) - 180

        # 作图阶段
        fig = plt.figure()

        ax_1 = fig.add_subplot(221)
        plt.sca(ax_1)
        ax_1.set_title("flux_map_direction")
        sns.heatmap(data=direction,
                    cmap=sns.diverging_palette(250, 15, sep=12, n=20, center="dark"),
                    # 区分度显著色盘：sns.diverging_palette()使用
                    vmax=180,
                    vmin=-180
                    )

        ax_2 = fig.add_subplot(222)
        plt.sca(ax_2)
        ax_2.set_title("flux_map_magnitude")
        sns.heatmap(data=magnitude)

        ax_3 = fig.add_subplot(223)
        plt.sca(ax_3)
        ax_3.set_title("dis_map_to_the_first_keypoint")
        sns.heatmap(data=dis_to_first)

        ax_4 = fig.add_subplot(224)
        plt.sca(ax_4)
        ax_4.set_title("dis_map_to_the_second_keypoint")
        sns.heatmap(data=dis_to_second)

        maps_save_path = osp.join("sample", f"sample_{idx}", f"{mode}_maps_{idx}_with_one_bone_{bone}.jpg")
        plt.savefig(maps_save_path)
        plt.close()


def maps_to_kp_2(front_vecs, front_diss, back_vecs, back_diss, maskss, device, res=64):
    batch_size = front_vecs.shape[0]
    locs = torch.zeros([batch_size, res, res, 2]).to(device, non_blocking=True)
    for y in range(res):
        locs[:, :, y, 0] = torch.FloatTensor(range(res))
        locs[:, :, y, 1] = y
    kp = torch.zeros([batch_size, 21, 2]).to(device, non_blocking=True)

    front_vecs = front_vecs.reshape(batch_size, 20, res, res, 2)
    front_diss = front_diss.reshape(batch_size, 20, res, res, 1)
    back_vecs = back_vecs.reshape(batch_size, 20, res, res, 2)
    back_diss = back_diss.reshape(batch_size, 20, res, res, 1)
    maskss = maskss.reshape(batch_size, 20, res, res, 1)

    # predicts the wrist keypoint 0
    for i in range(5):
        back_vec = back_vecs[:, 4 * i, :, :, :]
        back_dis = back_diss[:, 4 * i, :, :, :]
        masks = maskss[:, 4 * i, :, :, :]
        j = 0
        for mask in masks:
            if torch.sum(mask) != 0:
                proposal = mask * (back_vec[j] * back_dis[j, :, :, 0, None] * res + locs[j])
                # for x in range(64):
                #     for y in range(64):
                #         if mask[x, y, 0]:
                #             print(back_vec[j, x, y] * back_dis[j, x, y, 0, None] + locs[j][x][y])
                # print("proposal:", proposal.shape)
                kp[j][0] += torch.sum(torch.sum(proposal, dim=0), dim=0) / (torch.sum(mask) + 1e-6)
            j = j + 1
    kp[:, 0] = kp[:, 0] / 5

    for i in range(5):
        for j in range(4):
            k = 0
            for masks in maskss:
                front_votes = masks[4 * i + j] * (
                        front_vecs[k][4 * i + j] * front_diss[k, 4 * i + j, :, :, 0, None] * res + locs[k])
                front_pred = front_votes / (torch.sum(masks[4 * i + j]) + 1e-6)
                if j != 3:
                    # predict keypoint (4, 3, 2), (8, 7, 6), (12, 11, 10), (16, 15, 14), (20, 19, 18)
                    back_votes = masks[4 * i + j + 1] * (
                            back_vecs[k][4 * i + j + 1] * back_diss[k, 4 * i + j + 1, :, :, 0, None] * res + locs[k])
                    back_pred = back_votes / (torch.sum(masks[4 * i + j + 1]) + 1e-6)
                    kp[k, 4 * i + 4 - j] = torch.sum(torch.sum(front_pred + back_pred, dim=0), dim=0) / 2
                else:
                    # predicts the location of fingertips 1, 5, 9, 13, 17
                    kp[k, 4 * i + 4 - j] = torch.sum(torch.sum(front_pred, dim=0), dim=0)
                k = k + 1

    return kp * 4


def fill_maps_2(front_vec_map, front_dis_map, back_vec_map, back_dis_map, ske_mask, weit_map, phalanx_idx, first_point,
                second_point,
                res=64, sigma=1):
    phalanx = second_point - first_point
    direction, length = normalize(phalanx)
    ske_mask[phalanx_idx, :, :, 0], weit_map[phalanx_idx, :, :, 0] = is_on_skeleton(phalanx_idx, res, direction, length, first_point, sigma)
    for x in range(res):
        for y in range(res):
            if ske_mask[phalanx_idx, x, y, 0]:
                sup = second_point - np.array([x, y])
                front_vec_map[phalanx_idx, x, y, :] = direction
                front_dis_map[phalanx_idx, x, y, 0] = np.dot(sup, direction) / res
                sup = first_point - np.array([x, y])
                back_vec_map[phalanx_idx, x, y, :] = -direction
                back_dis_map[phalanx_idx, x, y, 0] = -np.dot(sup, direction) / res

            else:
                front_vec_map[phalanx_idx, x, y, :] = 0
                front_dis_map[phalanx_idx, x, y, :] = 0
                back_vec_map[phalanx_idx, x, y, :] = 0
                back_dis_map[phalanx_idx, x, y, :] = 0


# Use directions of phalanges
def kp_to_maps_2(kp, res=64, sigma=1):
    ske_mask = np.zeros([20, res, res, 1])
    weit_map = np.zeros([20, res, res, 1])

    front_vec_map = np.zeros([20, res, res, 2])
    front_dis_map = np.zeros([20, res, res, 1])
    back_vec_map = np.zeros([20, res, res, 2])
    back_dis_map = np.zeros([20, res, res, 1])

    wrist_coor = kp[0]
    # 0-4 4-3 3-2 2-1
    for j in range(5):

        # generates the maps of phalanges connected to wrist
        second_point = kp[4 * j + 4]
        first_point = wrist_coor
        fill_maps_2(front_vec_map, front_dis_map, back_vec_map, back_dis_map, ske_mask, weit_map, 4 * j, first_point,
                    second_point, res, sigma)

        # generates other 3 phalanges of each finger:
        # example: phalange 1 connects kp 4 and kp 3
        for i in range(3):
            curr = 4 * j + 4 - i
            first_point = kp[curr]
            second_point = kp[curr - 1]
            fill_maps_2(front_vec_map, front_dis_map, back_vec_map, back_dis_map, ske_mask, weit_map, 4 * j + i + 1,
                        first_point, second_point, res, sigma)
    return front_vec_map, front_dis_map, back_vec_map, back_dis_map, ske_mask, weit_map


# Here the idx is the batch number
def draw_maps_2(front_vec_map, front_dis_map, back_vec_map, back_dis_map, ske_mask_map, weit_map, idx, mode):
    front_vec_map = front_vec_map * weit_map
    back_vec_map = back_vec_map * weit_map
    front_vec = np.zeros([front_vec_map.shape[1], front_vec_map.shape[2]])
    front_dis = np.zeros([front_dis_map.shape[1], front_dis_map.shape[2]])
    back_vec = np.zeros([back_vec_map.shape[1], back_vec_map.shape[2]])
    back_dis = np.zeros([back_dis_map.shape[1], back_dis_map.shape[2]])
    ske_mask = np.zeros([back_dis_map.shape[1], back_dis_map.shape[2]])
    for bone in range(front_vec_map.shape[0]):
        for x in range(front_vec_map.shape[1]):
            for y in range(front_vec_map.shape[2]):
                if ske_mask_map[bone, x, y, 0] > 0:
                    ske_mask[y, x] = np.maximum(ske_mask[y, x], ske_mask_map[bone, x, y, 0])
                if front_vec_map[bone, x, y, 1] != 0 \
                        and np.abs(front_vec_map[bone, x, y, 1] + front_vec_map[bone, x, y, 0]) > np.mean(np.abs(front_vec_map)) \
                        and back_vec_map[bone, x, y, 1] != 0 \
                        and np.abs(back_vec_map[bone, x, y, 1] + back_vec_map[bone, x, y, 0]) > np.mean(np.abs(back_vec_map)):
                    back_vec[y, x] = back_vec_map[bone, x, y, 0] / back_vec_map[bone, x, y, 1]
                    back_dis[y, x] = back_dis_map[bone, x, y, 0]
                    front_vec[y, x] = front_vec_map[bone, x, y, 0] / front_vec_map[bone, x, y, 1]
                    front_dis[y, x] = front_dis_map[bone, x, y, 0]
                    if front_vec_map[bone, x, y, 1] > 0:
                        front_vec[y, x] = np.degrees(np.arctan(front_vec[y, x]))
                    if front_vec_map[bone, x, y, 1] < 0 < front_vec_map[bone, x, y, 0]:
                        front_vec[y, x] = np.degrees(np.arctan(front_vec[y, x])) + 180
                    if front_vec_map[bone, x, y, 0] < 0 and front_vec_map[bone, x, y, 1] < 0:
                        front_vec[y, x] = np.degrees(np.arctan(front_vec[y, x])) - 180
                    if back_vec_map[bone, x, y, 1] > 0:
                        back_vec[y, x] = np.degrees(np.arctan(back_vec[y, x]))
                    if back_vec_map[bone, x, y, 1] < 0 < back_vec_map[bone, x, y, 0]:
                        back_vec[y, x] = np.degrees(np.arctan(back_vec[y, x])) + 180
                    if back_vec_map[bone, x, y, 0] < 0 and back_vec_map[bone, x, y, 1] < 0:
                        back_vec[y, x] = np.degrees(np.arctan(back_vec[y, x])) - 180
                if front_vec_map[bone, x, y, 1] == 0:
                    if front_vec_map[bone, x, y, 0] > 0:
                        front_vec[y, x] = 90
                    elif front_vec_map[bone, x, y, 0] < 0:
                        front_vec[y, x] = -90
                if back_vec_map[bone, x, y, 1] == 0:
                    if back_vec_map[bone, x, y, 0] > 0:
                        back_vec[y, x] = 90
                    elif back_vec_map[bone, x, y, 0] < 0:
                        back_vec[y, x] = -90
    # 作图阶段
    fig = plt.figure()

    ax_1 = fig.add_subplot(221)
    plt.sca(ax_1)
    ax_1.set_title("front_vec")
    sns.heatmap(data=front_vec,
                cmap=sns.diverging_palette(250, 15, sep=12, n=20),  # 区分度显著色盘：sns.diverging_palette()使用
                vmax=180,
                vmin=-180
                )

    ax_2 = fig.add_subplot(222)
    plt.sca(ax_2)
    ax_2.set_title("front_dis")
    sns.heatmap(data=front_dis,
                cmap=sns.light_palette("orange", as_cmap=True),
                vmax=1,
                vmin=0)

    ax_3 = fig.add_subplot(223)
    plt.sca(ax_3)
    ax_3.set_title("back_vec")
    sns.heatmap(data=back_vec,
                cmap=sns.diverging_palette(250, 15, sep=12, n=20),  # 区分度显著色盘：sns.diverging_palette()使用
                vmax=180,
                vmin=-180
                )

    ax_4 = fig.add_subplot(224)
    plt.sca(ax_4)
    ax_4.set_title("back_dis")
    sns.heatmap(data=back_dis,
                cmap=sns.light_palette("orange", as_cmap=True),
                vmax=1,
                vmin=0)

    maps_save_path = osp.join("sample_2", f"sample_{idx}", f"{mode}_maps.jpg")
    plt.savefig(maps_save_path)
    plt.close(fig)

    fig_2 = plt.figure()
    ax_1 = fig_2.add_subplot(111)
    ax_1.set_title("ske_mask")
    sns.heatmap(data=ske_mask,
                cmap=sns.light_palette("orange", as_cmap=True)
                )

    maps_save_path = osp.join("sample_2", f"sample_{idx}", f"{mode}_ske.jpg")
    plt.savefig(maps_save_path)
    plt.close()


# Here the idx is the batch number
def draw_maps_with_one_bone_2(front_vec_map, front_dis_map, back_vec_map, back_dis_map, ske_mask_map, weit_map, idx, mode):
    for bone in range(front_vec_map.shape[0]):
        front_vec_map = front_vec_map * weit_map
        back_vec_map = back_vec_map * weit_map
        front_vec = np.zeros([front_vec_map.shape[1], front_vec_map.shape[2]])
        front_dis = np.zeros([front_dis_map.shape[1], front_dis_map.shape[2]])
        back_vec = np.zeros([back_vec_map.shape[1], back_vec_map.shape[2]])
        back_dis = np.zeros([back_dis_map.shape[1], back_dis_map.shape[2]])
        ske_mask = np.zeros([back_dis_map.shape[1], back_dis_map.shape[2]])
        weit = np.zeros([back_dis_map.shape[1], back_dis_map.shape[2]])
        for x in range(front_vec_map.shape[1]):
            for y in range(front_vec_map.shape[2]):
                ske_mask[y, x] = ske_mask_map[bone, x, y, 0]
                weit[y, x] = weit_map[bone, x, y, 0]
                if front_vec_map[bone, x, y, 1] != 0 \
                        and np.abs(front_vec_map[bone, x, y, 1] + front_vec_map[bone, x, y, 0]) > np.mean(np.abs(front_vec_map)) \
                        and back_vec_map[bone, x, y, 1] != 0 \
                        and np.abs(back_vec_map[bone, x, y, 1] + back_vec_map[bone, x, y, 0]) > np.mean(np.abs(back_vec_map)):
                    back_vec[y, x] = back_vec_map[bone, x, y, 0] / back_vec_map[bone, x, y, 1]
                    back_dis[y, x] = back_dis_map[bone, x, y, 0]
                    front_vec[y, x] = front_vec_map[bone, x, y, 0] / front_vec_map[bone, x, y, 1]
                    front_dis[y, x] = front_dis_map[bone, x, y, 0]
                    if front_vec_map[bone, x, y, 1] > 0:
                        front_vec[y, x] = np.degrees(np.arctan(front_vec[y, x]))
                    if front_vec_map[bone, x, y, 1] < 0 < front_vec_map[bone, x, y, 0]:
                        front_vec[y, x] = np.degrees(np.arctan(front_vec[y, x])) + 180
                    if front_vec_map[bone, x, y, 0] < 0 and front_vec_map[bone, x, y, 1] < 0:
                        front_vec[y, x] = np.degrees(np.arctan(front_vec[y, x])) - 180
                    if back_vec_map[bone, x, y, 1] > 0:
                        back_vec[y, x] = np.degrees(np.arctan(back_vec[y, x]))
                    if back_vec_map[bone, x, y, 1] < 0 < back_vec_map[bone, x, y, 0]:
                        back_vec[y, x] = np.degrees(np.arctan(back_vec[y, x])) + 180
                    if back_vec_map[bone, x, y, 0] < 0 and back_vec_map[bone, x, y, 1] < 0:
                        back_vec[y, x] = np.degrees(np.arctan(back_vec[y, x])) - 180
                if front_vec_map[bone, x, y, 1] == 0:
                    if front_vec_map[bone, x, y, 0] > 0:
                        front_vec[y, x] = 90
                    elif front_vec_map[bone, x, y, 0] < 0:
                        front_vec[y, x] = -90
                if back_vec_map[bone, x, y, 1] == 0:
                    if back_vec_map[bone, x, y, 0] > 0:
                        back_vec[y, x] = 90
                    elif back_vec_map[bone, x, y, 0] < 0:
                        back_vec[y, x] = -90
        # 作图阶段
        fig = plt.figure()

        ax_1 = fig.add_subplot(221)
        plt.sca(ax_1)
        ax_1.set_title("front_vec")
        sns.heatmap(data=front_vec,
                    cmap=sns.diverging_palette(250, 15, sep=12, n=20),
                    # 区分度显著色盘：sns.diverging_palette()使用
                    vmax=180,
                    vmin=-180
                    )

        ax_2 = fig.add_subplot(222)
        plt.sca(ax_2)
        ax_2.set_title("front_dis")
        sns.heatmap(data=front_dis,
                    cmap=sns.light_palette("orange", as_cmap=True),
                    # 区分度显著色盘：sns.diverging_palette()使用
                    vmax=1,
                    vmin=0)

        ax_3 = fig.add_subplot(223)
        plt.sca(ax_3)
        ax_3.set_title("back_vec")
        sns.heatmap(data=back_vec,
                    cmap=sns.diverging_palette(250, 15, sep=12, n=20),
                    # 区分度显著色盘：sns.diverging_palette()使用
                    vmax=180,
                    vmin=-180
                    )

        ax_4 = fig.add_subplot(224)
        plt.sca(ax_4)
        ax_4.set_title("back_dis")
        sns.heatmap(data=back_dis,
                    cmap=sns.light_palette("orange", as_cmap=True),
                    # 区分度显著色盘：sns.diverging_palette()使用
                    vmax=1,
                    vmin=0)

        maps_save_path = osp.join("sample_2", f"sample_{idx}", f"{mode}_maps_with_one_bone_{bone}.jpg")
        plt.savefig(maps_save_path)
        plt.close(fig)

        fig_2 = plt.figure(figsize=(16, 9))
        ax_1 = fig_2.add_subplot(121)
        ax_1.set_title("ske_mask")
        sns.heatmap(data=ske_mask,
                    cmap=sns.light_palette("orange", as_cmap=True)
                    )
        ax_2 = fig_2.add_subplot(122)
        ax_2.set_title("weights")
        sns.heatmap(data=weit,
                    cmap=sns.light_palette("red", as_cmap=True)
                    )

        maps_save_path = osp.join("sample_2", f"sample_{idx}", f"{mode}_ske_with_one_bone_{bone}.jpg")
        plt.savefig(maps_save_path)
        plt.close()


def draw_kps(kps, img, idx, mode):
    for kp in kps:
        img = cv2.circle(img, kp, 3, color=(255, 0, 0), thickness=1)
    for i in range(3):
        # pinkle
        img = cv2.line(img, kps[i + 1], kps[i + 2], color=(255, 0, 0), thickness=2)
        img = cv2.line(img, kps[4], kps[0], color=(255, 0, 0), thickness=2)
        #
        img = cv2.line(img, kps[4 + i + 1], kps[4 + i + 2], color=(0, 255, 0), thickness=2)
        img = cv2.line(img, kps[8], kps[0], color=(0, 255, 0), thickness=2)
        #
        img = cv2.line(img, kps[8 + i + 1], kps[8 + i + 2], color=(0, 0, 255), thickness=2)
        img = cv2.line(img, kps[12], kps[0], color=(0, 0, 255), thickness=2)
        #
        img = cv2.line(img, kps[12 + i + 1], kps[12 + i + 2], color=(0, 255, 255), thickness=2)
        img = cv2.line(img, kps[16], kps[0], color=(0, 255, 255), thickness=2)
        #
        img = cv2.line(img, kps[16 + i + 1], kps[16 + i + 2], color=(255, 0, 255), thickness=2)
        img = cv2.line(img, kps[20], kps[0], color=(255, 0, 255), thickness=2)

    cv2.imwrite(f"sample_2/sample_{idx}/{mode}_kps.jpg", img)


# Need to update?
def check_pairs(epoch, train_loader):
    if epoch == 0:
        sample_idxs = []
        for i in range(5):
            sample_idx = random.randint(0, len(train_loader))
            sample_idxs.append(sample_idx)
        for i, sample in enumerate(train_loader):
            for idx in sample_idxs:
                if idx == i:
                    image = sample["img_crop"][0]
                    flux = sample["flux_map"][0]
                    dis = sample["dis_map"][0]
                    draw_maps(flux, dis, idx, "targ")


def calc_dists(preds, target, normalize, mask):
    preds = preds.float()
    target = target.float()
    dists = torch.zeros(preds.size(1), preds.size(0))  # (njoint, B)
    for b in range(preds.size(0)):
        for j in range(preds.size(1)):
            if mask[b][j] == 0:
                dists[j, b] = -1
            elif target[b, j, 0] < 1 or target[b, j, 1] < 1:
                dists[j, b] = -1
            else:
                dists[j, b] = torch.dist(preds[b, j, :], target[b, j, :]) / normalize[b]

    return dists


def dist_acc(dist, thr=0.5):
    """ Return percentage below threshold while ignoring values with a -1 """
    dist = dist[dist != -1]
    if len(dist) > 0:
        return 1.0 * (dist < thr).sum().item() / len(dist)
    else:
        return -1


def gen_heatmap(img, pt, sigma):
    """generate heatmap based on pt coord.

    :param img: original heatmap, zeros
    :type img: np (H,W) float32
    :param pt: keypoint coord.
    :type pt: np (2,) int32
    :param sigma: guassian sigma
    :type sigma: float
    :return
    - generated heatmap, np (H, W) each pixel values id a probability
    - flag 0 or 1: indicate wheather this heatmap is valid(1)

    """

    pt = pt.astype(np.int32)
    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (
            ul[0] >= img.shape[1]
            or ul[1] >= img.shape[0]
            or br[0] < 0
            or br[1] < 0
    ):
        # If not, just return the image as is
        return img, 0

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img, 1


def get_heatmap_pred(heatmaps):
    """ get predictions from heatmaps in torch Tensor
        return type: torch.LongTensor
    """
    assert heatmaps.dim() == 4, 'Score maps should be 4-dim (B, nJoints, H, W)'
    maxval, idx = torch.max(heatmaps.view(heatmaps.size(0), heatmaps.size(1), -1), 2)

    maxval = maxval.view(heatmaps.size(0), heatmaps.size(1), 1)
    idx = idx.view(heatmaps.size(0), heatmaps.size(1), 1)

    preds = idx.repeat(1, 1, 2).float()  # (B, njoint, 2)

    preds[:, :, 0] = (preds[:, :, 0]) % heatmaps.size(3)  # + 1
    preds[:, :, 1] = torch.floor((preds[:, :, 1]) / heatmaps.size(3))  # + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds


def accuracy_heatmap(output, target, vis, thr=0.5):
    """ Calculate accuracy according to PCK, but uses ground truth heatmap rather than x,y locations
        First to be returned is average accuracy across 'idxs', Second is individual accuracies
    """
    preds = get_heatmap_pred(output).float()  # (B, njoint, 2)
    gts = get_heatmap_pred(target).float()
    val = MeanEPE(preds * vis, gts * vis)

    return val


def get_measures(self, val_min, val_max, steps):
    """ Outputs the average mean and median error as well as the pck score. """
    thresholds = np.linspace(val_min, val_max, steps)
    thresholds = np.array(thresholds)
    norm_factor = np.trapz(np.ones_like(thresholds), thresholds)

    # init mean measures
    epe_mean_all = list()
    epe_median_all = list()
    auc_all = list()
    pck_curve_all = list()

    # Create one plot for each part
    for part_id in range(self.num_kp):
        # mean/median error
        mean, median = self._get_epe(part_id)

        if mean is None:
            # there was no valid measurement for this keypoint
            continue

        epe_mean_all.append(mean)
        epe_median_all.append(median)

        # pck/auc
        pck_curve = list()
        for t in thresholds:
            pck = self._get_pck(part_id, t)
            pck_curve.append(pck)

        pck_curve = np.array(pck_curve)
        pck_curve_all.append(pck_curve)
        auc = np.trapz(pck_curve, thresholds)
        auc /= norm_factor
        auc_all.append(auc)
    # Display error per keypoint
    epe_mean_joint = epe_mean_all
    epe_mean_all = np.mean(np.array(epe_mean_all))
    epe_median_all = np.mean(np.array(epe_median_all))
    auc_all = np.mean(np.array(auc_all))
    pck_curve_all = np.mean(np.array(pck_curve_all), axis=0)  # mean only over keypoints

    return (
        epe_mean_all,
        epe_mean_joint,
        epe_median_all,
        auc_all,
        pck_curve_all,
        thresholds,
    )


def generate_bone_vis(vis, device):
    # vis shape (B, 21, 1)
    batch_size = vis.shape[0]
    bone_vis = torch.zeros((batch_size, 20)).to(device)
    for finger_idx in range(5):
        bone_vis[:, 0 + finger_idx * 4] = vis[:, 0, 0] * vis[:, 4 + finger_idx * 4, 0]
        bone_vis[:, 1 + finger_idx * 4] = vis[:, 4 + finger_idx * 4, 0] * vis[:, 3 + finger_idx * 4, 0]
        bone_vis[:, 2 + finger_idx * 4] = vis[:, 3 + finger_idx * 4, 0] * vis[:, 2 + finger_idx * 4, 0]
        bone_vis[:, 3 + finger_idx * 4] = vis[:, 2 + finger_idx * 4, 0] * vis[:, 1 + finger_idx * 4, 0]
    # bone_vis shape (B, 20, 1)
    return bone_vis[:, :, None]


def process_generating_bone_vis(vis):
    bone_vis = np.zeros(20)
    for finger_idx in range(5):
        bone_vis[0 + finger_idx * 4] = vis[0] * vis[4 + finger_idx * 4]
        bone_vis[1 + finger_idx * 4] = vis[4 + finger_idx * 4] * vis[3 + finger_idx * 4]
        bone_vis[2 + finger_idx * 4] = vis[3 + finger_idx * 4] * vis[2 + finger_idx * 4]
        bone_vis[3 + finger_idx * 4] = vis[2 + finger_idx * 4] * vis[1 + finger_idx * 4]
    return bone_vis



# *_*coding:utf-8 *_*# *_*coding:utf-8 *_*
import sys
sys.path.append("/data/gyro_demo/deep_gyroscope_optical_stabilization")
import cv2
import argparse
import subprocess
import math
import imageio
import os

import numpy as np

from numpy.linalg import norm
from scipy.linalg import pinv, svd
from functools import reduce
from collections import OrderedDict

# reshape size with = 562, height = 669
SHAPE = (669, 562)

def cmp_psnr(img1, img2):
    img1 = np.float32(img1)
    img2 = np.float32(img2)

    mse = cv2.norm(img1, img2, cv2.NORM_L2SQR)
    mse = mse / (img1.shape[0] * img1.shape[1])
    # mse = cv2.absdiff(img1, img2)
    # mse = mse.sum()
    # print("mse: {}".format(mse))
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def gaussian_function(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 /
                                                     (2 * sigma ** 2))


def generate_gaussian_weight(m, block_height, hom_block_num=10):
    blocks = np.array([i for i in range(hom_block_num)])
    # weight = gaussian_function(m, blocks, sigma=0.1 * hom_block_num)
    # 根据每个点的高度计算高斯权重
    patch_centers = blocks * block_height + block_height / 2
    weight = gaussian_function(m, patch_centers, sigma=0.05 * block_height * hom_block_num)
    return weight


# 根据每个点的位置，计算其所在的patch，以计算出的patch作为中点（mu），计算一个长度为hom_block_num，标准差为0.1*patch高度（sigma）的向量
def compute_gaussian_weights(block_height, src_pts, hom_block_num=10):
    inliers_height = src_pts[:, 1]
    weights = []
    for height in inliers_height:
        weight = generate_gaussian_weight(height, block_height, hom_block_num)
        weights.append(weight)
    return weights


def condition_points(pts):
    mean = np.mean(pts[:, :2], axis=0)
    dis = norm(pts[:, :2] - mean)
    s = math.sqrt(2.0) * len(pts) / dis
    tX = s * (-mean[0])
    tY = s * (-mean[1])
    T = np.float64([[s, 0, tX],
                    [0, s, tY],
                    [0, 0, 1]])
    normalizedPoints = np.zeros((len(pts), 3))
    for i, kp in enumerate(pts):
        normalizedPoints[i] = np.dot(T, kp)
    return normalizedPoints, T


def compute_homography_mixtures(src_pts, dst_pts, img, patch):
    assert len(src_pts) == len(dst_pts), "the dimesion is not match"

    hom_block_num = patch

    src_pts, dst_pts = np.array(src_pts), np.array(dst_pts)

    # 根绝每个点所在的位置计算高斯权重
    weights = compute_gaussian_weights(block_height=img[0].shape[0] // hom_block_num,
                                       src_pts=src_pts,
                                       hom_block_num=hom_block_num)

    src_pts, T_src = condition_points(src_pts)
    dst_pts, T_dst = condition_points(dst_pts)

    assert np.mean(src_pts[:, :2], axis=0)[0] < 1e-9, "points centroid should be at (0, 0)"
    assert 1.414 < norm(src_pts[:, :2] - np.mean(src_pts[:, :2], axis=0)) / len(src_pts) < 1.415, \
        "normalized distancedistance should be sqrt(2) insted of {}".format(
            norm(src_pts[:, :2] - np.mean(src_pts[:, :2], axis=0)) / len(src_pts))

    assert np.mean(dst_pts[:, :2], axis=0)[0] < 1e-9, "points centroid should be at (0, 0)"
    assert 1.414 < norm(dst_pts[:, :2] - np.mean(dst_pts[:, :2], axis=0)) / len(dst_pts) < 1.415, \
        "normalized distancedistance should be sqrt(2) insted of {}".format(
            norm(dst_pts[:, :2] - np.mean(dst_pts[:, :2], axis=0)) / len(dst_pts))

    x = list(zip(src_pts[:, 0], src_pts[:, 1], dst_pts[:, 0], dst_pts[:, 1]))

    A = [[
             float(-x1),
             float(-y1), -1, 0, 0, 0,
             float(x2) * float(x1),
             float(x2) * float(y1),
             float(x2)
         ] if i % 2 == 0 else [
        0, 0, 0,
        float(-x1),
        float(-y1), -1,
        float(y2) * float(x1),
        float(y2) * float(y1),
        float(y2)
    ] for x1, y1, x2, y2 in x for i in range(2)]
    A = np.array(A)

    """
    # test ax=0 formula using SVD
    _, _, VT_test = svd(A)
    # L_test = VT_test[-1, :] / VT_test[-1, -1]
    L_test = VT_test[-1, :]
    # print("\nL test second norm is: {}\n".format(np.linalg.norm(L_test, ord=2)))
    H_test = L_test.reshape(-1, 3, 3)
    for i, H in enumerate(H_test):
        H_test[i] = np.linalg.inv(T_dst).dot(H.dot(T_src))
    H_test = H_test / H_test[:, -1, -1].reshape(-1, 1, 1)
    # print("{} points least square computes a homography:\n {}\n".format(A.shape[0]//2, H_test))
    """

    Ax = []
    for i in range(0, len(A), 2):
        Ax_k = [A[i:i + 2] * weight for weight in weights[i // 2]]
        Ax_k_concat = reduce(lambda x, y: np.concatenate((x, y), axis=1), Ax_k)
        Ax.append(np.array(Ax_k_concat))
    Ax = np.array(Ax)
    Ax = Ax.reshape(-1, 9 * hom_block_num)
    # add regulariztion part
    reg_array = np.concatenate((np.eye(9, dtype=np.float32), -1 * np.eye(9, dtype=np.float32)), axis=1)
    regularizer = np.zeros(((hom_block_num - 1) * 9, hom_block_num * 9))
    for i in range(hom_block_num - 1):
        regularizer[i * 9:(i + 1) * 9, i * 9:(i + 2) * 9] = reg_array
    Ax = np.concatenate((Ax, regularizer))

    U, sigma, VT = svd(Ax)
    L = VT[-1, :]
    # print("\nL second norm is: {}\n".format(np.linalg.norm(L, ord=2)))
    H = L.reshape(-1, 3, 3)
    for i, h in enumerate(H):
        H[i] = np.linalg.inv(T_dst).dot(h.dot(T_src))
        # H[i] = T_dst.T.dot(h.dot(T_src))
    homography_mixtures = H / H[:, -1, -1].reshape(-1, 1, 1)
    # print(homography_mixtures)
    return homography_mixtures


def count_keypoints_per_patch(kpts, img, hom_block_num):
    kp_per_patch = OrderedDict()

    for i in range(hom_block_num):
        kp_per_patch[i] = 0

    block_height = img.shape[0] // hom_block_num

    for kpt_height in kpts[:, 1]:
        pos = kpt_height // block_height
        kp_per_patch[pos] += 1

    for k, v in kp_per_patch.items():
        print("patch {} contains {} keypoints\n".format(k, v))


def reject_outliers(src_pts, dst_pts, error_threhold):
    assert len(src_pts) >= 4, "keypoints number should large than 4"
    h, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, error_threhold)
    dst_pts_1 = h.dot(src_pts.T).T
    # normalize to homogeous system
    dst_pts_1[:, :2] = dst_pts_1[:, :2] / (dst_pts_1[:, -1].reshape((-1, 1)))
    # compute the mean translation
    mean_translation = np.sqrt(np.sum((dst_pts[:, :2] - dst_pts_1[:, :2]) ** 2)) / len(dst_pts_1)
    error = np.sqrt(np.sum((dst_pts[:, :2] - dst_pts_1[:, :2]) ** 2, 1))
    inliers_index = np.argwhere(error < error_threhold + mean_translation).squeeze()
    # print("number of inliers is {}/{}".format(len(inliers_index), len(src_pts)))
    return src_pts[inliers_index], dst_pts[inliers_index]


def reject_outliers_per_patch(src_pts, dst_pts, block_height, hom_block_num, error_threhold):
    src_pts_list = []
    dst_pts_list = []

    for i in range(1, hom_block_num + 1):
        index = np.argwhere((src_pts[:, 1] <= i * block_height) & (src_pts[:, 1] > (i - 1) * block_height))
        # print("block height: {}, contains {} kps".format(i * block_height, len(index)))
        try:
            _src_pts, _dst_pts = reject_outliers(src_pts[index.squeeze()], dst_pts[index.squeeze()], error_threhold)
        except Exception as e:
            print(e)
        src_pts_list.append(_src_pts)
        dst_pts_list.append(_dst_pts)

    src_pts_array = reduce(lambda x, y: np.concatenate((x, y), axis=0), src_pts_list)
    src_pts_array = np.array(src_pts_array).squeeze()
    dst_pts_array = reduce(lambda x, y: np.concatenate((x, y), axis=0), dst_pts_list)
    dst_pts_array = np.array(dst_pts_array).squeeze()
    return src_pts_array, dst_pts_array


def find_homography_mixtures(img, patch, feature_path=None):
    kpts_src = np.loadtxt(os.path.join(feature_path, "sourceOut.txt"), dtype='float_', delimiter=' ')
    kpts_target = np.loadtxt(os.path.join(feature_path, "targetOut.txt"), dtype='float_', delimiter=' ')
    keyPoint = (kpts_src, kpts_target)

    kpts_src = [np.squeeze(i).tolist() for i in keyPoint[0]]
    kpts_target = [np.squeeze(j).tolist() for j in keyPoint[1]]

    [i.append(1) for i in kpts_src]
    [j.append(1) for j in kpts_target]

    src_pts = np.array(kpts_src)
    dst_pts = np.array(kpts_target)

    src_pts_array, dst_pts_array = src_pts, dst_pts

    homography_mixtures = compute_homography_mixtures(src_pts_array, dst_pts_array, img, patch)
    return homography_mixtures


def transformImage(img, hom, patch):
    """
    :param img: 相邻两帧的第一个
    :param hom: 相邻两帧之间的Homography，shape: [patch, 3, 3]
    :param patch: 一帧垂直切成多少个patch
    :return: out_img warp之后的当前帧
    """
    height, width = img.shape[:2]
    num = height // patch
    out_img = np.zeros((height, width, 3))
    for row in range(hom.shape[0]):
        temp = cv2.warpPerspective(img, hom[row, :, :], (width, height))
        if row == patch - 1:
            out_img[row * num:] = temp[row * num:]
            continue
        out_img[row * num:row * num + num] = temp[row * num:row * num + num]
    return out_img


def make_gif(img1, img2, save_path, gif_name=""):
    '''
    IMG1_PATH: 原图的地址
    IMG2_PATH: warp_image's address
    '''
    TIME_GAP = 0.1
    with imageio.get_writer(os.path.join(save_path, "gifs/warp_{}.gif".format(gif_name)), mode='I',
                            duration=TIME_GAP) as writer:
        for image in [img1, img2]:
            writer.append_data(image)


def generate_keypoints(FRAME_PATH1, FRAME_PATH2):
    img1 = FRAME_PATH1
    img2 = FRAME_PATH2

    print(img1)
    print(img2)

    cmd = "./SGridSearch/bin/SGriSearch --input1 {} --input2 {}".format(img1, img2).split()

    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    popen.wait()
    output = popen.stdout.read()


def run(img_path_1, img_path_2, project_path='./'):
    generate_keypoints(img_path_1, img_path_2)

    img = [cv2.imread(i) for i in [img_path_1, img_path_2]]

    patch = 6
    homography_mixtures = find_homography_mixtures(img, patch=patch, feature_path=project_path)

    img_warp_mixtures = transformImage(img[0], homography_mixtures, patch=patch)

    # compute psnr
    _psnr_0 = cmp_psnr(np.float32(img[1][8:]), np.float32(img_warp_mixtures[8:]))
    print("\nimg_warp_mixtures PSNR is {}\n".format(_psnr_0))

    # gif maker
    make_gif(img[1][8:], img_warp_mixtures[8:], project_path, "img_warp_mixtures")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img1')
    parser.add_argument('--img2')

    args = parser.parse_args()
    run(args.img1, args.img2)



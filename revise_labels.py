
import os
import shutil
import numpy as np
from scipy import misc
from skimage import morphology, measure
import glob
import json


def main(dataset, partial_mask_dir):
    data_dir = './data/{:s}'.format(dataset)
    label_vor_dir = './data/{:s}/labels_voronoi_un'.format(dataset)
    label_cluster_dir = './data/{:s}/labels_cluster_un'.format(dataset)
    label_vor_dir_old = './data/{:s}/labels_voronoi'.format(dataset)
    label_cluster_dir_old = './data/{:s}/labels_cluster'.format(dataset)
    patch_folder = './data/{:s}/patches'.format(dataset)
    train_data_dir = './data_for_train/{:s}'.format(dataset)

    with open('{:s}/train_val_test.json'.format(data_dir), 'r') as file:
        data_list = json.load(file)
        train_list = data_list['train']

    # ------ revise the Voronoi label
    fuse_Voronoi_and_partial_mask(label_vor_dir_old, partial_mask_dir, label_vor_dir, train_list)
    #
    # ------ revise the cluster label
    fuse_cluster_and_partial_mask(label_cluster_dir_old, partial_mask_dir, label_cluster_dir, train_list)

    #
    # # ------ split large images into 250x250 patches
    print("Spliting large images into small patches...")
    split_patches(label_vor_dir, '{:s}/labels_voronoi'.format(patch_folder), 'label_vor')
    split_patches(label_cluster_dir, '{:s}/labels_cluster'.format(patch_folder), 'label_cluster')

    replace_training_labels(data_dir, train_data_dir)


def fuse_Voronoi_and_partial_mask(label_vor_dir, partial_mask_dir, save_dir, train_list):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    def get_point(img):
        a = np.where(img != 0)
        rmin, rmax, cmin, cmax = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
        return (rmin + rmax) // 2, (cmin + cmax) // 2

    print("Generating fused Voronoi label ...")
    N_total = len(train_list)
    N_processed = 0
    for img_name in sorted(train_list):
        if img_name not in train_list:
            continue

        N_processed += 1
        flag = '' if N_processed < N_total else '\n'
        print('\r\t{:d}/{:d}'.format(N_processed, N_total), end=flag)

        label_vor = misc.imread('{:s}/{:s}_label_vor.png'.format(label_vor_dir, img_name[:-4]))
        partial_mask = misc.imread('{:s}/{:s}_label_partial_mask.png'.format(partial_mask_dir, img_name[:-4]))

        areas_labeled = measure.label(255 - label_vor[:, :, 0])
        areas_indices = []
        mask_indices = np.unique(partial_mask)
        mask_indices = mask_indices[mask_indices != 0]
        for index in mask_indices:
            mask = partial_mask == index
            point = get_point(mask)
            areas_indices.append(areas_labeled[point[0], point[1]])
        areas_bg = np.isin(areas_labeled, areas_indices) * (partial_mask == 0)

        fused = np.zeros((partial_mask.shape[0], partial_mask.shape[1], 3), dtype=np.uint8)
        fused[:, :, 1] = ((label_vor[:, :, 1] > 0) + (partial_mask > 0)).astype(np.uint8) * 255
        fused[:, :, 0] = (((label_vor[:, :, 0] + areas_bg) > 0) * (fused[:, :, 1] == 0)).astype(np.uint8) * 255

        misc.imsave('{:s}/{:s}_label_vor.png'.format(save_dir, img_name[:-4]), fused)


def fuse_cluster_and_partial_mask(label_cluster_dir, partial_mask_dir, save_dir, train_list):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    print("Generating fused cluster label ...")
    N_total = len(train_list)
    N_processed = 0
    for img_name in sorted(train_list):
        if img_name not in train_list:
            continue

        N_processed += 1
        flag = '' if N_processed < N_total else '\n'
        print('\r\t{:d}/{:d}'.format(N_processed, N_total), end=flag)

        label_cluster = misc.imread('{:s}/{:s}_label_cluster.png'.format(label_cluster_dir, img_name[:-4]))

        partial_mask = misc.imread('{:s}/{:s}_label_partial_mask.png'.format(partial_mask_dir, img_name[:-4]))
        partial_mask = (partial_mask > 0).astype(np.uint8) * 255
        label_partial_mask = np.zeros((partial_mask.shape[0], partial_mask.shape[1], 3), dtype=np.uint8)
        label_partial_mask[:, :, 0] = morphology.dilation(partial_mask, morphology.disk(2)) - partial_mask
        label_partial_mask[:, :, 1] = partial_mask

        flag_mask = morphology.dilation(partial_mask, morphology.disk(5)) > 0
        flag_mask = np.repeat(flag_mask[...,None], 3, axis=2).astype(np.uint8)
        fused = label_cluster * (1-flag_mask) + label_partial_mask * flag_mask

        misc.imsave('{:s}/{:s}_label_cluster.png'.format(save_dir, img_name[:-4]), fused)


def extract_random_mask_from_instance(data_dir, save_dir, train_list, ratio=0.05):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    print("Generating random partial mask from instance label...")
    image_list = os.listdir(data_dir)
    N_total = len(train_list)
    N_processed = 0
    for image_name in sorted(image_list):
        name = image_name.split('.')[0]
        if '{:s}.png'.format(name[:-6]) not in train_list or name[-5:] != 'label':
            continue

        N_processed += 1
        flag = '' if N_processed < N_total else '\n'
        print('\r\t{:d}/{:d}'.format(N_processed, N_total), end=flag)

        image_path = os.path.join(data_dir, image_name)
        image = misc.imread(image_path)

        # randomly keep a ratio of full masks
        ids = np.unique(image[image > 0])
        np.random.seed(1)
        seleted_ids = np.random.choice(ids, (round(ids.size * ratio), 1))
        partial_mask = np.zeros_like(image)
        counter = 1
        for id in seleted_ids:
            partial_mask[image == id] = counter
            counter += 1

        misc.imsave('{:s}/{:s}_partial_mask.png'.format(save_dir, name), partial_mask)
        misc.imsave('{:s}/{:s}_partial_mask_binary.png'.format(save_dir, name), (partial_mask>0).astype(np.uint8) * 255)


def split_patches(data_dir, save_dir, post_fix=None):
    import math
    """ split large image into small patches """
    create_folder(save_dir)

    image_list = os.listdir(data_dir)
    for image_name in image_list:
        name = image_name.split('.')[0]
        if post_fix and name[-len(post_fix):] != post_fix:
            continue
        image_path = os.path.join(data_dir, image_name)
        image = misc.imread(image_path)
        seg_imgs = []

        # split into 16 patches of size 250x250
        h, w = image.shape[0], image.shape[1]
        patch_size = 250
        h_overlap = math.ceil((4 * patch_size - h) / 3)
        w_overlap = math.ceil((4 * patch_size - w) / 3)
        for i in range(0, h-patch_size+1, patch_size-h_overlap):
            for j in range(0, w-patch_size+1, patch_size-w_overlap):
                if len(image.shape) == 3:
                    patch = image[i:i+patch_size, j:j+patch_size, :]
                else:
                    patch = image[i:i + patch_size, j:j + patch_size]
                seg_imgs.append(patch)

        for k in range(len(seg_imgs)):
            if post_fix:
                misc.imsave('{:s}/{:s}_{:d}_{:s}.png'.format(save_dir, name[:-len(post_fix)-1], k, post_fix), seg_imgs[k])
            else:
                misc.imsave('{:s}/{:s}_{:d}.png'.format(save_dir, name, k), seg_imgs[k])


def replace_training_labels(data_dir, train_data_dir):

    print('Replacing training labels...')
    with open('{:s}/train_val_test.json'.format(data_dir), 'r') as file:
        data_list = json.load(file)
        train_list, val_list, test_list = data_list['train'], data_list['val'], data_list['test']

    for img_name in train_list:
        name = img_name.split('.')[0]
        # label_voronoi
        for file in glob.glob('{:s}/patches/labels_voronoi/{:s}*'.format(data_dir, name)):
            file_name = file.split('/')[-1]
            dst = '{:s}/labels_voronoi/train/{:s}'.format(train_data_dir, file_name)
            shutil.copyfile(file, dst)
        # label_cluster
        for file in glob.glob('{:s}/patches/labels_cluster/{:s}*'.format(data_dir, name)):
            file_name = file.split('/')[-1]
            dst = '{:s}/labels_cluster/train/{:s}'.format(train_data_dir, file_name)
            shutil.copyfile(file, dst)


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)


if __name__ == '__main__':
    dataset = 'LC'
    partial_mask_dir = './data/{:s}/selected_masks'.format(dataset)
    main(dataset, partial_mask_dir)


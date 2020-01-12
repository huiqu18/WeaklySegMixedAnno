
import os, json
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from PIL import Image
from skimage import measure
from scipy import misc
import matplotlib.pyplot as plt
import skimage.morphology as morph
from scipy.ndimage import gaussian_filter
from model import ResUNet34
import utils

from my_transforms import get_transforms


def main(opt, save_dir):
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.test['gpus'])

    # img_dir = opt.test['img_dir']
    ratio = opt.ratio
    img_dir = './data/{:s}/images'.format(opt.dataset)
    label_dir = './data/{:s}/labels_point'.format(opt.dataset)
    label_instance_dir = './data/{:s}/labels_instance'.format(opt.dataset)
    # save_dir = './data/{:s}/selected_masks'.format(opt.dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    model_path = opt.test['model_path']

    # data transforms
    test_transform = get_transforms(opt.transform['test'])
    
    model = ResUNet34(pretrained=opt.model['pretrained'], with_uncertainty=opt.with_uncertainty)
    model = model.cuda()
    cudnn.benchmark = True

    # ----- load trained model ----- #
    # print("=> loading trained model")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    # print("=> loaded model at epoch {}".format(checkpoint['epoch']))

    # switch to evaluate mode
    model.eval()
    apply_dropout(model)

    with open('./data/{:s}/train_val_test.json'.format(opt.dataset), 'r') as file:
        data_list = json.load(file)
        train_list = data_list['train']

    for img_name in tqdm(train_list):
        # load test image
        # print('=> Processing image {:s}'.format(img_name))
        img_path = '{:s}/{:s}'.format(img_dir, img_name)
        img = Image.open(img_path)
        ori_h = img.size[1]
        ori_w = img.size[0]
        name = os.path.splitext(img_name)[0]
        label_point = misc.imread('{:s}/{:s}_label_point.png'.format(label_dir, name))

        input = test_transform((img,))[0].unsqueeze(0)
        # print('\tComputing unertainty maps...')
        mean_sigma = np.zeros((2, ori_h, ori_w))
        mean_sigma_normalized = np.zeros((2, ori_h, ori_w))
        mean_prob = np.zeros((2, ori_h, ori_w))
        for _ in range(opt.T):
            output, log_var = get_probmaps(input, model, opt)
            output = output.astype(np.float64)
            log_var = log_var.astype(np.float64)
            sigma_map = np.exp(log_var / 2)
            sigma_map_normalized = sigma_map / (np.exp(output) + 1e-8)

            mean_prob += np.exp(output) / np.sum(np.exp(output), axis=0)
            mean_sigma += sigma_map
            mean_sigma_normalized += sigma_map_normalized

        mean_prob /= opt.T
        mean_sigma /= opt.T
        mean_sigma_normalized /= opt.T

        un_data_normalized = mean_sigma_normalized ** 2

        pred = np.argmax(mean_prob, axis=0)
        un_data_normalized = np.sum(un_data_normalized * utils.onehot_encoding(pred), axis=0)

        # find the area of largest uncertainty for visualization
        threshed = un_data_normalized > 1.0
        large_unc_area = morph.opening(threshed, selem=morph.disk(1))
        large_unc_area = morph.remove_small_objects(large_unc_area, min_size=64)
        un_data_smoothed = gaussian_filter(un_data_normalized * large_unc_area, sigma=5)

        # cmap = plt.cm.jet
        # plt.imsave('{:s}/{:s}_uncertainty.png'.format(save_dir, name), cmap(un_data_normalized))

        points = measure.label(label_point)
        uncertainty_list = []
        radius = 10
        for k in range(1, np.max(points)+1):
            x, y = np.argwhere(points == k)[0]
            r1 = x - radius if x - radius > 0 else 0
            r2 = x + radius if x + radius < ori_h else ori_h
            c1 = y - radius if y - radius > 0 else 0
            c2 = y + radius if y + radius < ori_w else ori_w
            uncertainty = np.mean(un_data_smoothed[r1:r2, c1:c2])
            uncertainty_list.append([k, uncertainty])

        uncertainty_list = np.array(uncertainty_list)
        sorted_list = uncertainty_list[uncertainty_list[:,1].argsort()[::-1]]
        indices = sorted_list[:int(ratio*np.max(points)), 0]

        # annotation
        label_instance = misc.imread('{:s}/{:s}_label.png'.format(label_instance_dir, name))
        new_anno = np.zeros_like(label_instance)
        counter = 1
        for idx in indices:
            nuclei_idx = np.unique(label_instance[points == idx])[0]
            if nuclei_idx == 0:
                continue
            new_anno += (label_instance == nuclei_idx) * counter
            counter += 1
            # utils.show_figures((new_anno,))

        misc.imsave('{:s}/{:s}_label_partial_mask.png'.format(save_dir, name), new_anno.astype(np.uint8))
        misc.imsave('{:s}/{:s}_label_partial_mask_binary.png'.format(save_dir, name), (new_anno>0).astype(np.uint8) * 255)

    print('=> Processed all images')


def get_probmaps(input, model, opt):
    size = opt.test['patch_size']
    overlap = opt.test['overlap']

    output, log_var = utils.split_forward(model, input, size, overlap, with_uncertainty=True)
    output = output.squeeze(0)
    log_var = log_var.squeeze(0)

    return output.cpu().numpy(), log_var.cpu().numpy()


def apply_dropout(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout) or isinstance(m, torch.nn.Dropout2d):
            m.train()


if __name__ == '__main__':
    from options import Options
    opt = Options(isTrain=False)
    save_dir = './data/{:s}/selected_masks'.format(opt.dataset)
    main(opt, save_dir)

import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import shutil
from PIL import Image
import numpy as np
import random
import logging
from scipy.ndimage.morphology import binary_fill_holes
from skimage import measure
from scipy import misc
import skimage.morphology as ski_morph

from model import ResUNet34
import utils
from accuracy import compute_metrics
from dataset import DataFolder
from my_transforms import get_transforms


def main(opt):
    global best_score, logger, logger_results
    best_score = 0
    opt.save_options()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.train['gpus'])

    # set up logger
    logger, logger_results = setup_logging(opt)
    opt.print_options(logger)

    if opt.train['random_seed'] >= 0:
        # logger.info("=> Using random seed {:d}".format(opt.train['random_seed']))
        torch.manual_seed(opt.train['random_seed'])
        torch.cuda.manual_seed(opt.train['random_seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(opt.train['random_seed'])
        random.seed(opt.train['random_seed'])
    else:
        torch.backends.cudnn.benchmark = True

    # ----- create model ----- #
    model = ResUNet34(pretrained=opt.model['pretrained'], with_uncertainty=opt.with_uncertainty)
    # model = nn.DataParallel(model)
    model = model.cuda()

    # ----- define optimizer ----- #
    optimizer = torch.optim.Adam(model.parameters(), opt.train['lr'], betas=(0.9, 0.99),
                                 weight_decay=opt.train['weight_decay'])

    # ----- define criterion ----- #
    criterion = torch.nn.NLLLoss(ignore_index=2).cuda()

    # ----- load data ----- #
    data_transforms = {'train': get_transforms(opt.transform['train']),
                       'val': get_transforms(opt.transform['val'])}

    img_dir = '{:s}/train'.format(opt.train['img_dir'])
    target_vor_dir = '{:s}/train'.format(opt.train['label_vor_dir'])
    target_cluster_dir = '{:s}/train'.format(opt.train['label_cluster_dir'])
    dir_list = [img_dir, target_vor_dir, target_cluster_dir]
    post_fix = ['label_vor.png', 'label_cluster.png']
    num_channels = [3, 3, 3]
    train_set = DataFolder(dir_list, post_fix, num_channels, data_transforms['train'])
    train_loader = DataLoader(train_set, batch_size=opt.train['batch_size'], shuffle=True,
                              num_workers=opt.train['workers'])

    # ----- optionally load from a checkpoint for validation or resuming training ----- #
    if opt.train['checkpoint']:
        if os.path.isfile(opt.train['checkpoint']):
            logger.info("=> loading checkpoint '{}'".format(opt.train['checkpoint']))
            checkpoint = torch.load(opt.train['checkpoint'])
            opt.train['start_epoch'] = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(opt.train['checkpoint'], checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(opt.train['checkpoint']))

    # ----- training and validation ----- #
    num_epochs = opt.train['num_epochs']

    for epoch in range(opt.train['start_epoch'], num_epochs):
        # train for one epoch or len(train_loader) iterations
        logger.info('Epoch: [{:d}/{:d}]'.format(epoch+1, num_epochs))
        train_loss, train_loss_vor, train_loss_cluster = train(opt, train_loader, model, optimizer, criterion)

        # evaluate on val set
        with torch.no_grad():
            val_acc, val_aji = validate(opt, model, data_transforms['val'])

        # check if it is the best accuracy
        is_best = val_aji > best_score
        best_score = max(val_aji, best_score)

        cp_flag = (epoch+1) % opt.train['checkpoint_freq'] == 0
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, epoch, opt.train['save_dir'], is_best, cp_flag)

        # save the training results to txt files
        logger_results.info('{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
                            .format(epoch+1, train_loss, train_loss_vor, train_loss_cluster,
                                    val_acc, val_aji))

    for i in list(logger.handlers):
        logger.removeHandler(i)
        i.flush()
        i.close()
    for i in list(logger_results.handlers):
        logger_results.removeHandler(i)
        i.flush()
        i.close()


def train(opt, train_loader, model, optimizer, criterion):
    # list to store the average loss for this epoch
    results = utils.AverageMeter(3)

    # switch to train mode
    model.train()
    for i, sample in enumerate(train_loader):
        input, target1, target2 = sample

        # for b in range(input.size(0)):
        #     utils.show_figures((input[b].numpy().transpose(1, 2, 0), target1[b,0,:,:].numpy(), target2[b,0,:,:]))

        input = input.cuda()
        target1 = target1.squeeze(1)
        target2 = target2.squeeze(1)

        if opt.with_uncertainty:
            fx, log_var = model(input)
            log_var = torch.clamp(log_var, max=160)  # avoid inf value for float32 type
            prob_maps = torch.zeros(fx.size()).cuda()
            sigma = torch.exp(log_var / 2)
            for t in range(opt.T):
                x_t = fx + sigma * torch.randn(fx.size()).cuda()
                prob_maps += F.softmax(x_t, dim=1)
            prob_maps /= opt.T
            log_prob_maps = torch.log(prob_maps + 1e-8)
        else:
            output = model(input)
            log_prob_maps = F.log_softmax(output, dim=1)

        loss_vor = criterion(log_prob_maps, target1.cuda())
        loss_cluster = criterion(log_prob_maps, target2.cuda())
        loss = loss_vor + loss_cluster

        result = [loss.item(), loss_vor.item(), loss_cluster.item()]
        results.update(result, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % opt.train['log_interval'] == 0:
            logger.info('\tIteration: [{:d}/{:d}]'
                        '\tLoss {r[0]:.4f}'
                        '\tLoss_vor {r[1]:.4f}'
                        '\tLoss_cluster {r[2]:.4f}'.format(i, len(train_loader), r=results.avg))

    logger.info('\t=> Train Avg: Loss {r[0]:.4f}'
                '\tloss_vor {r[1]:.4f}'
                '\tloss_cluster {r[2]:.4f}'.format(r=results.avg))

    return results.avg


def validate(opt, model, test_transform):
    # list to store the losses and accuracies: [pixel_acc, aji ]
    results = utils.AverageMeter(2)

    # switch to evaluate mode
    model.eval()

    img_dir = '{:s}/val'.format(opt.train['img_dir'])
    label_dir = opt.test['label_dir']

    img_names = os.listdir(img_dir)
    for img_name in img_names:
        img_path = '{:s}/{:s}'.format(img_dir, img_name)
        img = Image.open(img_path)
        name = os.path.splitext(img_name)[0]

        label_path = '{:s}/{:s}_label.png'.format(label_dir, name)
        gt = misc.imread(label_path)

        input = test_transform((img,))[0].unsqueeze(0)

        if opt.with_uncertainty:
            output, log_var = get_probmaps(input, model, opt)
            output = output.astype(np.float32)
            log_var = log_var.astype(np.float32)
            log_var = np.clip(log_var, a_min=np.min(log_var), a_max=700)  # avoid inf value for float32 type
            prob_maps = np.zeros(output.shape)
            sigma = np.exp(log_var / 2)
            sigma = np.clip(sigma, a_min=0, a_max=700)  # avoid inf value for float32 type
            for t in range(opt.T):
                x_t = output + sigma * np.random.normal(0, 1, output.shape)
                x_t = np.clip(x_t, a_min=0, a_max=700)
                prob_maps += np.exp(x_t) / (np.sum(np.exp(x_t), axis=0) + 1e-8)
            prob_maps /= opt.T
        else:
            prob_maps = get_probmaps(input, model, opt)

        pred = np.argmax(prob_maps, axis=0)  # prediction

        pred_labeled = measure.label(pred)
        pred_labeled = ski_morph.remove_small_objects(pred_labeled, opt.post['min_area'])
        pred_labeled = binary_fill_holes(pred_labeled > 0)
        pred_labeled = measure.label(pred_labeled)

        metrics = compute_metrics(pred_labeled, gt, ['acc', 'aji'])
        result = [metrics['acc'], metrics['aji']]

        results.update(result, input.size(0))

    logger.info('\t=> Val Avg:\tAcc {r[0]:.4f}\tAJI {r[1]:.4f}'.format(r=results.avg))

    return results.avg


def get_probmaps(input, model, opt):
    size = opt.test['patch_size']
    overlap = opt.test['overlap']

    if opt.with_uncertainty:
        output, log_var = utils.split_forward(model, input, size, overlap, with_uncertainty=True)
        output = output.squeeze(0)
        log_var = log_var.squeeze(0)
        return output.cpu().numpy(), log_var.cpu().numpy()
    else:
        output = utils.split_forward(model, input, size, overlap, with_uncertainty=False)
        output = output.squeeze(0)
        prob_maps = F.softmax(output, dim=0).cpu().numpy()
        return prob_maps


def save_checkpoint(state, epoch, save_dir, is_best, cp_flag):
    cp_dir = '{:s}/checkpoints'.format(save_dir)
    if not os.path.exists(cp_dir):
        os.mkdir(cp_dir)
    filename = '{:s}/checkpoint.pth.tar'.format(cp_dir)
    torch.save(state, filename)
    if cp_flag:
        shutil.copyfile(filename, '{:s}/checkpoint_{:d}.pth.tar'.format(cp_dir, epoch+1))
    if is_best:
        shutil.copyfile(filename, '{:s}/checkpoint_best.pth.tar'.format(cp_dir))


def setup_logging(opt):
    mode = 'a' if opt.train['checkpoint'] else 'w'

    # create logger for training information
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.DEBUG)
    # create console handler and file handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler('{:s}/train_log.txt'.format(opt.train['save_dir']), mode=mode)
    file_handler.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s\t%(message)s', datefmt='%m-%d %I:%M')
    # add formatter to handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    # add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # create logger for epoch results
    logger_results = logging.getLogger('results')
    logger_results.setLevel(logging.DEBUG)
    file_handler2 = logging.FileHandler('{:s}/epoch_results.txt'.format(opt.train['save_dir']), mode=mode)
    file_handler2.setFormatter(logging.Formatter('%(message)s'))
    logger_results.addHandler(file_handler2)

    logger.info('***** Training starts *****')
    logger.info('save directory: {:s}'.format(opt.train['save_dir']))
    if mode == 'w':
        logger_results.info('epoch\ttrain_loss\ttrain_loss_vor\ttrain_loss_cluster\tval_acc\tval_AJI')

    return logger, logger_results


if __name__ == '__main__':
    from options import Options
    opt = Options(isTrain=True)
    opt.parse()
    main(opt)

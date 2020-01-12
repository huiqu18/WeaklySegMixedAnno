import os
import numpy as np
import argparse
from collections import OrderedDict


class Options:
    def __init__(self, dataset='MO', isTrain=True):
        self.dataset = dataset    # LC: Lung Cancer, MO: MultiOrgan
        self.isTrain = isTrain
        self.root_save_dir = './experiments/{:s}'.format(self.dataset)  # path to save experimental results
        self.with_uncertainty = True
        self.T = 20    # number of MC dropout sampling
        self.ratio = 0.05    # ratio of nuclei to be annotated with masks

        # --- model hyper-parameters --- #
        self.model = dict()
        self.model['name'] = 'ResUNet34'
        self.model['pretrained'] = True
        self.model['fix_params'] = False
        self.model['in_c'] = 3  # input channel

        # --- training params --- #
        self.train = dict()
        self.train['random_seed'] = -1
        self.train['data_dir'] = './data_for_train/{:s}'.format(self.dataset)  # path to data
        self.train['save_dir'] = self.root_save_dir  # path to save train results
        self.train['input_size'] = 224         # input size of the image
        self.train['num_epochs'] = 100       # number of training epochs
        self.train['batch_size'] = 8           # batch size
        self.train['checkpoint_freq'] = 50      # epoch to save checkpoints
        self.train['lr'] = 0.0001               # initial learning rate
        self.train['weight_decay'] = 1e-4       # weight decay
        self.train['log_interval'] = 30         # iterations to print training results
        self.train['workers'] = 1               # number of workers to load images
        self.train['gpus'] = [0, ]              # select gpu devices
        # --- resume training --- #
        self.train['start_epoch'] = 0    # start epoch
        self.train['checkpoint'] = None

        # --- data transform --- #
        self.transform = dict()

        # --- test parameters --- #
        self.test = dict()
        self.test['test_epoch'] = 'best'
        self.test['gpus'] = [0, ]
        self.test['img_dir'] = './data_for_train/{:s}/images/test'.format(self.dataset)
        self.test['label_dir'] = './data/{:s}/labels_instance'.format(self.dataset)
        self.test['save_flag'] = True
        self.test['patch_size'] = 224
        self.test['overlap'] = 80
        self.test['save_dir'] = '{:s}/{:s}'.format(self.root_save_dir, self.test['test_epoch'])
        self.test['model_path'] = '{:s}/checkpoints/checkpoint_{:s}.pth.tar'.format(self.root_save_dir, self.test['test_epoch'])
        # --- post processing --- #
        self.post = dict()
        self.post['min_area'] = 20  # minimum area for an object

    def parse(self):
        """ Parse the options, replace the default value if there is a new input """
        parser = argparse.ArgumentParser(description='')
        if self.isTrain:
            parser.add_argument('--dataset', type=str, default=self.dataset, help='dataset')
            parser.add_argument('--save-dir', type=str, default=self.root_save_dir, help='directory to save training results')
            parser.add_argument('--batch-size', type=int, default=self.train['batch_size'], help='input batch size for training')
            parser.add_argument('--epochs', type=int, default=self.train['num_epochs'], help='number of epochs to train')
            parser.add_argument('--lr', type=float, default=self.train['lr'], help='learning rate')
            parser.add_argument('--random-seed', type=int, default=self.train['random_seed'], help='random seed')
            parser.add_argument('--log-interval', type=int, default=self.train['log_interval'], help='how many batches to wait before logging training status')
            parser.add_argument('--gpus', type=list, default=self.train['gpus'], help='GPUs for training')
            parser.add_argument('--data-dir', type=str, default=self.train['data_dir'], help='directory of training data')
            parser.add_argument('--checkpoint-path', type=str, default=self.train['checkpoint'], help='directory to load a checkpoint')
            args = parser.parse_args()

            self.dataset = args.dataset
            self.train['batch_size'] = args.batch_size
            self.train['random_seed'] = args.random_seed
            self.train['num_epochs'] = args.epochs
            self.train['lr'] = args.lr
            self.train['log_interval'] = args.log_interval
            self.train['gpus'] = args.gpus
            self.train['data_dir'] = args.data_dir
            self.train['img_dir'] = '{:s}/images'.format(self.train['data_dir'])
            self.train['label_vor_dir'] = '{:s}/labels_voronoi'.format(self.train['data_dir'])
            self.train['label_cluster_dir'] = '{:s}/labels_cluster'.format(self.train['data_dir'])

            self.train['checkpoint'] = args.checkpoint_path

            self.root_save_dir = args.save_dir
            if not os.path.exists(self.root_save_dir):
                os.makedirs(self.root_save_dir, exist_ok=True)
            self.train['save_dir'] = self.root_save_dir

            # define data transforms for training
            self.transform['train'] = OrderedDict()
            self.transform['train'] = {
                'random_resize': [0.8, 1.25],
                'horizontal_flip': True,
                'random_affine': 0.3,
                'random_rotation': 90,
                'random_crop': self.train['input_size'],
                'label_encoding': 2,
                'to_tensor': 1,
                'normalize': np.load('{:s}/mean_std.npy'.format(self.train['data_dir']))
            }
            self.transform['val'] = OrderedDict()
            self.transform['val'] = {
                'to_tensor': 1,
                'normalize': np.load('{:s}/mean_std.npy'.format(self.train['data_dir']))
            }

        else:
            parser.add_argument('--dataset', type=str, default=self.dataset, help='dataset')
            parser.add_argument('--save-flag', type=bool, default=self.test['save_flag'], help='flag to save the network outputs and predictions')
            parser.add_argument('--gpus', type=list, default=self.test['gpus'], help='GPUs for training')
            parser.add_argument('--img-dir', type=str, default=self.test['img_dir'], help='directory of test images')
            parser.add_argument('--label-dir', type=str, default=self.test['label_dir'], help='directory of labels')
            parser.add_argument('--save-dir', type=str, default=self.test['save_dir'], help='directory to save test results')
            parser.add_argument('--model-path', type=str, default=self.test['model_path'], help='train model to be evaluated')
            args = parser.parse_args()
            self.dataset = args.dataset
            self.test['gpus'] = args.gpus
            self.test['save_flag'] = args.save_flag
            self.test['img_dir'] = args.img_dir
            self.test['label_dir'] = args.label_dir
            self.test['save_dir'] = args.save_dir
            self.test['model_path'] = args.model_path

            if not os.path.exists(self.test['save_dir']):
                os.makedirs(self.test['save_dir'], exist_ok=True)

            self.transform['test'] = OrderedDict()
            self.transform['test'] = {
                'to_tensor': 1,
                'normalize': np.load('{:s}/mean_std.npy'.format(self.train['data_dir']))
            }

    def print_options(self, logger=None):
        message = '\n'
        message += self._generate_message_from_options()
        if not logger:
            print(message)
        else:
            logger.info(message)

    def save_options(self):
        if self.isTrain:
            filename = '{:s}/train_options.txt'.format(self.train['save_dir'])
        else:
            filename = '{:s}/test_options.txt'.format(self.test['save_dir'])
        message = self._generate_message_from_options()
        file = open(filename, 'w')
        file.write(message)
        file.close()

    def _generate_message_from_options(self):
        message = ''
        message += '# {str:s} Options {str:s} #\n'.format(str='-'*25)
        train_groups = ['model', 'train', 'transform']
        test_groups = ['model', 'test', 'post', 'transform']
        cur_group = train_groups if self.isTrain else test_groups

        for group, options in self.__dict__.items():
            if group not in train_groups + test_groups:
                message += '{:>20}: {:<35}\n'.format(group, str(options))
            elif group in cur_group:
                message += '\n{:s} {:s} {:s}\n'.format('*' * 15, group, '*' * 15)
                if group == 'transform':
                    for name, val in options.items():
                        if (self.isTrain and name != 'test') or (not self.isTrain and name == 'test'):
                            message += '{:s}:\n'.format(name)
                            for t_name, t_val in val.items():
                                t_val = str(t_val).replace('\n', ',\n{:22}'.format(''))
                                message += '{:>20}: {:<35}\n'.format(t_name, str(t_val))
                else:
                    for name, val in options.items():
                        message += '{:>20}: {:<35}\n'.format(name, str(val))
        message += '# {str:s} End {str:s} #\n'.format(str='-'*26)
        return message

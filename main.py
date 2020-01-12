
from options import Options
from prepare_data import main as prepare_data
from train import main as train
from mask_selection import main as mask_selection
from revise_labels import main as revise_labels
from test import main as test


opt = Options(dataset='MO')

# ----- Step 1: Uncertainty prediction ----- #
print('Preparing training data for {:s} dataset'.format(opt.dataset))
prepare_data(opt)

opt.isTrain = True
opt.with_uncertainty = True
opt.root_save_dir = './experiments/{:s}/step1'.format(opt.dataset)
opt.train['epochs'] = 100
opt.parse()
train(opt)

opt.isTrain = False
opt.test['model_path'] = '{:s}/checkpoints/checkpoint_best.pth.tar'.format(opt.root_save_dir)
mask_dir = '{:s}/selected_masks'.format(opt.root_save_dir)
opt.test['save_dir'] = mask_dir
opt.parse()
print('Selecting nuclei for mask annotation')
mask_selection(opt, save_dir=mask_dir)

# ----- Step 2: Training with mixed annotation ----- #
print('Revising Voronoi and cluster labels using the selected masks')
revise_labels(opt.dataset, partial_mask_dir=mask_dir)

# training with mixed labels
opt.isTrain = True
opt.with_uncertainty = False
opt.root_save_dir = './experiments/{:s}/step2'.format(opt.dataset)
opt.train['num_epochs'] = 150
opt.parse()
train(opt)

# test
opt.isTrain = False
opt.test['save_dir'] = '{:s}/best'.format(opt.root_save_dir)
opt.test['model_path'] = '{:s}/checkpoints/checkpoint_best.pth.tar'.format(opt.root_save_dir)
opt.parse()
test(opt)

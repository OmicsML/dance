"""Main functionality for starting training.

This code is based on https://github.com/NVlabs/MUNIT.

"""
import argparse
import os
import random

import torch
from sklearn import preprocessing

from dance.datasets.multimodality import ModalityMatchingDataset
from dance.modules.multi_modality.match_modality.cmae import CMAE
from dance.utils import set_seed


def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory


if __name__ == '__main__':
    rndseed = random.randint(0, 2147483647)
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='./match_modality/output', help="outputs path")
    parser.add_argument('-d', '--data_folder', default='./data/modality_matching')
    parser.add_argument("--resume", action="store_true")
    parser.add_argument('-t', '--subtask', default='openproblems_bmmc_cite_phase2_rna')
    parser.add_argument('-device', '--device', default='cuda')
    parser.add_argument('-cpu', '--cpus', default=1, type=int)
    parser.add_argument('-seed', '--rnd_seed', default=rndseed, type=int)
    parser.add_argument('-pk', '--pickle_suffix', default='_lsi_input_pca_count.pkl')

    parser.add_argument('--max_epochs', default=50, type=int, help='maximum number of training epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--log_data', default=True, type=bool, help='take a log1p of the data as input')
    parser.add_argument('--normalize_data', default=True, type=bool,
                        help='normalize the data (after the log, if applicable)')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--beta1', default=0.5, type=float, help='Adam parameter')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam parameter')
    parser.add_argument('--init', default='kaiming', type=str,
                        help='initialization [gaussian/kaiming/xavier/orthogonal]')
    parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--lr_policy', default='step', type=str, help='learning rate scheduler')
    parser.add_argument('--step_size', default=100000, type=int, help='how often to decay learning rate')
    parser.add_argument('--gamma', default=0.5, type=float, help='how much to decay learning rate')
    parser.add_argument('--gan_w', default=10, type=int, help='weight of adversarial loss')
    parser.add_argument('--recon_x_w', default=10, type=int, help='weight of image reconstruction loss')
    parser.add_argument('--recon_h_w', default=0, type=int, help='weight of hidden reconstruction loss')
    parser.add_argument('--recon_kl_w', default=0, type=int, help='weight of KL loss for reconstruction')
    parser.add_argument('--supervise', default=1, type=float, help='fraction to supervise')
    parser.add_argument('--super_w', default=0.1, type=float, help='weight of supervision loss')

    opts = parser.parse_args()

    torch.set_num_threads(opts.cpus)
    rndseed = opts.rnd_seed
    set_seed(rndseed)
    pkl_path = opts.subtask + opts.pickle_suffix
    dataset = ModalityMatchingDataset(
        opts.subtask, data_dir=opts.data_folder).load_data().load_sol().preprocess(kind='feature_selection')
    device = opts.device

    # Setup logger and output folders
    output_directory = os.path.join(opts.output_path, "outputs")
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)

    le = preprocessing.LabelEncoder()
    train_batch = dataset.modalities[0].obs['batch']
    batch = le.fit_transform(train_batch)
    train_batch = torch.from_numpy(batch).long().to(device)
    train_mod1 = torch.from_numpy(dataset.numpy_features(0)).float().to(device)
    train_mod2 = torch.from_numpy(dataset.numpy_features(1)).float().to(device)
    test_mod1 = torch.from_numpy(dataset.numpy_features(2)).float().to(device)
    test_mod2 = torch.from_numpy(dataset.numpy_features(3)).float().to(device)
    z_test = torch.from_numpy(dataset.test_sol.X.toarray())

    config = vars(opts)
    # Some Fixed Settings
    config['input_dim_a'] = dataset.modalities[0].shape[1]
    config['input_dim_b'] = dataset.modalities[1].shape[1]
    config['resume'] = opts.resume
    config['num_of_classes'] = max(batch) + 1
    config['shared_layer'] = True
    config['gen'] = {
        'dim': 100,  # hidden layer
        'latent': 50,  # latent layer size
        'activ': 'relu',
    }  # activation function [relu/lrelu/prelu/selu/tanh]
    config['dis'] = {
        'dim': 100,
        'norm': None,  # normalization layer [none/bn/in/ln]
        'activ': 'lrelu',  # activation function [relu/lrelu/prelu/selu/tanh]
        'gan_type': 'lsgan',
    }  # GAN loss [lsgan/nsgan]

    model = CMAE(config)
    model.to(device)

    model.fit(train_mod1, train_mod2, checkpoint_directory=checkpoint_directory)
    print(model.predict(test_mod1, test_mod2))
    print(model.score(test_mod1, test_mod2, z_test))
""" To reproduce CMAE on other samples, please refer to command lines belows:
GEX-ADT:
python cmae.py --subtask openproblems_bmmc_cite_phase2_rna --device cuda

GEX-ATAC:
python cmae.py --subtask openproblems_bmmc_multiome_phase2_rna --device cuda

"""

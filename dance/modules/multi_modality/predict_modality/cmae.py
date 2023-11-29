"""Reimplementation of Cross-Model AutoEncoder method.

Extended from https://github.com/uhlerlab/cross-modal-autoencoders

Reference
---------
Yang, Karren Dai, et al. "Multi-domain translation between single-cell imaging and sequencing data using autoencoders." Nature communications 12.1 (2021): 1-10.

"""
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from dance.utils import SimpleIndexDataset


class Discriminator(nn.Module):

    def __init__(self, input_dim, params):
        super().__init__()
        self.gan_type = params['gan_type']
        self.dim = params['dim']
        self.norm = params['norm']
        self.input_dim = input_dim
        self.net = self._make_net()

    def _make_net(self):
        return nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.input_dim, self.dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.dim, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x)

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0 = [self.forward(input_fake)]
        outs1 = [self.forward(input_real)]
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = [self.forward(input_fake)]
        loss = 0
        for it, (out0) in enumerate(outs0):
            # 1 = real data
            loss += torch.mean((out0 - 1)**2)
        return loss

    def calc_gen_loss_reverse(self, input_real):
        # calculate the loss to train G
        outs0 = [self.forward(input_real)]
        loss = 0
        for it, (out0) in enumerate(outs0):
            # 0 = fake data
            loss += torch.mean((out0 - 0)**2)
        return loss

    def calc_gen_loss_half(self, input_fake):
        # calculate the loss to train G
        outs0 = [self.forward(input_fake)]
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0.5)**2)
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss


##################################################################################
# Generator
##################################################################################


class VAEGen(nn.Module):
    # VAE architecture
    def __init__(self, input_dim, params, shared_layer=False):
        super().__init__()
        self.dim = params['dim']
        self.latent = params['latent']
        self.input_dim = input_dim

        # encoder_layers = [nn.Linear(self.input_dim, self.input_dim),
        #                   nn.LeakyReLU(0.2, inplace=True),
        #                   nn.Linear(self.input_dim, self.input_dim),
        #                   nn.LeakyReLU(0.2, inplace=True),
        #                   nn.Linear(self.input_dim, self.input_dim),
        #                   nn.LeakyReLU(0.2, inplace=True),
        #                   nn.Linear(self.input_dim, self.dim),
        #                   nn.LeakyReLU(0.2, inplace=True)]
        #
        # decoder_layers = [nn.LeakyReLU(0.2, inplace=True),
        #                   nn.Linear(self.dim, self.input_dim),
        #                   nn.LeakyReLU(0.2, inplace=True),
        #                   nn.Linear(self.input_dim, self.input_dim),
        #                   nn.LeakyReLU(0.2, inplace=True),
        #                   nn.Linear(self.input_dim, self.input_dim),
        #                   nn.LeakyReLU(0.2, inplace=True),
        #                   nn.Linear(self.input_dim, self.input_dim),
        #                   nn.LeakyReLU(0.2, inplace=True)]

        if self.input_dim > 1000:
            hid_size = 1000
        else:
            hid_size = self.input_dim

        encoder_layers = [
            nn.Linear(self.input_dim, hid_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hid_size, hid_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hid_size, hid_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hid_size, self.dim),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        decoder_layers = [
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.dim, hid_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hid_size, hid_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hid_size, hid_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hid_size, self.input_dim),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        if shared_layer:
            encoder_layers += [shared_layer["enc"], nn.LeakyReLU(0.2, inplace=True)]
            decoder_layers = [shared_layer["dec"]] + decoder_layers
        else:
            encoder_layers += [nn.Linear(self.dim, self.latent), nn.LeakyReLU(0.2, inplace=True)]
            decoder_layers = [nn.Linear(self.latent, self.dim)] + decoder_layers
        self.enc = nn.Sequential(*encoder_layers)
        self.dec = nn.Sequential(*decoder_layers)

    def forward(self, images):
        # This is a reduced VAE implementation where we assume the outputs are multivariate Gaussian distribution with mean = hiddens and std_dev = all ones.
        hiddens = self.encode(images)
        if self.training == True:
            noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
            images_recon = self.decode(hiddens + noise)
        else:
            images_recon = self.decode(hiddens)
        return images_recon, hiddens

    def encode(self, images):
        hiddens = self.enc(images)
        noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
        return hiddens, noise

    def decode(self, hiddens):
        images = self.dec(hiddens)
        return images


##################################################################################
# Classifier
##################################################################################


class Classifier(nn.Module):

    def __init__(self, input_dim, cls=3):
        super().__init__()
        self.input_dim = input_dim
        self.classes = cls
        self.net = self._make_net()

        self.cel = nn.CrossEntropyLoss()

    def _make_net(self):
        return nn.Sequential(nn.Linear(self.input_dim, self.classes))

    def forward(self, x):
        return self.net(x)

    def class_loss(self, input, target):
        return self.cel(input, target)


def weights_init(init_type='gaussian'):

    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [
        os.path.join(dirname, f) for f in os.listdir(dirname)
        if os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f
    ]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None  # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler


class CMAE(nn.Module):
    """CMAE class.

    Parameters
    ----------
    hyperparameters : dictionary
        A dictionary that contains arguments of CMAE. For details of parameters in parser args, please refer to link (parser help document).

    """

    def __init__(self, hyperparameters):
        super().__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        shared_layer = False
        if "shared_layer" in hyperparameters and hyperparameters["shared_layer"]:
            shared_layer = {}
            shared_layer["dec"] = nn.Linear(hyperparameters['gen']['latent'], hyperparameters['gen']['dim'])
            shared_layer["enc"] = nn.Linear(hyperparameters['gen']['dim'], hyperparameters['gen']['latent'])

        self.gen_a = VAEGen(hyperparameters['input_dim_a'], hyperparameters['gen'],
                            shared_layer)  # auto-encoder for domain a
        self.gen_b = VAEGen(hyperparameters['input_dim_b'], hyperparameters['gen'],
                            shared_layer)  # auto-encoder for domain b
        self.dis_latent = Discriminator(hyperparameters['gen']['latent'],
                                        hyperparameters['dis'])  # discriminator for latent space

        self.classifier = Classifier(hyperparameters['gen']['latent'],
                                     cls=hyperparameters['num_of_classes'])  # classifier on the latent space

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_latent.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters()) + list(self.classifier.parameters())
        self.dis_opt = torch.optim.AdamW([p for p in dis_params if p.requires_grad], lr=lr, betas=(beta1, beta2),
                                         weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.AdamW([p for p in gen_params if p.requires_grad], lr=lr, betas=(beta1, beta2),
                                         weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_latent.apply(weights_init('gaussian'))
        self.hyperparameters = hyperparameters

    def _mae_loss(self, input, target):
        """A simple criterion function for MAE loss.

        Parameters
        ----------
        inputs : torch.Tensor
            A input tensor.
        target : torch.Tensor
            A target tensor.

        Returns
        -------
        loss : float
            MAE loss between input and target tensors.

        """
        return torch.mean(torch.abs(input - target))

    def predict(self, mod1):
        """Predict function to get prediction of target modality features.

        Parameters
        ----------
        mod1 : torch.Tensor
            Input modality features.

        Returns
        -------
        pred : torch.Tensor
            Predicted features of target modality.

        """
        with torch.no_grad():
            emb, _ = self.gen_a.encode(mod1)
            pred = self.gen_b.decode(emb)
        return pred

    def score(self, mod1, mod2):
        """Score function to get score of prediction.

        Parameters
        ----------
        mod1 : torch.Tensor
            Input modality features.
        mod2 : torch.Tensor
            Output modality features.

        Returns
        -------
        score : float
            RMSE loss of predicted output modality features.

        """
        with torch.no_grad():
            pred = self.predict(mod1)
            mse = nn.MSELoss()
            score = math.sqrt(mse(pred, mod2))
            return score

    def forward(self, mod1, mod2):
        """Forward function for torch.nn.Module.

        Parameters
        ----------
        mod1 : torch.Tensor
            Input modality features.
        mod2 : torch.Tensor
            Target modality features.

        Returns
        -------
        x_ab : torch.Tensor
            Prediction of target modality from input modality.
        x_ba : torch.Tensor
            Prediction of input modality from target modality.

        """
        self.eval()
        h_a, _ = self.gen_a.encode(mod1)
        h_b, _ = self.gen_b.encode(mod2)
        x_ba = self.gen_a.decode(h_b)
        x_ab = self.gen_b.decode(h_a)
        self.train()
        return x_ab, x_ba

    def _gen_update(self, x_a, x_b, super_a, super_b, hyperparameters, a_labels=None, b_labels=None, variational=True):
        true_samples = Variable(torch.randn(200, hyperparameters['gen']['latent']), requires_grad=False).cuda()

        self.gen_opt.zero_grad()
        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # decode (within domain)
        if variational:
            h_a = h_a + n_a
            h_b = h_b + n_b

        x_a_recon = self.gen_a.decode(h_a)
        x_b_recon = self.gen_b.decode(h_b)

        classes_a = self.classifier.forward(h_a)
        classes_b = self.classifier.forward(h_b)

        # reconstruction loss
        self.loss_gen_recon_x_a = self._mae_loss(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self._mae_loss(x_b_recon, x_b)

        # GAN loss
        self.loss_latent_a = self.dis_latent.calc_gen_loss(h_a)
        self.loss_latent_b = self.dis_latent.calc_gen_loss_reverse(h_b)

        # Classification Loss
        if a_labels is not None and b_labels is not None:
            self.loss_class_a = self.classifier.class_loss(classes_a, a_labels)
            self.loss_class_b = self.classifier.class_loss(classes_b, b_labels)
        else:
            self.loss_class_a = self.loss_class_b = 0

        # supervision
        s_a, n_a = self.gen_a.encode(super_a)
        s_b, n_b = self.gen_b.encode(super_b)

        self.loss_supervision = self._mae_loss(s_a, s_b)

        class_weight = hyperparameters['gan_w'] if "class_w" not in hyperparameters else hyperparameters["class_w"]

        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_latent_a + \
                              hyperparameters['gan_w'] * self.loss_latent_b + \
                              class_weight * self.loss_class_a + \
                              class_weight * self.loss_class_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['super_w'] * self.loss_supervision

        if variational:
            self.loss_gen_total += hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_a + \
                                   hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_b

        self.loss_gen_total.backward()
        self.gen_opt.step()

    def _sample(self, x_a, x_b):
        self.eval()
        x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []
        for i in range(x_a.size(0)):
            h_a, _ = self.gen_a.encode(x_a[i].unsqueeze(0))
            h_b, _ = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(h_a))
            x_b_recon.append(self.gen_b.decode(h_b))
            x_ba.append(self.gen_a.decode(h_b))
            x_ab.append(self.gen_b.decode(h_a))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba = torch.cat(x_ba)
        x_ab = torch.cat(x_ab)
        self.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba

    def _dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()
        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # D loss
        self.loss_dis_latent = self.dis_latent.calc_dis_loss(h_a, h_b)
        self.loss_dis_total = hyperparameters['gan_w'] * (self.loss_dis_latent)
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def _update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir):
        """Resume function to resume from checkpoint file.

        Parameters
        ----------
        checkpoint_dir : str
            Path to the checkpoint file.

        Returns
        -------
        iterations : int
            Current iteration number of resumed model.

        """

        hyperparameters = self.hyperparameters
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_latent.load_state_dict(state_dict['latent'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, checkpoint_dir, iterations):
        """Save function to save parameters to checkpoint file.

        Parameters
        ----------
        checkpoint_dir : str
            Path to the checkpoint file.
        iterations : int
            Current number of training iterations.

        Returns
        -------
        None.

        """
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(checkpoint_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(checkpoint_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(checkpoint_dir, 'optimizer.pt')
        torch.save(
            {
                'a': self.gen_a.state_dict(),
                'b': self.gen_b.state_dict(),
                "classifier": self.classifier.state_dict()
            }, gen_name)
        torch.save({'latent': self.dis_latent.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)

    def fit(self, train_mod1, train_mod2, aux_labels=None, checkpoint_directory='./checkpoint', val_ratio=0.15):
        """Train CMAE.

        Parameters
        ----------
        train_mod1 : torch.Tensor
            Features of input modality.
        train_mod2 : torch.Tensor
            Features of target modality.
        aux_labels : torch.Tensor optional
            Auxiliary labels for extra supervision during training.
        checkpoint_directory : str optional
            Path to the checkpoint file, by default to be './checkpoint'.
        val_ratio : float
            Ratio for automatic train-validation split.

        Returns
        -------
        None.

        """

        hyperparameters = self.hyperparameters
        idx = torch.randperm(train_mod1.shape[0])
        train_idx = idx[:int(idx.shape[0] * (1 - val_ratio))]
        val_idx = idx[int(idx.shape[0] * (1 - val_ratio)):]

        train_dataset = SimpleIndexDataset(train_idx)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=hyperparameters['batch_size'],
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        # Start training
        iterations = self.resume(checkpoint_directory,
                                 hyperparameters=hyperparameters) if hyperparameters['resume'] else 0
        num_disc = 1 if "num_disc" not in hyperparameters else hyperparameters["num_disc"]
        num_gen = 1 if "num_gen" not in hyperparameters else hyperparameters["num_gen"]

        while True:
            print('Iteration: ', iterations)
            for it, batch_idx in enumerate(train_loader):
                mod1, mod2 = train_mod1[batch_idx], train_mod2[batch_idx]

                for _ in range(num_disc):
                    self._dis_update(mod1, mod2, hyperparameters)
                for _ in range(num_gen):
                    if aux_labels is not None:
                        self._gen_update(mod1, mod2, mod1, mod2, hyperparameters, aux_labels[batch_idx],
                                         aux_labels[batch_idx], variational=False)
                    else:
                        self._gen_update(mod1, mod2, mod1, mod2, hyperparameters, variational=False)
                self._update_learning_rate()

            print('RMSE Loss:', self.score(train_mod1[val_idx], train_mod2[val_idx]))

            iterations += 1
            if iterations >= hyperparameters['max_epochs']:
                print('Finish training')
                break

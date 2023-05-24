import math
from typing import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def kld_loss(p, q):

    def kld(target, pred):
        return torch.mean(torch.sum(target * torch.log(target / (pred + 1e-6)), dim=1))

    loss = kld(p, q)
    return loss


###############################################
##    Created by BABEL
###############################################
class BCELoss(nn.BCELoss):
    """Custom BCE loss that can correctly ignore the encoded latent space output."""

    def forward(self, x, target):
        input = x[0]
        return F.binary_cross_entropy(input, target, weight=self.weight, reduction=self.reduction)


class MSELoss(nn.MSELoss):
    """Custom MSE loss that can correctly ignore the encoded latent space output."""

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x[0], target, reduction=self.reduction)


class RMSELoss(nn.MSELoss):
    """Custom RMSE loss that can correctly ignore the encoded latent space output."""

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(F.mse_loss(x[0], target, reduction=self.reduction))


class DistanceProbLoss(nn.Module):
    """Analog of above log prob loss, but using distances May be useful for aligning
    latent spaces."""

    def __init__(self, weight: float = 5.0, norm: int = 1):
        super().__init__()
        assert weight > 0
        self.weight = weight
        self.norm = norm

    def forward(self, x, target_z):
        z, logp = x[:2]
        d = F.pairwise_distance(
            z,
            target_z,
            p=self.norm,
            eps=1e-6,
            keepdim=False,  # Default value
        )
        if len(d.shape) == 2:
            d = torch.mean(d, dim=1)  # Drop 1 dimension
        per_ex = self.weight * d - logp
        retval = torch.mean(per_ex)
        if retval != retval:
            raise ValueError("NaN")
        return retval
        return torch.mean(d)


class NegativeBinomialLoss(nn.Module):
    """Negative binomial loss.

    Preds should be a tuple of (mean, dispersion)

    """

    def __init__(
        self,
        scale_factor: float = 1.0,
        eps: float = 1e-10,
        l1_lambda: float = 0.0,
        mean: bool = True,
    ):
        super().__init__()
        self.loss = negative_binom_loss(
            scale_factor=scale_factor,
            eps=eps,
            mean=mean,
            debug=True,
        )
        self.l1_lambda = l1_lambda

    def forward(self, preds, target):
        preds, theta = preds[:2]
        l = self.loss(
            preds=preds,
            theta=theta,
            truth=target,
        )
        encoded = preds[:-1]
        l += self.l1_lambda * torch.abs(encoded).sum()
        return l


class ZeroInflatedNegativeBinomialLoss(nn.Module):
    """ZINB loss.

    Preds should be a tuple of (mean, dispersion, dropout) General
    notes: total variation seems to do poorly (at least for atacseq)

    """

    def __init__(
        self,
        ridge_lambda: float = 0.0,
        tv_lambda: float = 0.0,
        l1_lambda: float = 0.0,
        eps: float = 1e-10,
        scale_factor: float = 1.0,
        debug: bool = True,
    ):
        super().__init__()
        self.loss = zero_inflated_negative_binom_loss(
            ridge_lambda=ridge_lambda,
            tv_lambda=tv_lambda,
            eps=eps,
            scale_factor=scale_factor,
            debug=debug,
        )
        self.l1_lambda = l1_lambda

    def forward(self, preds, target):
        preds, theta, pi = preds[:3]
        l = self.loss(
            preds=preds,
            theta_disp=theta,
            pi_dropout=pi,
            truth=target,
        )
        encoded = preds[:-1]
        l += self.l1_lambda * torch.abs(encoded).sum()
        return l


class PairedLoss(nn.Module):
    """Paired loss function.

    Automatically unpacks and encourages the encoded representation to
    be similar using a given distance function. link_strength parameter
    controls how strongly we encourage this loss2_weight controls how
    strongly we weight the second loss, relative to the first A value of
    1.0 indicates that they receive equal weight, and a value larger
    indicates that the second loss receives greater weight. link_func
    should be a callable that takes in the two encoded representations
    and outputs a metric where a larger value indicates greater
    divergence

    """

    def __init__(
            self,
            loss1=NegativeBinomialLoss,
            loss2=ZeroInflatedNegativeBinomialLoss,
            link_func=lambda x, y: (x - y).abs().mean(),
            link_strength=1e-3,
    ):
        super().__init__()
        self.loss1 = loss1()
        self.loss2 = loss2()
        self.link = link_strength
        self.link_f = link_func

        self.warmup = SigmoidWarmup(
            midpoint=1000,
            maximum=link_strength,
        )

    def forward(self, preds, target):
        """Unpack and feed to each loss, averaging at end."""
        preds1, preds2 = preds
        target1, target2 = target

        loss1 = self.loss1(preds1, target1)
        loss2 = self.loss2(preds2, target2)
        retval = loss1 + loss2

        # Align the encoded representation assuming the last output is encoded representation
        encoded1 = preds1[-1]
        encoded2 = preds2[-1]
        if self.link > 0:
            l = next(self.warmup)
            if l > 1e-6:
                d = self.link_f(encoded1, encoded2).mean()
                retval += l * d

        return retval


class PairedLossInvertible(nn.Module):
    """Paired loss function with additional invertible (RealNVP) layer loss Loss 1 is
    for the first autoencoder Loss 2 is for the second autoencoder Loss 3 is for the
    invertible network at bottleneck."""

    def __init__(
            self,
            loss1=NegativeBinomialLoss,
            loss2=ZeroInflatedNegativeBinomialLoss,
            loss3=DistanceProbLoss,
            link_func=lambda x, y: (x - y).abs().mean(),
            link_strength=1e-3,
            inv_strength=1.0,
    ):
        super().__init__()
        self.loss1 = loss1()
        self.loss2 = loss2()
        self.loss3 = loss3()
        self.link = link_strength
        self.link_f = link_func

        # self.link_warmup = layers.SigmoidWarmup(
        #     midpoint=1000,
        #     maximum=link_strength,
        # )
        self.link_warmup = DelayedLinearWarmup(
            delay=1000,
            inc=5e-3,
            t_max=link_strength,
        )

        self.inv_warmup = DelayedLinearWarmup(
            delay=2000,
            inc=5e-3,
            t_max=inv_strength,
        )

    def forward(self, preds, target):
        """Unpack and feed to each loss."""
        # Both enc1_pred and enc2_pred are tuples of 2 values
        preds1, preds2, (enc1_pred, enc2_pred) = preds
        target1, target2 = target

        loss1 = self.loss1(preds1, target1)
        loss2 = self.loss2(preds2, target2)
        retval = loss1 + loss2

        # Align the encoded representations
        encoded1 = preds1[-1]
        encoded2 = preds2[-1]
        if self.link > 0:
            l = next(self.link_warmup)
            if l > 1e-6:
                d = self.link_f(encoded1, encoded2).mean()
                retval += l * d

        # Add a term for invertible network
        inv_loss1 = self.loss3(enc1_pred, enc2_pred[0])
        inv_loss2 = self.loss3(enc2_pred, enc1_pred[0])
        retval += next(self.inv_warmup) * (inv_loss1 + inv_loss2)

        return retval


class QuadLoss(PairedLoss):
    """Paired loss, but for the spliced autoencoder with 4 outputs."""

    def __init__(
        self,
        loss1=NegativeBinomialLoss,
        loss2=BCELoss,
        loss2_weight: float = 3.0,
        cross_weight: float = 1.0,
        cross_warmup_delay: int = 0,
        link_strength: float = 0.0,
        link_func: Callable = lambda x, y: (x - y).abs().mean(),
        link_warmup_delay: int = 0,
        record_history: bool = False,
    ):
        super().__init__()
        self.loss1 = loss1()
        self.loss2 = loss2()
        self.loss2_weight = loss2_weight
        self.history = []  # Eventually contains list of tuples per call
        self.record_history = record_history

        if link_warmup_delay:
            self.warmup = SigmoidWarmup(
                midpoint=link_warmup_delay,
                maximum=link_strength,
            )
            # self.warmup = layers.DelayedLinearWarmup(
            #     delay=warmup_delay,
            #     t_max=link_strength,
            #     inc=1e-3,
            # )
        else:
            self.warmup = NullWarmup(t_max=link_strength)
        if cross_warmup_delay:
            self.cross_warmup = SigmoidWarmup(
                midpoint=cross_warmup_delay,
                maximum=cross_weight,
            )
        else:
            self.cross_warmup = NullWarmup(t_max=cross_weight)

        self.link_strength = link_strength
        self.link_func = link_func

    def get_component_losses(self, preds, target):
        """Return the four losses that go into the overall loss, without scaling."""
        preds11, preds12, preds21, preds22 = preds
        if not isinstance(target, (list, tuple)):
            # Try to unpack into the correct parts
            target = torch.split(target, [preds11[0].shape[-1], preds22[0].shape[-1]], dim=-1)
        target1, target2 = target  # Both are torch tensors

        loss11 = self.loss1(preds11, target1)
        loss21 = self.loss1(preds21, target1)
        loss12 = self.loss2(preds12, target2)
        loss22 = self.loss2(preds22, target2)

        return loss11, loss21, loss12, loss22

    def forward(self, preds, target):
        loss11, loss21, loss12, loss22 = self.get_component_losses(preds, target)
        if self.record_history:
            detensor = lambda x: x.detach().cpu().numpy().item()
            self.history.append([detensor(l) for l in (loss11, loss21, loss12, loss22)])

        loss = loss11 + self.loss2_weight * loss22
        loss += next(self.cross_warmup) * (loss21 + self.loss2_weight * loss12)

        if self.link_strength > 0:
            l = next(self.warmup)
            if l > 1e-6:  # If too small we disregard
                preds11, preds12, preds21, preds22 = preds
                encoded1 = preds11[-1]  # Could be preds12
                encoded2 = preds22[-1]  # Could be preds21
                d = self.link_func(encoded1, encoded2)
                loss += self.link_strength * d
        return loss


def scvi_log_nb_positive(x, mu, theta, eps=1e-8):
    """
    Taken from scVI log_likelihood.py - scVI invocation is:
    reconst_loss = -log_nb_positive(x, px_rate, px_r).sum(dim=-1)
    scVI decoder outputs px_scale, px_r, px_rate, px_dropout
    px_scale is subject to Softmax
    px_r is just a Linear layer
    px_rate = torch.exp(library) * px_scale
    mu = mean of NB
    theta = indverse dispersion parameter
    Here, x appears to correspond to y_true in the below negative_binom_loss (aka the observed counts)
    """
    # if theta.ndimension() == 1:
    #     theta = theta.view(
    #         1, theta.size(0)
    #     )  # In this case, we reshape theta for broadcasting

    log_theta_mu_eps = torch.log(theta + mu + eps)
    res = (
        theta * (torch.log(theta + eps) - log_theta_mu_eps) + x * (torch.log(mu + eps) - log_theta_mu_eps) +
        torch.lgamma(x + theta) - torch.lgamma(theta)  # Present (in negative) for DCA
        - torch.lgamma(x + 1))

    return res.mean()


def scvi_log_zinb_positive(x, mu, theta, pi, eps=1e-8):
    """https://github.com/YosefLab/scVI/blob/6c9f43e3332e728831b174c1c1f0c9127b
    77cba0/scvi/models/log_likelihood.py#L206."""
    # theta is the dispersion rate. If .ndimension() == 1, it is shared for all cells (regardless of batch or labels)
    if theta.ndimension() == 1:
        theta = theta.view(1, theta.size(0))  # In this case, we reshape theta for broadcasting

    softplus_pi = F.softplus(-pi)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    case_non_zero = (
        -softplus_pi + pi_theta_log + x * (torch.log(mu + eps) - log_theta_mu_eps)  # Found above
        + torch.lgamma(x + theta)  # Found above
        - torch.lgamma(theta)  # Found above
        - torch.lgamma(x + 1)  # Found above
    )
    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero

    return res.mean()


def negative_binom_loss(
    scale_factor: float = 1.0,
    eps: float = 1e-10,
    mean: bool = True,
    debug: bool = False,
) -> Callable:
    """Return a function that calculates the binomial loss
    https://github.com/theislab/dca/blob/master/dca/loss.py combination of the Poisson
    distribution and a gamma distribution is a negative binomial distribution."""

    def loss(preds, theta, truth):
        """Calculates negative binomial loss as defined in the NB class in link
        above."""
        y_true = truth
        y_pred = preds * scale_factor

        if debug:  # Sanity check before loss calculation
            assert not torch.isnan(y_pred).any(), y_pred
            assert not torch.isinf(y_pred).any(), y_pred
            assert not (y_pred < 0).any()  # should be non-negative
            assert not (theta < 0).any()

        # Clip theta values
        theta = torch.clamp(theta, max=1e6)

        t1 = (torch.lgamma(theta + eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + theta + eps))
        t2 = (theta + y_true) * torch.log1p(y_pred /
                                            (theta + eps)) + (y_true *
                                                              (torch.log(theta + eps) - torch.log(y_pred + eps)))
        if debug:  # Sanity check after calculating loss
            assert not torch.isnan(t1).any(), t1
            assert not torch.isinf(t1).any(), (t1, torch.sum(torch.isinf(t1)))
            assert not torch.isnan(t2).any(), t2
            assert not torch.isinf(t2).any(), t2

        retval = t1 + t2
        if debug:
            assert not torch.isnan(retval).any(), retval
            assert not torch.isinf(retval).any(), retval

        return torch.mean(retval) if mean else retval

    return loss


def zero_inflated_negative_binom_loss(
    ridge_lambda: float = 0.0,
    tv_lambda: float = 0.0,
    eps: float = 1e-10,
    scale_factor: float = 1.0,
    debug: bool = False,
):
    """Return a function that calculates ZINB loss
    https://github.com/theislab/dca/blob/master/dca/loss.py."""
    nb_loss_func = negative_binom_loss(mean=False, eps=eps, scale_factor=scale_factor, debug=debug)

    def loss(preds, theta_disp, pi_dropout, truth):
        if debug:
            assert not (pi_dropout > 1.0).any()
            assert not (pi_dropout < 0.0).any()
        nb_case = nb_loss_func(preds, theta_disp, truth) - torch.log(1.0 - pi_dropout + eps)

        y_true = truth
        y_pred = preds * scale_factor
        theta = torch.clamp(theta_disp, max=1e6)

        zero_nb = torch.pow(theta / (theta + y_pred + eps), theta)
        zero_case = -torch.log(pi_dropout + ((1.0 - pi_dropout) * zero_nb) + eps)
        result = torch.where(y_true < 1e-8, zero_case, nb_case)

        # Ridge regularization on pi dropout term
        ridge = ridge_lambda * torch.pow(pi_dropout, 2)
        result += ridge

        # Total variation regularization on pi dropout term
        tv = tv_lambda * total_variation(pi_dropout)
        result += tv

        retval = torch.mean(result)
        # if debug:
        #     assert retval.item() > 0
        return retval

    return loss


def total_variation(x):
    """Given a 2D input (where one dimension is a batch dimension, the actual values are
    one dimensional) compute the total variation (within a 1 position shift)"""
    t = torch.sum(torch.abs(x[:, :-1] - x[:, 1:]))
    return t


class Warmup:  # Doesn't have to be nn.Module because it's not learned
    """Warmup layer similar to.

    Sonderby 2016 - Linear deterministic warm-up

    """

    def __init__(self, inc: float = 5e-3, t_max: float = 1.0):
        self.t = 0.0
        self.t_max = t_max
        self.inc = inc
        self.counter = 0  # Track number of times called next

    def __iter__(self):
        return self

    def __next__(self):
        retval = self.t
        t_next = self.t + self.inc
        self.t = min(t_next, self.t_max)
        self.counter += 1
        return retval


class DelayedLinearWarmup:
    """"""

    def __init__(self, delay: int = 2000, inc: float = 5e-3, t_max: float = 1.0):
        self.t = 0.0
        self.t_max = t_max
        self.inc = inc
        self.delay = delay
        self.counter = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.counter += 1
        retval = self.t
        if self.counter < self.delay:
            return retval
        self.t = min(self.t + self.inc, self.t_max)
        return retval


class SigmoidWarmup:
    """Sigmoid warmup Midpoints defines the number of iterations before we hit
    0.5 Scale determines how quickly we hit 1 after this point."""

    def __init__(self, midpoint: int = 500, scale: float = 0.1, maximum: float = 1.0):
        self.midpoint = midpoint
        self.scale = scale
        self.maximum = maximum
        self.counter = 0
        self.t = 0.0

    def __iter__(self):
        return self

    def __next__(self):
        retval = self.t
        t_next = 1.0 / (1.0 + np.exp(-self.scale * (self.counter - self.midpoint)))
        self.t = t_next
        self.counter += 1
        return self.maximum * retval


class NullWarmup(Warmup):
    """
    No warmup - but provides a consistent API
    """

    def __init__(self, delay: int = 0, t_max: float = 1.0):
        self.val = t_max

    def __next__(self):
        return self.val


####################################
## Created by scMVAE
####################################


def GMM_loss(gamma, c_params, z_params):
    """
    L elbo(x) = Eq(z,c|x)[ log p(x|z) ] - KL(q(z,c|x)||p(z,c))
              = Eq(z,c|x)[ log p(x|z) + log p(z|c) + log p(c) - log q(z|x) - log q(c|x) ]
    """
    mu_c, var_c, pi = c_params
    # print(mu_c.size(), var_c.size(), pi.size())
    n_centroids = pi.size(1)
    mu, logvar = z_params
    mu_expand = mu.unsqueeze(2).expand(mu.size(0), mu.size(1), n_centroids)
    logvar_expand = logvar.unsqueeze(2).expand(logvar.size(0), logvar.size(1), n_centroids)

    # log p(z|c)
    logpzc = -0.5 * torch.sum(gamma * torch.sum(math.log(2 * math.pi) + \
                                                torch.log(var_c) + \
                                                torch.exp(logvar_expand) / var_c + \
                                                (mu_expand - mu_c) ** 2 / var_c, dim=1), dim=1)
    # log p(c)
    logpc = torch.sum(gamma * torch.log(pi), 1)

    # log q(z|x) or q entropy
    qentropy = -0.5 * torch.sum(1 + logvar + math.log(2 * math.pi), 1)

    # log q(c|x)
    logqcx = torch.sum(gamma * torch.log(gamma), 1)

    kld = -logpzc - logpc + qentropy + logqcx

    return kld


####################################
## Created by DCCA
####################################


class Eucli_dis(nn.Module):
    """Like what you like: knowledge distill via neuron selectivity transfer."""

    def __init__(self):
        super().__init__()
        pass

    def forward(self, g_s, g_t):
        g_s = g_s.float()
        g_t = g_t.float()
        ret = torch.pow((g_s - g_t), 2)

        return torch.sum(ret, dim=1)


class L1_dis(nn.Module):
    """Like what you like: knowledge distill via neuron selectivity transfer."""

    def __init__(self):
        super().__init__()
        pass

    def forward(self, g_s, g_t):
        g_s = g_s.float()
        g_t = g_t.float()

        ret = torch.abs(g_s - g_t)

        return torch.sum(ret, dim=1)


class NSTLoss(nn.Module):
    """Like what you like: knowledge distill via neuron selectivity transfer."""

    def __init__(self):
        super().__init__()
        pass

    def forward(self, g_s, g_t):
        return [self.nst_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]

    def nst_loss(self, f_s, f_t):

        f_s = f_s.view(f_s.shape[0], f_s.shape[1], -1)
        f_s = F.normalize(f_s, dim=2)
        f_t = f_t.view(f_t.shape[0], f_t.shape[1], -1)
        f_t = F.normalize(f_t, dim=2)

        # set full_loss as False to avoid unnecessary computation
        full_loss = False
        if full_loss:
            return (self.poly_kernel(f_t, f_t).mean().detach() + self.poly_kernel(f_s, f_s).mean() -
                    2 * self.poly_kernel(f_s, f_t).mean())
        else:
            return self.poly_kernel(f_s, f_s).mean() - 2 * self.poly_kernel(f_s, f_t).mean()

    def poly_kernel(self, a, b):
        a = a.unsqueeze(1)
        b = b.unsqueeze(2)
        res = (a * b).sum(-1).pow(2)
        return res


class FactorTransfer(nn.Module):
    """Paraphrasing Complex Network: Network Compression via Factor Transfer, NeurIPS
    2018."""

    def __init__(self, p1=2, p2=1):
        super().__init__()
        self.p1 = p1
        self.p2 = p2

    def forward(self, f_s, f_t):
        return self.factor_loss(f_s, f_t)

    def factor_loss(self, f_s, f_t):
        s_H, t_H = f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
        else:
            pass
        if self.p2 == 1:
            return (self.factor(f_s) - self.factor(f_t)).abs().mean()
        else:
            return (self.factor(f_s) - self.factor(f_t)).pow(self.p2).mean()

    def factor(self, f):
        return F.normalize(f.pow(self.p1).mean(1).view(f.size(0), -1))


class Similarity(nn.Module):
    """Similarity-Preserving Knowledge Distillation, ICCV2019, verified by original
    author."""

    def __init__(self):
        super().__init__()

    def forward(self, g_s, g_t):
        return [self.similarity_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]

    def similarity_loss(self, f_s, f_t):
        bsz = f_s.shape[0]
        f_s = f_s.view(bsz, -1)
        f_t = f_t.view(bsz, -1)

        G_s = torch.mm(f_s, torch.t(f_s))
        # G_s = G_s / G_s.norm(2)
        G_s = torch.nn.functional.normalize(G_s)
        G_t = torch.mm(f_t, torch.t(f_t))
        # G_t = G_t / G_t.norm(2)
        G_t = torch.nn.functional.normalize(G_t)

        G_diff = G_t - G_s
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        return loss


class Correlation(nn.Module):
    """Correlation Congruence for Knowledge Distillation, ICCV 2019.

    The authors nicely shared the code with me. I restructured their code to be
    compatible with my running framework. Credits go to the original author

    """

    def __init__(self):
        super().__init__()

    def forward(self, f_s, f_t):
        delta = torch.abs(f_s - f_t)
        loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
        return loss


class KL_diver(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, mean_1, logvar_1, mean_2, logvar_2):
        loss = kl(Normal(mean_1, logvar_1), Normal(mean_2, logvar_2)).sum(dim=1)

        return loss


class Attention(nn.Module):
    """Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks
    via Attention Transfer
    code: https://github.com/szagoruyko/attention-transfer"""

    def __init__(self, p=2):
        super().__init__()
        self.p = p

    def forward(self, g_s, g_t):
        g_s_norm = F.normalize(g_s, p=2, dim=1)
        g_t_norm = F.normalize(g_t, p=2, dim=1)
        diff_g = g_s_norm - g_t_norm

        result = (diff_g.norm(p=2, dim=1, keepdim=True)).sum(dim=1)

        return result


class ZINBLoss(nn.Module):
    """ZINB loss class."""

    def __init__(self):
        super().__init__()

    def forward(self, x, mean, disp, pi, scale_factor, ridge_lambda=0.0):
        """Forward propagation.

        Parameters
        ----------
        x :
            input features.
        mean :
            data mean.
        disp :
            data dispersion.
        pi :
            data dropout probability.
        scale_factor : list
            scale factor of mean.
        ridge_lambda : float optional
            ridge parameter.

        Returns
        -------
        result : float
            ZINB loss.

        """
        eps = 1e-10
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor

        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0 - pi + eps)
        zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

        if ridge_lambda > 0:
            ridge = ridge_lambda * torch.square(pi)
            result += ridge

        result = torch.mean(result)
        return result


def dist_loss(data, min_dist, max_dist=20):
    pairwise_dist = cdisttf(data, data)
    dist = pairwise_dist - min_dist
    bigdist = max_dist - pairwise_dist
    loss = torch.exp(-dist) + torch.exp(-bigdist)
    return loss


def cdisttf(data_1, data_2):
    prod = torch.sum((data_1.unsqueeze(1) - data_2.unsqueeze(0))**2, 2)
    return (prod + 1e-10)**(1 / 2)

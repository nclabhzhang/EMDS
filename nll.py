### Original code from Equivariant Diffusion for Molecule Generation in 3D (EDM)
### https://github.com/ehoogeboom/e3_diffusion_for_molecules

import torch
from utils.graph_utils import node_flags, gen_noise, mask_x, mask_pos, mask_adjs
from losses import get_score_fn_3d
import numpy as np
import math



def log_prob(batch_n_nodes, dataset, remove_h):
    histogram = {22: 3393, 17: 13025, 23: 4848, 21: 9970, 19: 13832, 20: 9482, 16: 10644, 13: 3060,
                15: 7796, 25: 1506, 18: 13364, 12: 1689, 11: 807, 24: 539, 14: 5136, 26: 48, 7: 16, 10: 362,
                8: 49, 9: 124, 27: 266, 4: 4, 29: 25, 6: 9, 5: 5, 3: 1}  # atoms_num of QM9

    n_nodes = []
    prob = []
    keys = {}
    for i, nodes in enumerate(histogram):
        n_nodes.append(nodes)
        keys[nodes] = i
        prob.append(histogram[nodes])
    prob = np.array(prob)
    prob = prob / np.sum(prob)
    prob = torch.from_numpy(prob).float()

    idcs = [keys[i.item()] for i in batch_n_nodes]
    idcs = torch.tensor(idcs).to(batch_n_nodes.device)

    log_p = torch.log(prob + 1e-30)
    log_p = log_p.to(batch_n_nodes.device)
    log_probs = log_p[idcs]

    return log_probs


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(-1)


def compute_error(net_out, eps):
    eps_t = net_out
    error = sum_except_batch((eps - eps_t) ** 2)
    return error


def SNR(std_t):
    return 1 / (std_t ** 2)


def log_constants_p_x_given_z0(pos, sde_pos, node_mask):  # pos: [B, 29, 3]  node_mask: [B, 29, 1]
    batch_size = pos.size(0)

    n_nodes = node_mask.squeeze(2).sum(1)  # N has shape [B]
    assert n_nodes.size() == (batch_size,)
    degrees_of_freedom_x = (n_nodes - 1) * pos.size(2)

    t_zeros = torch.zeros(batch_size, device=pos.device)
    _, std_pos0 = sde_pos.marginal_prob(pos, t_zeros)
    log_sigma_x = torch.log(std_pos0 * (sde_pos.sigma_min / sde_pos.sigma_max))

    return degrees_of_freedom_x * (- log_sigma_x - 0.5 * np.log(2 * np.pi))


def cdf_standard_gaussian(x):
    return 0.5 * (1. + torch.erf(x / math.sqrt(2)))


def gaussian_KL(q_mu, q_sigma, p_mu, p_sigma, node_mask):
    return sum_except_batch(
            (
                torch.log(p_sigma / q_sigma)
                + 0.5 * (q_sigma**2 + (q_mu - p_mu)**2) / (p_sigma**2)
                - 0.5
            ) * node_mask
        )


def gaussian_KL_for_dimension(q_mu, q_sigma, p_mu, p_sigma, d):
    mu_norm2 = sum_except_batch((q_mu - p_mu)**2)
    assert len(q_sigma.size()) == 1
    assert len(p_sigma.size()) == 1
    return d * torch.log(p_sigma / q_sigma) + 0.5 * (d * q_sigma**2 + mu_norm2) / (p_sigma**2) - 0.5 * d


def kl_prior(pos, x, sde_pos, sde_x, node_mask):
    # Compute the last alpha value, alpha_T.
    t_ones = torch.ones(pos.size(0), device=pos.device)
    mean_x, std_x = sde_x.marginal_prob(x, t_ones)
    mean_pos, std_pos = sde_pos.marginal_prob(pos, t_ones)

    zeros_x, ones_x = mean_x, torch.ones_like(std_x[:, None, None]) * sde_x.sigma_max
    KL_distance_x = gaussian_KL(mean_x, std_x[:, None, None], zeros_x, ones_x, node_mask)

    zeros_pos, ones_pos = mean_pos, torch.ones_like(std_pos) * sde_pos.sigma_max
    number_of_nodes = torch.sum(node_mask.squeeze(2), dim=1)
    subspace_d = (number_of_nodes - 1) * pos.size(2)
    KL_distance_pos = gaussian_KL_for_dimension(mean_pos, std_pos, zeros_pos, ones_pos, d=subspace_d)

    return KL_distance_x + KL_distance_pos


def log_pxh_given_z0_without_constants(x, pos, std_x_0, z_pos_0, x_out_0, pos_out_0, node_mask, epsilon=1e-10):
    log_p_pos_given_z_without_constants = -0.5 * compute_error(pos_out_0, z_pos_0)

    x_centered = x_out_0 - 1
    std_x = std_x_0

    log_px_proportional = torch.log(
        cdf_standard_gaussian((x_centered + 0.5) / std_x[:, None, None])
        - cdf_standard_gaussian((x_centered - 0.5) / std_x[:, None, None])
        + epsilon)

    log_Z = torch.logsumexp(log_px_proportional, dim=2, keepdim=True)
    log_probabilities = log_px_proportional - log_Z

    log_px = sum_except_batch(log_probabilities * x * node_mask)

    log_p_x_pos_given_z = log_p_pos_given_z_without_constants + log_px

    return log_p_x_pos_given_z


def compute_NLL(x, pos, adj, model_x, model_pos, model_adj, sde_x, sde_pos, sde_adj, eps=1e-5):

    score_fn_x = get_score_fn_3d(sde_x, model_x, train=False, continuous=True)
    score_fn_pos = get_score_fn_3d(sde_pos, model_pos, train=False, continuous=True)

    t_int = torch.randint(1, sde_x.N + 1, size=(x.size(0),), device=x.device).float()
    s_int = t_int - 1

    s = s_int / sde_x.N
    t = t_int / sde_x.N

    flags = node_flags(adj)

    z_pos = gen_noise(pos, flags, sym=False)
    mean_pos, std_pos = sde_pos.marginal_prob(pos, t)
    _, std_pos_s = sde_pos.marginal_prob(pos, s)
    perturbed_pos = mean_pos + std_pos[:, None, None] * z_pos
    perturbed_pos = mask_pos(perturbed_pos, flags)

    z_x = gen_noise(x, flags, sym=False)
    mean_x, std_x = sde_x.marginal_prob(x, t)
    _, std_x_s = sde_x.marginal_prob(x, s)
    perturbed_x = mean_x + std_x[:, None, None] * z_x
    perturbed_x = mask_x(perturbed_x, flags)

    z_adj = gen_noise(adj, flags, sym=True)
    mean_adj, std_adj = sde_adj.marginal_prob(adj, t)
    perturbed_adj = mean_adj + std_adj[:, None, None] * z_adj
    perturbed_adj = mask_adjs(perturbed_adj, flags)

    score_pos = score_fn_pos(perturbed_x, perturbed_pos, perturbed_adj, flags, t)
    score_x = score_fn_x(perturbed_x, perturbed_pos, perturbed_adj, flags, t)
    pos_out = - (score_pos * std_pos[:, None, None])
    x_out = - (score_x * std_x[:, None, None])
    error_pos = compute_error(pos_out, z_pos)
    error_x = compute_error(x_out, z_x)

    SNR_weight_pos = 0.5 * (SNR(std_pos_s) / SNR(std_pos) - 1)
    SNR_weight_x = 0.5 * (SNR(std_x_s) / SNR(std_x) - 1)

    loss_t_larger_than_zero_pos = SNR_weight_pos * error_pos
    loss_t_larger_than_zero_x = SNR_weight_x * error_x

    neg_log_constants = -log_constants_p_x_given_z0(pos, sde_pos, flags[:, :, None])

    KL_prior = kl_prior(pos, x, sde_pos, sde_x, flags[:, :, None])

    loss_t_pos = sde_x.N * loss_t_larger_than_zero_pos
    loss_t_x = sde_x.N * loss_t_larger_than_zero_x

    t_zeros = torch.zeros_like(t)
    z_pos_0 = gen_noise(pos, flags, sym=False)
    mean_pos_0, std_pos_0 = sde_pos.marginal_prob(pos, t_zeros)
    pos_0 = mean_pos_0 + std_pos_0[:, None, None] * z_pos_0
    pos_0 = mask_pos(pos_0, flags)

    z_x_0 = gen_noise(x, flags, sym=False)
    mean_x_0, std_x_0 = sde_x.marginal_prob(x, t_zeros)
    x_0 = mean_x_0 + std_x_0[:, None, None] * z_x_0
    x_0 = mask_x(x_0, flags)

    z_adj_0 = gen_noise(adj, flags, sym=True)
    mean_adj_0, std_adj_0 = sde_adj.marginal_prob(adj, t_zeros)
    adj_0 = mean_adj_0 + std_adj_0[:, None, None] * z_adj_0
    adj_0 = mask_adjs(adj_0, flags)

    score_pos_0 = score_fn_pos(x_0, pos_0, adj_0, flags, t_zeros)
    score_x_0 = score_fn_x(x_0, pos_0, adj_0, flags, t_zeros)
    pos_out_0 = - (score_pos_0 * std_pos_0[:, None, None])
    x_out_0 = - (score_x_0 * std_x_0[:, None, None])

    loss_term_0 = -log_pxh_given_z0_without_constants(x, pos, std_x_0, z_pos_0, x_0, pos_out_0, flags[:, :, None])
    loss = KL_prior + loss_t_x + loss_t_pos + neg_log_constants + loss_term_0

    N = flags.sum(1).long()
    log_pN = log_prob(N, 'QM93D', False)

    nll = loss - log_pN
    nll = nll.mean(0)

    return nll







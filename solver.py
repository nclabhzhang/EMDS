import torch
import numpy as np
import abc
from tqdm import trange

from losses import get_score_fn_3d, get_sde_loss_fn_3d
from utils.graph_utils import mask_adjs, mask_x, mask_pos, gen_noise
from sde import VPSDE, subVPSDE


class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__()
    self.sde = sde
    # Compute the reverse SDE/ODE
    self.rsde = sde.reverse(score_fn, probability_flow)
    self.score_fn = score_fn

  @abc.abstractmethod
  def update_fn(self, x, t, flags):
    pass

class EulerMaruyamaPredictor(Predictor):
  def __init__(self, obj, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)
    self.obj = obj

  def update_fn(self, x, pos, adj, flags, t):
    dt = -1. / self.rsde.N

    if self.obj == 'x':
      z = gen_noise(x, flags, sym=False)
      drift, diffusion = self.rsde.sde(x, pos, adj, flags, t, 'x')
      x_mean = x + drift * dt
      x = x_mean + diffusion[:, None, None] * np.sqrt(-dt) * z
      return x, x_mean

    if self.obj == 'pos':
      z = gen_noise(pos, flags, sym=False)
      drift, diffusion = self.rsde.sde(x, pos, adj, flags, t, 'pos')
      pos_mean = pos + drift * dt
      pos = pos_mean + diffusion[:, None, None] * np.sqrt(-dt) * z
      return pos, pos_mean

    elif self.obj == 'adj':
      z = gen_noise(adj, flags)
      drift, diffusion = self.rsde.sde(x, pos, adj, flags, t, 'adj')
      adj_mean = adj + drift * dt
      adj = adj_mean + diffusion[:, None, None] * np.sqrt(-dt) * z
      return adj, adj_mean

    else:
      raise NotImplementedError(f"obj {self.obj} not yet supported.")

class ReverseDiffusionPredictor(Predictor):
  def __init__(self, obj, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)
    self.obj = obj

  def update_fn(self, x, pos, adj, flags, t):

    if self.obj == 'x':
      f, G = self.rsde.discretize(x, pos, adj, flags, t, 'x')
      z = gen_noise(x, flags, sym=False)
      x_mean = x - f
      x = x_mean + G[:, None, None] * z
      return x, x_mean

    if self.obj == 'pos':
      f, G = self.rsde.discretize(x, pos, adj, flags, t, 'pos')
      z = gen_noise(pos, flags, sym=False)
      pos_mean = pos - f
      pos = pos_mean + G[:, None, None] * z
      return pos, pos_mean

    elif self.obj == 'adj':
      f, G = self.rsde.discretize(x, pos, adj, flags, t, 'adj')
      z = gen_noise(adj, flags)
      adj_mean = adj - f
      adj = adj_mean + G[:, None, None] * z
      return adj, adj_mean
    
    else:
      raise NotImplementedError(f"obj {self.obj} not yet supported.")

class Corrector(abc.ABC):
  """The abstract class for a corrector algorithm."""
  def __init__(self, sde, score_fn, snr, scale_eps, n_steps):
    super().__init__()
    self.sde = sde
    self.score_fn = score_fn
    self.snr = snr   # signal-to-noise ratio
    self.scale_eps = scale_eps
    self.n_steps = n_steps

  @abc.abstractmethod
  def update_fn(self, x, t, flags):
    pass

class NoneCorrector(Corrector):
  """An empty corrector that does nothing."""
  def __init__(self, obj, sde, score_fn, snr, scale_eps, n_steps):
    self.obj = obj
    pass

  def update_fn(self, x, pos, adj, flags, t):
    if self.obj == 'x':
      return x, x
    elif self.obj == 'pos':
      return pos, pos
    elif self.obj == 'adj':
      return adj, adj
    else:
      raise NotImplementedError(f"obj {self.obj} not yet supported.")


class LangevinCorrector(Corrector):
  def __init__(self, obj, sde, score_fn, snr, scale_eps, n_steps):
    super().__init__(sde, score_fn, snr, scale_eps, n_steps)
    self.obj = obj

  def update_fn(self, x, pos, adj, flags, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    seps = self.scale_eps

    if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)   # a=1, if VESDE

    if self.obj == 'x':
      for i in range(n_steps):
        grad = score_fn(x, pos, adj, flags, t)         # g -- grad
        noise = gen_noise(x, flags, sym=False)    # z -- noise
        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()     # g_2norm
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()  # z_2norm
        step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha         # upsilon = 2a(snr*z_norm/g_norm)^2
        x_mean = x + step_size[:, None, None] * grad    # mean = x{t-1} + upsilon * g_norm
        x = x_mean + torch.sqrt(2 * step_size)[:, None, None] * noise * seps  # x{t} = x{t-1} + upsilon * g_norm + sqrt(2*upsilon) * z * noise_scale
      return x, x_mean

    elif self.obj == 'pos':
      for i in range(n_steps):
        grad = score_fn(x, pos, adj, flags, t)         # g -- grad
        noise = gen_noise(pos, flags, sym=False)    # z -- noise
        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()     # g_2norm
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()  # z_2norm
        step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha         # upsilon = 2a(snr*z_norm/g_norm)^2
        pos_mean = pos + step_size[:, None, None] * grad    # mean = x{t-1} + upsilon * g_norm
        pos = pos_mean + torch.sqrt(2 * step_size)[:, None, None] * noise * seps  # pos{t} = pos{t-1} + upsilon * g_norm + sqrt(2*upsilon) * z * noise_scale
      return pos, pos_mean

    elif self.obj == 'adj':
      for i in range(n_steps):
        grad = score_fn(x, pos, adj, flags, t)
        noise = gen_noise(adj, flags)
        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        adj_mean = adj + step_size[:, None, None] * grad
        adj = adj_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * seps
      return adj, adj_mean

    else:
      raise NotImplementedError(f"obj {self.obj} not yet supported")


# -------- PC sampler --------
def get_pc_sampler(sde_x, sde_pos, sde_adj, shape_x, shape_pos, shape_adj, predictor='Euler', corrector='None',
                   snr=0.1, scale_eps=1.0, n_steps=1, 
                   probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda'):

  def pc_sampler(model_x, model_pos, model_adj, init_flags):

    score_fn_x = get_score_fn_3d(sde_x, model_x, train=False, continuous=continuous)
    score_fn_pos = get_score_fn_3d(sde_pos, model_pos, train=False, continuous=continuous)
    score_fn_adj = get_score_fn_3d(sde_adj, model_adj, train=False, continuous=continuous)

    predictor_fn = ReverseDiffusionPredictor if predictor=='Reverse' else EulerMaruyamaPredictor     # ReverseDiffusionPredictor for Molecule Generation
    corrector_fn = LangevinCorrector if corrector=='Langevin' else NoneCorrector     # Langevin for Molecule Generation

    predictor_obj_x = predictor_fn('x', sde_x, score_fn_x, probability_flow)
    corrector_obj_x = corrector_fn('x', sde_x, score_fn_x, snr, scale_eps, n_steps)

    predictor_obj_pos = predictor_fn('pos', sde_pos, score_fn_pos, probability_flow)
    corrector_obj_pos = corrector_fn('pos', sde_pos, score_fn_pos, snr, scale_eps, n_steps)

    predictor_obj_adj = predictor_fn('adj', sde_adj, score_fn_adj, probability_flow)
    corrector_obj_adj = corrector_fn('adj', sde_adj, score_fn_adj, snr, scale_eps, n_steps)

    with torch.no_grad():
      # -------- Initial sample --------
      x = sde_x.prior_sampling(shape_x).to(device)
      pos = sde_pos.prior_sampling(shape_pos).to(device)
      adj = sde_adj.prior_sampling_sym(shape_adj).to(device)

      flags = init_flags

      x = mask_x(x, flags)
      pos = mask_pos(pos, flags)
      adj = mask_adjs(adj, flags)

      diff_steps = sde_adj.N
      timesteps = torch.linspace(sde_adj.T, eps, diff_steps, device=device)

      # -------- Reverse diffusion process --------
      for i in trange(0, (diff_steps), desc='[Sampling]', position=1, leave=False):
        t = timesteps[i]
        vec_t = torch.ones(shape_adj[0], device=t.device) * t

        # _x, _pos = x, pos
        x, x_mean = corrector_obj_x.update_fn(x, pos, adj, flags, vec_t)
        adj, adj_mean = corrector_obj_adj.update_fn(x, pos, adj, flags, vec_t)
        pos, pos_mean = corrector_obj_pos.update_fn(x, pos, adj, flags, vec_t)

        # _x, _pos = x, pos
        x, x_mean = predictor_obj_x.update_fn(x, pos, adj, flags, vec_t)
        adj, adj_mean = predictor_obj_adj.update_fn(x, pos, adj, flags, vec_t)
        pos, pos_mean = predictor_obj_pos.update_fn(x, pos, adj, flags, vec_t)
      print(' ')
      return (x_mean if denoise else x), (pos_mean if denoise else pos), (adj_mean if denoise else adj), diff_steps * (n_steps + 1)

  return pc_sampler
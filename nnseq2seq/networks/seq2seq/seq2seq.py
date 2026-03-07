import torch
import torch.nn as nn
import torch.nn.functional as F

from nnseq2seq.networks.seq2seq.model2d.model_jit import JiT_models as JiT_2d_models
from nnseq2seq.networks.seq2seq.model3d.model_jit import JiT_models as JiT_3d_models


class Seq2Seq2d(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ndim = 2

        self.in_channels = args['in_channels']
        self.num_classes = args['num_classes']
        
        self.net = JiT_2d_models[args['model_name']](
            input_size=args['patch_size'],
            in_channels=self.in_channels+1,
            num_classes=self.in_channels+1,
        )

        self.P_mean = -0.8
        self.P_std = 0.8
        self.t_eps = 2e-5
        self.noise_scale = 1.0

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device)
        t_pos = torch.sigmoid(z * self.P_std + self.P_mean)
        t_neg = torch.clamp(z*0.25+0.25, 0, 1)
        t = torch.where(z >= 1, t_pos, t_neg)
        return t
    
    def forward(self, x, src_code, all_tgt_code):
        x = x * 2 - 1
        src_code_mask = src_code.view(x.size(0), x.size(1), *([1] * (x.ndim - 2)))
        all_tgt_code_mask = all_tgt_code.view(x.size(0), x.size(1), *([1] * (x.ndim - 2)))

        t = self.sample_t(x.size(0)*x.size(1), device=x.device).view(-1, x.size(1), *([1] * (x.ndim - 2)))
        t = t * all_tgt_code_mask
        t = t * (1 - src_code_mask) + (1 - self.t_eps) * src_code_mask
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        x_pred = self.net(z, t)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        # l2 loss
        loss = (v - v_pred) ** 2 * all_tgt_code_mask * (1 - src_code_mask) + \
            ((v - v_pred)*(1 - t).clamp_min(self.t_eps)) ** 2 * all_tgt_code_mask * src_code_mask
        loss = loss.mean(dim=(1, 2, 3)).mean()

        z = z.detach() * 0.5 + 0.5
        x_pred = x_pred.detach() * 0.5 + 0.5

        return z, x_pred, loss
    
    @torch.no_grad()
    def infer_atten_weights(self, x, t):
        x = x * 2 - 1
        
        e = torch.randn_like(x) * self.noise_scale
        t = t.view(-1, x.size(1), *([1] * (x.ndim - 2)))
        z = t * x + (1 - t) * e

        _, attn_weights = self.net(z, t, return_attn_weights=True)
        return attn_weights
    
    @torch.no_grad()
    def infer_diffusion(self, x, src_code, tgt_code=None, timesteps=None, num_inference_steps=50):
        x = x * 2 - 1
        
        t0 = src_code.view(x.size(0), x.size(1), *([1] * (x.ndim - 2)))
        torch.manual_seed(42)
        if x.is_cuda:
            torch.cuda.manual_seed_all(42)
        e = torch.randn_like(x) * self.noise_scale

        z = t0 * x + (1 - t0) * e

        if timesteps is None:
            timesteps = torch.linspace(0.0, 1.0, num_inference_steps+1, device=x.device).view(-1, *([1] * z.ndim)).expand(-1, -1, x.size(1), -1, -1)
            if tgt_code is not None:
                tgt_code = tgt_code.view(x.size(0), x.size(1), *([1] * (x.ndim - 2)))
                timesteps = timesteps * tgt_code.unsqueeze(0) * (1 - t0.unsqueeze(0)) + 1 * t0.unsqueeze(0)
            else:
                timesteps = timesteps * (1 - t0.unsqueeze(0)) + 1 * t0.unsqueeze(0)
            
        if num_inference_steps==1:
            x_pred = self.net(z, timesteps[0])
            x_pred = x_pred * 0.5 + 0.5
            return x_pred

        for i in range(num_inference_steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = self._heun_step(z, t, t_next)
            z = t0 * x + (1 - t0) * z
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1])
        z = t0 * x + (1 - t0) * z

        z = torch.clamp(z, -1, 1)
        z = z * 0.5 + 0.5
        return z
    
    @torch.no_grad()
    def _forward_sample(self, z, t):
        x = self.net(z, t)
        v = (x - z) / (1.0 - t).clamp_min(self.t_eps)
        return v
    
    @torch.no_grad()
    def _euler_step(self, z, t, t_next):
        v_pred = self._forward_sample(z, t)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next):
        v_pred_t = self._forward_sample(z, t)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next
    
    
class Seq2Seq3d(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ndim = 3

        self.in_channels = args['in_channels']
        self.num_classes = args['num_classes']
        
        self.net = JiT_3d_models[args['model_name']](
            input_size=args['patch_size'],
            in_channels=self.in_channels+1,
            num_classes=self.in_channels+1,
        )

        self.P_mean = -0.8
        self.P_std = 0.8
        self.t_eps = 2e-5
        self.noise_scale = 1.0
    
    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device)
        t_pos = torch.sigmoid(z * self.P_std + self.P_mean)
        t_neg = torch.clamp(z*0.25+0.25, 0, 1)
        t = torch.where(z >= 1, t_pos, t_neg)
        return t
    
    def forward(self, x, src_code, all_tgt_code):
        x = x * 2 - 1
        src_code_mask = src_code.view(x.size(0), x.size(1), *([1] * (x.ndim - 2)))
        all_tgt_code_mask = all_tgt_code.view(x.size(0), x.size(1), *([1] * (x.ndim - 2)))

        t = self.sample_t(x.size(0)*x.size(1), device=x.device).view(-1, x.size(1), *([1] * (x.ndim - 2)))
        t = t * all_tgt_code_mask
        t = t * (1 - src_code_mask) + (1 - self.t_eps) * src_code_mask
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        x_pred = self.net(z, t)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        # l2 loss
        loss = (v - v_pred) ** 2 * all_tgt_code_mask * (1 - src_code_mask) + \
            ((v - v_pred)*(1 - t).clamp_min(self.t_eps)) ** 2 * all_tgt_code_mask * src_code_mask
        loss = loss.mean(dim=(1, 2, 3, 4)).mean()

        z = z.detach() * 0.5 + 0.5
        x_pred = x_pred.detach() * 0.5 + 0.5

        return z, x_pred, loss
    
    @torch.no_grad()
    def infer_diffusion(self, x, src_code, tgt_code=None, timesteps=None, num_inference_steps=50):
        x = x * 2 - 1
        
        t0 = src_code.view(x.size(0), x.size(1), *([1] * (x.ndim - 2)))
        torch.manual_seed(42)
        if x.is_cuda:
            torch.cuda.manual_seed_all(42)
        e = torch.randn_like(x) * self.noise_scale

        z = t0 * x + (1 - t0) * e

        if timesteps is None:
            timesteps = torch.linspace(0.0, 1.0, num_inference_steps+1, device=x.device).view(-1, *([1] * z.ndim)).expand(-1, -1, x.size(1), -1, -1, -1)
            if tgt_code is not None:
                tgt_code = tgt_code.view(x.size(0), x.size(1), *([1] * (x.ndim - 2)))
                timesteps = timesteps * tgt_code.unsqueeze(0) * (1 - t0.unsqueeze(0)) + 1 * t0.unsqueeze(0)
            else:
                timesteps = timesteps * (1 - t0.unsqueeze(0)) + 1 * t0.unsqueeze(0)

        if num_inference_steps==1:
            x_pred = self.net(z, timesteps[0])
            x_pred = x_pred * 0.5 + 0.5
            return x_pred
        
        for i in range(num_inference_steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = self._heun_step(z, t, t_next)
            z = t0 * x + (1 - t0) * z
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1])
        z = t0 * x + (1 - t0) * z

        z = torch.clamp(z, -1, 1)
        z = z * 0.5 + 0.5
        return z
    
    @torch.no_grad()
    def _forward_sample(self, z, t):
        x = self.net(z, t)
        v = (x - z) / (1.0 - t).clamp_min(self.t_eps)
        return v
    
    @torch.no_grad()
    def _euler_step(self, z, t, t_next):
        v_pred = self._forward_sample(z, t)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next):
        v_pred_t = self._forward_sample(z, t)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next
# --------------------------------------------------------
# References:
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------

from math import pi
from typing import Optional, Tuple, Union

import torch
from torch import nn
import numpy as np

from einops import rearrange, repeat


def _to_hw(x: Union[int, Tuple[int, int]], name: str = "size") -> Tuple[int, int]:
    """
    Normalize size into (H, W).
    - if x is int: (x, x)
    - if x is tuple/list of length 2: (x[0], x[1])
    """
    if isinstance(x, int):
        return (x, x)
    if isinstance(x, (tuple, list)) and len(x) == 2:
        h, w = int(x[0]), int(x[1])
        return (h, w)
    raise ValueError(f"{name} must be int or tuple/list of length 2, got: {type(x)} {x}")


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')


class VisionRotaryEmbedding(nn.Module):
    """
    支持 pt_seq_len / ft_seq_len 传入:
    - int: 认为是正方形 (S, S)
    - tuple(H, W): 非正方形
    """
    def __init__(
        self,
        dim,
        pt_seq_len: Union[int, Tuple[int, int]],
        ft_seq_len: Optional[Union[int, Tuple[int, int]]] = None,
        custom_freqs=None,
        freqs_for='lang',
        theta=10000,
        max_freq=10,
        num_freqs=1,
    ):
        super().__init__()

        if custom_freqs is not None:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        pt_h, pt_w = _to_hw(pt_seq_len, "pt_seq_len")
        if ft_seq_len is None:
            ft_h, ft_w = pt_h, pt_w
        else:
            ft_h, ft_w = _to_hw(ft_seq_len, "ft_seq_len")

        # 关键修改：H/W 分开插值尺度
        t_h = torch.arange(ft_h) / ft_h * pt_h
        t_w = torch.arange(ft_w) / ft_w * pt_w

        freqs_h = torch.einsum('..., f -> ... f', t_h, freqs)
        freqs_h = repeat(freqs_h, '... n -> ... (n r)', r=2)  # [H, dim]

        freqs_w = torch.einsum('..., f -> ... f', t_w, freqs)
        freqs_w = repeat(freqs_w, '... n -> ... (n r)', r=2)  # [W, dim]

        # [H, 1, dim] and [1, W, dim] -> [H, W, 2*dim]
        freqs_2d = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim=-1)

        self.register_buffer("freqs_cos", freqs_2d.cos(), persistent=False)
        self.register_buffer("freqs_sin", freqs_2d.sin(), persistent=False)

    def forward(self, t, start_index=0):
        rot_dim = self.freqs_cos.shape[-1]
        end_index = start_index + rot_dim
        assert rot_dim <= t.shape[-1], (
            f'feature dimension {t.shape[-1]} is not of sufficient size to rotate '
            f'in all the positions {rot_dim}'
        )
        t_left, t_mid, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
        t_mid = (t_mid * self.freqs_cos) + (rotate_half(t_mid) * self.freqs_sin)
        return torch.cat((t_left, t_mid, t_right), dim=-1)


class VisionRotaryEmbeddingFast(nn.Module):
    """
    支持 pt_seq_len / ft_seq_len 传入:
    - int: 认为是正方形 (S, S)
    - tuple(H, W): 非正方形

    该版本通常用于 token 已经 flatten 成 [B, N, D] 的情况：
    - N_img = H*W
    - 如有 cls token，则 N = num_cls_token + H*W
    """
    def __init__(
        self,
        dim,
        pt_seq_len: Union[int, Tuple[int, int]] = 16,
        ft_seq_len: Optional[Union[int, Tuple[int, int]]] = None,
        custom_freqs=None,
        freqs_for='lang',
        theta=10000,
        max_freq=10,
        num_freqs=1,
        num_cls_token=0
    ):
        super().__init__()

        if custom_freqs is not None:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        pt_h, pt_w = _to_hw(pt_seq_len, "pt_seq_len")
        if ft_seq_len is None:
            ft_h, ft_w = pt_h, pt_w
        else:
            ft_h, ft_w = _to_hw(ft_seq_len, "ft_seq_len")

        # 关键修改：H/W 分开
        t_h = torch.arange(ft_h) / ft_h * pt_h
        t_w = torch.arange(ft_w) / ft_w * pt_w

        freqs_h = torch.einsum('..., f -> ... f', t_h, freqs)
        freqs_h = repeat(freqs_h, '... n -> ... (n r)', r=2)  # [H, dim]

        freqs_w = torch.einsum('..., f -> ... f', t_w, freqs)
        freqs_w = repeat(freqs_w, '... n -> ... (n r)', r=2)  # [W, dim]

        # [H, W, 2*dim]
        freqs_2d = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim=-1)

        # flatten to [H*W, 2*dim]
        freqs_flat = freqs_2d.view(-1, freqs_2d.shape[-1])
        cos_img = freqs_flat.cos()
        sin_img = freqs_flat.sin()

        if num_cls_token > 0:
            # prepend cls tokens: cos=1, sin=0
            N_img, D = cos_img.shape
            cos_pad = torch.ones(num_cls_token, D, dtype=cos_img.dtype, device=cos_img.device)
            sin_pad = torch.zeros(num_cls_token, D, dtype=sin_img.dtype, device=sin_img.device)
            cos_all = torch.cat([cos_pad, cos_img], dim=0)  # [N_cls+N_img, D]
            sin_all = torch.cat([sin_pad, sin_img], dim=0)
        else:
            cos_all = cos_img
            sin_all = sin_img

        # 用 buffer，避免写死 .cuda()，这样 module.to(device) 会自动搬运
        self.register_buffer("freqs_cos", cos_all, persistent=False)
        self.register_buffer("freqs_sin", sin_all, persistent=False)

    def forward(self, t):
        # t: [B, N, D_rot] where N matches freqs_cos.shape[0]
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


def get_2d_sincos_pos_embed(embed_dim, grid_size: Union[int, Tuple[int, int]], cls_token=False, extra_tokens=0):
    """
    grid_size:
      - int: (S, S)
      - tuple(H, W): 非正方形

    return:
      pos_embed: [H*W, embed_dim] or [extra_tokens+H*W, embed_dim]
    """
    H, W = _to_hw(grid_size, "grid_size")

    grid_h = np.arange(H, dtype=np.float32)
    grid_w = np.arange(W, dtype=np.float32)

    # meshgrid: X(w) first, Y(h) second -> shapes [H, W]
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)  # [2, H, W]
    grid = grid.reshape([2, 1, H, W])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim], dtype=pos_embed.dtype), pos_embed], axis=0)

    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # 注意：这里沿用原代码的顺序（grid[0]、grid[1]），只做非正方形支持，不改行为
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2)

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

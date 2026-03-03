# --------------------------------------------------------
# References:
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------

from math import pi

import torch
from torch import nn
import numpy as np

from einops import rearrange, repeat


# -----------------------------
# Helpers
# -----------------------------
def parse_3d_shape(x, name="shape"):
    """
    Normalize a 3D shape spec into (D, H, W).

    Supports:
      - int -> (x, x, x)
      - tuple/list/torch.Size of length 3 -> (d, h, w)
      - dict with keys in {'d','h','w'} -> (d, h, w)
    """
    if isinstance(x, int):
        return x, x, x

    if isinstance(x, (tuple, list, torch.Size)):
        assert len(x) == 3, f"{name} must be int or a 3-tuple/list/Size, got len={len(x)}"
        return int(x[0]), int(x[1]), int(x[2])

    if isinstance(x, dict):
        assert all(k in x for k in ("d", "h", "w")), f"{name} dict must have keys d,h,w"
        return int(x["d"]), int(x["h"]), int(x["w"])

    raise TypeError(f"Unsupported {name} type: {type(x)}")


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), "invalid dimensions for broadcastable concatentation"
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


# -----------------------------
# Rotary Embeddings (3D)
# -----------------------------
class VisionRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len,          # int or (D,H,W) or dict(d,h,w)
        ft_seq_len=None,     # int or (D,H,W) or dict(d,h,w)
        custom_freqs=None,
        freqs_for="lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
    ):
        super().__init__()

        pt_d, pt_h, pt_w = parse_3d_shape(pt_seq_len, "pt_seq_len")
        if ft_seq_len is None:
            ft_d, ft_h, ft_w = pt_d, pt_h, pt_w
        else:
            ft_d, ft_h, ft_w = parse_3d_shape(ft_seq_len, "ft_seq_len")

        if custom_freqs is not None:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f"unknown modality {freqs_for}")

        # map FT grid onto PT grid scale
        tz = torch.arange(ft_d) / ft_d * pt_d
        ty = torch.arange(ft_h) / ft_h * pt_h
        tx = torch.arange(ft_w) / ft_w * pt_w

        # each axis: [L, dim/2] -> repeat to [L, dim]
        fz = torch.einsum("l, f -> l f", tz, freqs)
        fz = repeat(fz, "l n -> l (n r)", r=2)

        fy = torch.einsum("l, f -> l f", ty, freqs)
        fy = repeat(fy, "l n -> l (n r)", r=2)

        fx = torch.einsum("l, f -> l f", tx, freqs)
        fx = repeat(fx, "l n -> l (n r)", r=2)

        # concat into [D,H,W, rot_dim], rot_dim = 3*dim
        freqs_3d = broadcat(
            (fz[:, None, None, :], fy[None, :, None, :], fx[None, None, :, :]),
            dim=-1,
        )

        self.register_buffer("freqs_cos", freqs_3d.cos())
        self.register_buffer("freqs_sin", freqs_3d.sin())

    def forward(self, t, start_index=0):
        rot_dim = self.freqs_cos.shape[-1]
        end_index = start_index + rot_dim
        assert rot_dim <= t.shape[-1], f"feature dim {t.shape[-1]} < rot_dim {rot_dim}"

        # t: [..., D, H, W, C]
        t_left = t[..., :start_index]
        t_mid = t[..., start_index:end_index]
        t_right = t[..., end_index:]

        t_mid = (t_mid * self.freqs_cos) + (rotate_half(t_mid) * self.freqs_sin)
        return torch.cat((t_left, t_mid, t_right), dim=-1)


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len=(16, 16, 16),   # int or (D,H,W) or dict(d,h,w)
        ft_seq_len=None,           # int or (D,H,W) or dict(d,h,w)
        custom_freqs=None,
        freqs_for="lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
        num_cls_token=0,
    ):
        super().__init__()

        pt_d, pt_h, pt_w = parse_3d_shape(pt_seq_len, "pt_seq_len")
        if ft_seq_len is None:
            ft_d, ft_h, ft_w = pt_d, pt_h, pt_w
        else:
            ft_d, ft_h, ft_w = parse_3d_shape(ft_seq_len, "ft_seq_len")

        if custom_freqs is not None:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f"unknown modality {freqs_for}")

        tz = torch.arange(ft_d) / ft_d * pt_d
        ty = torch.arange(ft_h) / ft_h * pt_h
        tx = torch.arange(ft_w) / ft_w * pt_w

        fz = torch.einsum("l, f -> l f", tz, freqs)
        fz = repeat(fz, "l n -> l (n r)", r=2)

        fy = torch.einsum("l, f -> l f", ty, freqs)
        fy = repeat(fy, "l n -> l (n r)", r=2)

        fx = torch.einsum("l, f -> l f", tx, freqs)
        fx = repeat(fx, "l n -> l (n r)", r=2)

        freqs_3d = broadcat(
            (fz[:, None, None, :], fy[None, :, None, :], fx[None, None, :, :]),
            dim=-1,
        )  # [D,H,W, rot_dim]

        freqs_flat = freqs_3d.reshape(-1, freqs_3d.shape[-1])  # [N, rot_dim]
        cos_img = freqs_flat.cos()
        sin_img = freqs_flat.sin()

        if num_cls_token > 0:
            N, Drot = cos_img.shape
            cos_pad = torch.ones(num_cls_token, Drot, dtype=cos_img.dtype, device=cos_img.device)
            sin_pad = torch.zeros(num_cls_token, Drot, dtype=sin_img.dtype, device=sin_img.device)
            cos_img = torch.cat([cos_pad, cos_img], dim=0)
            sin_img = torch.cat([sin_pad, sin_img], dim=0)

        self.register_buffer("freqs_cos", cos_img)
        self.register_buffer("freqs_sin", sin_img)

    def forward(self, t):
        # t: [B, N, rot_dim] (or [*, N, rot_dim])
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin


# -----------------------------
# RMSNorm
# -----------------------------
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


# -----------------------------
# 3D SinCos Positional Embeddings
# -----------------------------
def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int -> (D,H,W)=(grid,grid,grid)
               or (D,H,W) / dict(d,h,w)
    return:
      pos_embed: [D*H*W, embed_dim] or [extra_tokens + D*H*W, embed_dim] if cls_token
    """
    gd, gh, gw = parse_3d_shape(grid_size, "grid_size")

    # split equally across 3 axes => each axis uses embed_dim/3, and 1D sincos needs even dim
    assert embed_dim % 6 == 0, "For 3D sincos with equal split, require embed_dim % 6 == 0."

    grid_d = np.arange(gd, dtype=np.float32)
    grid_h = np.arange(gh, dtype=np.float32)
    grid_w = np.arange(gw, dtype=np.float32)

    grid = np.meshgrid(grid_d, grid_h, grid_w, indexing="ij")  # 3 arrays [D,H,W]
    grid = np.stack(grid, axis=0).reshape([3, 1, gd, gh, gw])

    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 6 == 0
    dim_each = embed_dim // 3
    assert dim_each % 2 == 0

    emb_d = get_1d_sincos_pos_embed_from_grid(dim_each, grid[0])  # (D*H*W, dim_each)
    emb_h = get_1d_sincos_pos_embed_from_grid(dim_each, grid[1])
    emb_w = get_1d_sincos_pos_embed_from_grid(dim_each, grid[2])

    emb = np.concatenate([emb_d, emb_h, emb_w], axis=1)  # (D*H*W, embed_dim)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position (must be even)
    pos: positions (any shape) -> flattened
    out: (M, embed_dim)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000 ** omega)  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb

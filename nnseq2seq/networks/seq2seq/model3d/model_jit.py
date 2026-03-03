# --------------------------------------------------------
# References:
# SiT: https://github.com/willisma/SiT
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnseq2seq.networks.seq2seq.model3d.model_util import (
    VisionRotaryEmbeddingFast,
    get_3d_sincos_pos_embed,
    RMSNorm,
)


# -----------------------------
# Helpers
# -----------------------------
def parse_3d_shape(x, name="shape"):
    """
    Normalize a 3D shape spec into (D, H, W).

    Supports:
      - int -> (x, x, x)
      - tuple/list/torch.Size of length 3 -> (d, h, w)
      - dict with keys d,h,w -> (d,h,w)
    """
    if isinstance(x, int):
        return int(x), int(x), int(x)
    if isinstance(x, (tuple, list, torch.Size)):
        assert len(x) == 3, f"{name} must be int or a 3-tuple/list/Size, got len={len(x)}"
        return int(x[0]), int(x[1]), int(x[2])
    if isinstance(x, dict):
        assert all(k in x for k in ("d", "h", "w")), f"{name} dict must have keys d,h,w"
        return int(x["d"]), int(x["h"]), int(x["w"])
    raise TypeError(f"Unsupported {name} type: {type(x)}")


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class BottleneckPatchEmbed(nn.Module):
    """3D Volume to Patch Embedding (supports non-cubic D/H/W)"""

    def __init__(
        self,
        img_size=224,          # int or (D,H,W)
        patch_size=16,         # int or (pD,pH,pW)
        in_chans=3,
        pca_dim=768,
        embed_dim=768,
        bias=True,
    ):
        super().__init__()

        img_size = parse_3d_shape(img_size, "img_size")          # (D,H,W)
        patch_size = parse_3d_shape(patch_size, "patch_size")    # (pD,pH,pW)

        self.img_size = img_size
        self.patch_size = patch_size
        self.vol_size = img_size

        D, H, W = img_size
        pD, pH, pW = patch_size

        assert D % pD == 0 and H % pH == 0 and W % pW == 0, \
            f"img_size={img_size} must be divisible by patch_size={patch_size}"

        gd = D // pD
        gh = H // pH
        gw = W // pW

        self.grid_size = (gd, gh, gw)
        self.num_patches = gd * gh * gw

        self.proj1 = nn.Conv3d(
            in_chans, pca_dim, kernel_size=(pD, pH, pW), stride=(pD, pH, pW), bias=False
        )
        self.proj2 = nn.Conv3d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        # x: [B,C,D,H,W]
        B, C, D, H, W = x.shape
        assert (D, H, W) == self.vol_size, \
            f"Input size {(D,H,W)} doesn't match model {self.vol_size}."
        x = self.proj2(self.proj1(x))          # [B,embed_dim,gd,gh,gw]
        x = x.flatten(2).transpose(1, 2)       # [B, N, embed_dim]
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations.
    Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


def scaled_dot_product_attention(query, key, value, dropout_p=0.0) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))

    # device-safe (avoid hard-coded .cuda())
    attn_bias = torch.zeros(query.size(0), 1, L, S, dtype=query.dtype, device=query.device)

    with torch.amp.autocast(device_type=query.device.type, enabled=False):
        attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=query.requires_grad and dropout_p > 0 and torch.is_grad_enabled())
    return attn_weight @ value


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = rope(q)
        k = rope(k)

        x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop=0.0, bias=True) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class FinalLayer(nn.Module):
    """
    The final layer of JiT.
    Supports patch_size as int or (pD,pH,pW).
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)

        pD, pH, pW = parse_3d_shape(patch_size, "patch_size")
        patch_vol = pD * pH * pW

        self.linear = nn.Linear(hidden_size, patch_vol * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class JiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, num_classes, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        # channel attention
        self.norm3 = RMSNorm(hidden_size, eps=1e-6)
        self.attn2 = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm4 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp2 = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation2 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(num_classes*hidden_size, 6 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x, c, feat_rope=None):
        B, C, N, L = x.shape
        x = x.reshape(-1,N,L)
        c = c.reshape(-1,L)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        x = x.reshape(B,C,N,L).permute(0,2,1,3).reshape(-1,C,L)
        c = c.reshape(B,1,-1).repeat(1,N,1).reshape(B*N,-1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation2(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn2(modulate(self.norm3(x), shift_msa, scale_msa), rope=nn.Identity())
        x = x + gate_mlp.unsqueeze(1) * self.mlp2(modulate(self.norm4(x), shift_mlp, scale_mlp))

        x = x.reshape(B,N,C,L).permute(0,2,1,3)
        return x


class JiT(nn.Module):
    """
    Just image Transformer (3D).
    Now supports input_size=(D,H,W) (non-cubic) and patch_size=(pD,pH,pW).
    """
    def __init__(
        self,
        input_size=256,         # int or (D,H,W)
        patch_size=16,          # int or (pD,pH,pW)
        in_channels=3,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        num_classes=1000,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=8
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = 1

        self.patch_size = parse_3d_shape(patch_size, "patch_size")   # store as (pD,pH,pW)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = parse_3d_shape(input_size, "input_size")   # store as (D,H,W)

        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.num_classes = num_classes

        # time and class embed
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        # linear embed (3D)
        self.x_embedder = BottleneckPatchEmbed(
            img_size=self.input_size,
            patch_size=self.patch_size,
            in_chans=1,
            pca_dim=bottleneck_dim,
            embed_dim=hidden_size,
            bias=True
        )

        # fixed sin-cos embedding
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.mod_embed = nn.Parameter(torch.zeros(1, in_channels, 1, hidden_size), requires_grad=True)
        nn.init.normal_(self.mod_embed, std=0.02)

        # in-context tokens
        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True)
            torch.nn.init.normal_(self.in_context_posemb, std=.02)

        # rope
        gd, gh, gw = self.x_embedder.grid_size
        head_dim = hidden_size // num_heads

        rope_rot_dim = (head_dim // 6) * 6
        if rope_rot_dim == 0:
            raise ValueError(f"head_dim={head_dim} 太小，无法做 3D RoPE（需要至少 6 的倍数）。")

        dim_for_rope = rope_rot_dim // 3  # because rot_dim = 3 * dim

        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=dim_for_rope,
            pt_seq_len=(gd, gh, gw),
            num_cls_token=0
        )
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=dim_for_rope,
            pt_seq_len=(gd, gh, gw),
            num_cls_token=self.in_context_len
        )

        # transformer
        self.blocks = nn.ModuleList([
            JiTBlock(
                hidden_size, num_heads, in_channels, mlp_ratio=mlp_ratio,
                attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0
            )
            for i in range(depth)
        ])

        # linear predict
        self.final_layer = FinalLayer(hidden_size, self.patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # 3D sin-cos pos embed
        grid_size = self.x_embedder.grid_size  # (gd,gh,gw)
        if self.pos_embed.shape[-1] % 6 != 0:
            raise ValueError(
                f"hidden_size={self.pos_embed.shape[-1]} 不能被 6 整除，无法使用均分式 3D sincos pos embed。"
            )
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # init patch_embed conv weights
        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        if self.x_embedder.proj2.bias is not None:
            nn.init.constant_(self.x_embedder.proj2.bias, 0)

        # label embedding
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: [B, N, (pD*pH*pW*out_channels)]
        return: [B, out_channels, D, H, W]
        """
        c = self.out_channels
        gd, gh, gw = self.x_embedder.grid_size
        pD, pH, pW = self.patch_size

        assert x.shape[1] == gd * gh * gw, f"N={x.shape[1]} != gd*gh*gw={gd*gh*gw}"
        assert x.shape[-1] == (pD * pH * pW * c), \
            f"last_dim={x.shape[-1]} != pD*pH*pW*c={(pD*pH*pW*c)}"

        B = x.shape[0]
        x = x.view(B, gd, gh, gw, pD, pH, pW, c)            # [B,gd,gh,gw,pD,pH,pW,C]
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()  # [B,C,gd,pD,gh,pH,gw,pW]
        x = x.view(B, c, gd * pD, gh * pH, gw * pW)         # [B,C,D,H,W]
        return x

    def forward(self, x, t):
        """
        x: [B, C, D, H, W]  (supports non-cubic D/H/W)
        t: [B, C]
        """
        y = torch.arange(x.size(1)).unsqueeze(0).expand(x.size(0), -1).to(x.device)

        # class and time embeddings
        t_emb = self.t_embedder(t.reshape(-1)).reshape(x.shape[0], x.shape[1], -1)
        y_emb = self.y_embedder(y.reshape(-1)).reshape(x.shape[0], x.shape[1], -1)
        c = t_emb + y_emb

        # forward JiT
        B, C, D, W, H = x.shape
        x = self.x_embedder(x.reshape(-1,1,D,W,H))   # [B,N,hidden]
        _, N, L = x.shape
        x = x.reshape(B,C,N,L) + self.pos_embed.unsqueeze(1) + self.mod_embed

        for i, block in enumerate(self.blocks):
            if self.in_context_len > 0 and i == self.in_context_start:
                in_context_tokens = y_emb.unsqueeze(2).repeat(1, 1, self.in_context_len, 1)
                in_context_tokens = in_context_tokens + self.in_context_posemb.unsqueeze(1)
                x = torch.cat([in_context_tokens, x], dim=2) # B,C,M+N,L

            x = block(
                x, c,
                self.feat_rope if i < self.in_context_start else self.feat_rope_incontext
            )

        x = x[:, :, self.in_context_len:]  # drop in-context tokens
        x = self.final_layer(x.reshape(B*C,N,L), c.reshape(B*C,-1))      # [B,N,pD*pH*pW*C]
        output = self.unpatchify(x).reshape(B,C,D,W,H)     # [B,C,D,H,W]
        return output


# -----------------------------
# Model factories (unchanged)
# -----------------------------
def JiT_B_16(**kwargs):
    return JiT(depth=12, hidden_size=792, num_heads=12,
               bottleneck_dim=256, in_context_len=32, in_context_start=4, patch_size=8, **kwargs)

def JiT_B_32(**kwargs):
    return JiT(depth=12, hidden_size=792, num_heads=12,
               bottleneck_dim=256, in_context_len=32, in_context_start=4, patch_size=16, **kwargs)

def JiT_L_16(**kwargs):
    return JiT(depth=24, hidden_size=1056, num_heads=16,
               bottleneck_dim=256, in_context_len=32, in_context_start=8, patch_size=8, **kwargs)

def JiT_L_32(**kwargs):
    return JiT(depth=24, hidden_size=1056, num_heads=16,
               bottleneck_dim=256, in_context_len=32, in_context_start=8, patch_size=16, **kwargs)

def JiT_H_16(**kwargs):
    return JiT(depth=32, hidden_size=1344, num_heads=16,
               bottleneck_dim=512, in_context_len=32, in_context_start=10, patch_size=8, **kwargs)

def JiT_H_32(**kwargs):
    return JiT(depth=32, hidden_size=1344, num_heads=16,
               bottleneck_dim=512, in_context_len=32, in_context_start=10, patch_size=16, **kwargs)


JiT_models = {
    'JiT-B/16': JiT_B_16,
    'JiT-B/32': JiT_B_32,
    'JiT-L/16': JiT_L_16,
    'JiT-L/32': JiT_L_32,
    'JiT-H/16': JiT_H_16,
    'JiT-H/32': JiT_H_32,
}

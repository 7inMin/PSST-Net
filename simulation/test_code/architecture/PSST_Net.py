import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import to_2tuple, trunc_normal_


def shift_back(inputs, step=2):
    bs, nC, row, col = inputs.shape
    down_sample = 256 // row
    step = float(step) / float(down_sample * down_sample)
    output = torch.zeros_like(inputs)
    for i in range(nC):
        shift = int(step * i)
        if shift < col:
            output[:, i, :, : col - shift] = inputs[:, i, :, shift:]
    return output


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def window_partition_channels_first(x, window_size):
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return windows


def window_reverse_channels_first(windows, window_size, H, W):
    B = int(windows.shape[0] / ((H // window_size) * (W // window_size)))
    C = windows.shape[1]
    x = windows.view(B, H // window_size, W // window_size, C, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)
    return x


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=2):
        super().__init__()
        hidden_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False, groups=hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.net(x)
        return x.permute(0, 2, 3, 1)


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.input_resolution = to_2tuple(input_resolution)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(window_size), num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, mult=mlp_ratio / 2.0)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, H_in, W_in, C = x.shape
        shortcut = x
        x = self.norm1(x)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class MGA(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(n_feat, n_feat, kernel_size=5, padding=2, bias=True, groups=n_feat)

    def forward(self, mask_shifted):
        mask_shifted = self.conv1(mask_shifted)
        mask_gate = torch.sigmoid(self.depth_conv(self.conv2(mask_shifted)))
        mask_shifted = mask_shifted * mask_gate + mask_shifted
        return shift_back(mask_shifted)


class LocalResidualRefiner(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.dw5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False)
        self.dw11 = nn.Conv2d(channels, channels, kernel_size=11, padding=5, groups=channels, bias=False)
        self.post = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.act(self.pre(x))
        x = self.act(self.dw5(x))
        x = self.dw11(x)
        x = self.post(x)
        return residual + x


class SSHM(nn.Module):
    def __init__(self, dim, stage=1, mode="full"):
        super().__init__()
        self.mode = str(mode).lower().strip()
        self.local_refiner = LocalResidualRefiner(dim)
        input_resolution = 256 // (2 ** stage)
        num_heads = 2 ** stage
        self.swa4_w = self.swa4_sw = self.swa8_w = self.swa8_sw = None
        if self.mode in {"full", "w_only"}:
            self.swa4_w = SwinTransformerBlock(dim, input_resolution, num_heads, window_size=4, shift_size=0)
            self.swa8_w = SwinTransformerBlock(dim, input_resolution, num_heads, window_size=8, shift_size=0)
        if self.mode in {"full", "sw_only"}:
            self.swa4_sw = SwinTransformerBlock(dim, input_resolution, num_heads, window_size=4, shift_size=2)
            self.swa8_sw = SwinTransformerBlock(dim, input_resolution, num_heads, window_size=8, shift_size=4)
        self.proj_local = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.proj_global = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.gate_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.global_fuse = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)
        self.pn = PreNorm(dim, FeedForward(dim=dim))
        self.norm = nn.LayerNorm(dim)

    def forward_mode(self, x, mode="full"):
        x_local = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_local = self.local_refiner(x_local)
        x_global_in = x.permute(0, 2, 3, 1)
        if mode == "full":
            x_g4 = self.swa4_sw(self.swa4_w(x_global_in))
            x_g8 = self.swa8_sw(self.swa8_w(x_global_in))
        elif mode == "w_only":
            x_g4 = self.swa4_w(x_global_in)
            x_g8 = self.swa8_w(x_global_in)
        else:
            x_g4 = self.swa4_sw(x_global_in)
            x_g8 = self.swa8_sw(x_global_in)
        x_g4 = x_g4.permute(0, 3, 1, 2)
        x_g8 = x_g8.permute(0, 3, 1, 2)
        x_global = self.global_fuse(torch.cat([x_g4, x_g8], dim=1))
        x_local = self.proj_local(x_local)
        x_global = self.proj_global(x_global)
        gate = self.gate_conv(torch.cat([x_local, x_global], dim=1))
        x = x + gate * x_global + (1.0 - gate) * x_local
        x = x.permute(0, 2, 3, 1)
        x = self.pn(x) + x
        return x.permute(0, 3, 1, 2)


class SpatialAttentionBlock(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, stage_level=0, sshm_mode="full"):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.mga = MGA(dim)
        self.sshm = SSHM(dim, stage=stage_level, mode=sshm_mode)
        self.dw11 = LocalResidualRefiner(dim)
        self.spatial_gate_conv = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim, bias=False)
        self.attention_type = "full"

    def forward(self, x_in, mask):
        b, h, w, c = x_in.shape
        x_spatial_input = x_in.permute(0, 3, 1, 2)
        x_spatial_restored = self.sshm.forward_mode(x_spatial_input, mode=self.sshm.mode)
        x_spatial_restored = self.dw11(x_spatial_restored)
        x_spatial_gate = self.spatial_gate_conv(mask.permute(0, 3, 1, 2)) + mask.permute(0, 3, 1, 2)
        if x_spatial_gate.shape[3] != x_spatial_restored.shape[3]:
            x_spatial_gate = shift_back(x_spatial_gate)
        x = (x_spatial_restored * x_spatial_gate).permute(0, 2, 3, 1).reshape(b, h * w, c)
        q_proj = self.to_q(x)
        k_proj = self.to_k(x)
        v_proj = self.to_v(x)
        mask_gate = self.mga(mask.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        if b != 0:
            mask_gate = mask_gate[0].expand([b, h, w, c])
        q, k, v, mask_gate = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads),
            (q_proj, k_proj, v_proj, mask_gate.flatten(1, 2)),
        )
        v = v * mask_gate
        q = F.normalize(q.transpose(-2, -1), dim=-1, p=2)
        k = F.normalize(k.transpose(-2, -1), dim=-1, p=2)
        v = v.transpose(-2, -1)
        spectral_attn = (k @ q.transpose(-2, -1)) * self.rescale
        spectral_attn = spectral_attn.softmax(dim=-1)
        x = spectral_attn @ v
        x = x.permute(0, 3, 1, 2).reshape(b, h * w, self.num_heads * self.dim_head)
        spectral_out = self.proj(x).view(b, h, w, c)
        pos_out = self.pos_emb(v_proj.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return spectral_out + pos_out


class SpatialOnlyBlock(nn.Module):
    def __init__(self, dim, dim_head, heads, num_blocks, stage_level, sshm_mode):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        SpatialAttentionBlock(dim=dim, dim_head=dim_head, heads=heads, stage_level=stage_level, sshm_mode=sshm_mode),
                        PreNorm(dim, FeedForward(dim=dim)),
                    ]
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x, mask):
        x = x.permute(0, 2, 3, 1)
        mask_hw = mask.permute(0, 2, 3, 1)
        for attn, ffn in self.blocks:
            x = attn(x, mask_hw) + x
            x = ffn(x) + x
        return x.permute(0, 3, 1, 2)


class SpectralTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = x + shortcut
        x = x + self.ffn(self.norm2(x))
        return x


class SpecTR(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=4, is_shifted=False):
        super().__init__()
        self.window_size = window_size
        self.is_shifted = is_shifted
        self.shift_size = window_size // 2
        self.mask_modulation_gate = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )
        self.spectral_transformer = SpectralTransformerBlock(dim=window_size * window_size, num_heads=num_heads)

    def forward(self, x, aligned_mask_feature):
        shortcut = x
        B, C, H, W = x.shape
        if aligned_mask_feature.shape[-2:] != (H, W):
            aligned_mask_feature = F.interpolate(aligned_mask_feature, size=(H, W), mode="bilinear", align_corners=False)
        x = x * self.mask_modulation_gate(aligned_mask_feature)
        if self.is_shifted:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        windows = window_partition_channels_first(x, self.window_size)
        B_nw, _, ws, _ = windows.shape
        spectral_tokens = windows.view(B_nw, C, ws * ws)
        windows_processed = self.spectral_transformer(spectral_tokens)
        windows_out = windows_processed.view(B_nw, C, ws, ws)
        x_out = window_reverse_channels_first(windows_out, self.window_size, H, W)
        if self.is_shifted:
            x_out = torch.roll(x_out, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        return x_out + shortcut


class SpatialOnlyUNet(nn.Module):
    def __init__(self, dim=28, stage=2, num_blocks=None, input_resolution=256):
        super().__init__()
        self.dim = int(dim)
        self.stage = int(stage)
        if num_blocks is None:
            num_blocks = [1] * (self.stage + 1)
        if len(num_blocks) != self.stage + 1:
            raise ValueError("num_blocks length must be stage+1")
        self.embedding = nn.Conv2d(28, self.dim, 3, 1, 1, bias=False)
        self.meas_mga = MGA(self.dim)
        self.meas_cond = nn.Sequential(
            nn.Conv2d(28, self.dim, 3, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(self.dim, self.dim, 1, 1, 0, bias=False),
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.encoder_layers = nn.ModuleList([])
        dim_stage = self.dim
        for i in range(self.stage):
            self.encoder_layers.append(
                nn.ModuleList(
                    [
                        SpatialOnlyBlock(dim_stage, self.dim, dim_stage // self.dim, num_blocks[i], i, "w_only"),
                        nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
                        nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
                    ]
                )
            )
            dim_stage *= 2
        self.bottleneck = SpatialOnlyBlock(dim_stage, self.dim, dim_stage // self.dim, num_blocks[-1], self.stage, "w_only")
        self.decoder_layers = nn.ModuleList([])
        for i in range(self.stage):
            out_dim = dim_stage // 2
            self.decoder_layers.append(
                nn.ModuleList(
                    [
                        nn.ConvTranspose2d(dim_stage, out_dim, kernel_size=2, stride=2, bias=False),
                        nn.Conv2d(dim_stage, out_dim, 1, 1, bias=False),
                        SpatialOnlyBlock(out_dim, self.dim, out_dim // self.dim, num_blocks[self.stage - 1 - i], self.stage - 1 - i, "sw_only"),
                    ]
                )
            )
            dim_stage = out_dim
        self.mapping = nn.Conv2d(self.dim, 28, 3, 1, 1, bias=False)

    def forward(self, x, mask, x_raw):
        feat = self.lrelu(self.embedding(x))
        if x_raw.shape[-2:] != feat.shape[-2:]:
            x_raw = F.interpolate(x_raw, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        if mask.shape[-2:] != feat.shape[-2:]:
            mask_in = F.interpolate(mask, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        else:
            mask_in = mask
        mask_mod = self.meas_mga(mask_in)
        feat = feat + self.meas_cond(x_raw) * (1.0 + mask_mod)
        skips = []
        mask_pyr = []
        mask_feat = mask_in
        for block, down_feat, down_mask in self.encoder_layers:
            feat = block(feat, mask_feat)
            skips.append(feat)
            mask_pyr.append(mask_feat)
            feat = down_feat(feat)
            mask_feat = down_mask(mask_feat)
        feat = self.bottleneck(feat, mask_feat)
        for idx, (up, fuse, block) in enumerate(self.decoder_layers):
            feat = up(feat)
            feat = fuse(torch.cat([feat, skips[self.stage - 1 - idx]], dim=1))
            feat = block(feat, mask_pyr[self.stage - 1 - idx])
        return self.mapping(feat) + x


class SpectralOnlyRefiner(nn.Module):
    def __init__(self, bands=28, window_size=8, small_window_size=4, num_heads=4, num_blocks=2):
        super().__init__()
        self.mask_aligner = MGA(bands)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                nn.ModuleList(
                    [
                        SpecTR(dim=bands, window_size=window_size, num_heads=num_heads, is_shifted=False),
                        SpecTR(dim=bands, window_size=window_size, num_heads=num_heads, is_shifted=True),
                        SpecTR(dim=bands, window_size=small_window_size, num_heads=num_heads, is_shifted=False),
                        SpecTR(dim=bands, window_size=small_window_size, num_heads=num_heads, is_shifted=True),
                        nn.Conv2d(bands * 2, bands, kernel_size=1, bias=False),
                    ]
                )
            )
        self.beta = nn.Parameter(torch.full((1, bands, 1, 1), 0.1, dtype=torch.float32))

    def forward(self, x, mask):
        if mask.shape[-2:] != x.shape[-2:]:
            mask = F.interpolate(mask, size=x.shape[-2:], mode="bilinear", align_corners=False)
        mask_aligned = self.mask_aligner(mask)
        out = x
        for w8, sw8, w4, sw4, fuse in self.blocks:
            x8 = sw8(w8(out, mask_aligned), mask_aligned)
            x4 = sw4(w4(out, mask_aligned), mask_aligned)
            delta = fuse(torch.cat([x4, x8], dim=1))
            out = out + self.beta * delta
        return out


class SpatialSpectralGroup(nn.Module):
    def __init__(self, dim=28, stage=2, num_blocks=None, input_resolution=256, spectral_blocks=2):
        super().__init__()
        self.spatial = SpatialOnlyUNet(dim=dim, stage=stage, num_blocks=num_blocks, input_resolution=input_resolution)
        self.spectral = SpectralOnlyRefiner(bands=28, num_blocks=spectral_blocks)

    def forward(self, x, mask, x_raw):
        return self.spectral(self.spatial(x, mask, x_raw), mask)


class SSLT(nn.Module):
    def __init__(self, dim=28, stage=2, num_blocks=None, attention_type="full", input_resolution=256, spectral_blocks=2):
        super().__init__()
        if num_blocks is None:
            num_blocks = [1] * (stage + 1)
        self.group1 = SpatialSpectralGroup(dim=dim, stage=stage, num_blocks=[2,2,2], input_resolution=input_resolution, spectral_blocks=spectral_blocks)
        self.group2 = SpatialSpectralGroup(dim=dim, stage=stage, num_blocks=[2,2,2], input_resolution=input_resolution, spectral_blocks=spectral_blocks)
        self.group3 = SpatialSpectralGroup(dim=dim, stage=2, num_blocks=[3,2,2], input_resolution=input_resolution, spectral_blocks=spectral_blocks)

    def forward(self, x, mask=None, return_stages=False):
        if mask is None:
            raise ValueError("SSLT_93 requires an explicit mask input.")
        out1 = self.group1(x, mask, x)
        out2 = self.group2(out1, mask, x)
        out3 = self.group3(out2, mask, x)
        if return_stages:
            return out1, out2, out3
        return out3


if __name__ == "__main__":
    model = SSLT(dim=28, stage=2, num_blocks=[2, 2, 2], input_resolution=256).cuda()
    y = torch.randn([1, 28, 256, 256]).float().cuda()
    mask = torch.randn([1, 28, 256, 256]).float().cuda()
    out1, out2, out3 = model(y, mask=mask, return_stages=True)
    print(out1.shape, out2.shape, out3.shape)

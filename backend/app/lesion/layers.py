# coding=utf-8
import numpy as np
import torch
from torch import nn
import math
from torch.nn.functional import interpolate
from torch.nn.modules.batchnorm import _BatchNorm
from functools import partial
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
# from bn_lib.nn.modules import SynchronizedBatchNorm2d



class SwitchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=True,
                 last_gamma=False):
        super(SwitchNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


# ================================================================================================
# Swin Transformer Bottleneck Module (STBM) for CAUNet
# 插入位置: Encoder E5输出后, Decoder输入前
# 作用: 在40x40语义层级进行全局上下文推理,抑制血管/噪声等假阳性
# ================================================================================================

class SwinBottleneckModule(nn.Module):
    """
    Swin Transformer Bottleneck Module (STBM) with Input-Adaptive Gating
    - 输入自适应门控: 根据全局特征动态决定Swin修正强度
    - 残差缩放: 初始为0,渐进学习
    """

    def __init__(self, in_ch=256, embed_dim=128, num_heads=4, window_size=8,
                 depth=3, mlp_ratio=2.0, drop_path=0.1, bn_momentum=0.2):
        super(SwinBottleneckModule, self).__init__()

        self.in_ch = in_ch
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.depth = depth

        # 输入投影
        self.proj_in = nn.Sequential(
            nn.Conv2d(in_ch, embed_dim, kernel_size=1, bias=False),
            SwitchNorm2d(embed_dim, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )

        # Swin Blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path * i / max(depth - 1, 1),
                bn_momentum=bn_momentum
            )
            for i in range(depth)
        ])

        # 输出投影
        self.proj_out = nn.Sequential(
            nn.Conv2d(embed_dim, in_ch, kernel_size=1, bias=False),
            SwitchNorm2d(in_ch, momentum=bn_momentum),
        )

        # [方案二新增] 输入自适应门控: 根据输入特征动态决定修正强度
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, in_ch // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 残差缩放系数
        self.residual_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.window_size == 0 and W % self.window_size == 0, \
            f"STBM expects H,W divisible by {self.window_size}, got {H}x{W}"

        # 输入投影
        x_proj = self.proj_in(x)

        # 转序列格式
        x_seq = x_proj.permute(0, 2, 3, 1)

        # Swin Blocks
        for blk in self.blocks:
            x_seq = blk(x_seq)

        # 转回卷积格式
        x_out = x_seq.permute(0, 3, 1, 2)

        # 输出投影
        x_out = self.proj_out(x_out)

        # [方案二新增] 自适应门控
        gate = self.gate(x)  # (B, 1, 1, 1)

        # 残差连接: gate * residual_scale * x_out
        out = x + gate * self.residual_scale * x_out

        return out


class WindowAttention(nn.Module):
    """
    基于窗口的多头自注意力
    """

    def __init__(self, dim, window_size, num_heads, drop_path=0.0, bn_momentum=0.2):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        Wh, Ww = window_size, window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * Wh - 1) * (2 * Ww - 1), num_heads)
        )

        coords_h = torch.arange(Wh)
        coords_w = torch.arange(Ww)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += Wh - 1
        relative_coords[:, :, 1] += Ww - 1
        relative_coords[:, :, 0] *= 2 * Ww - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size * self.window_size,
            self.window_size * self.window_size, -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = torch.softmax(attn, dim=-1)
        else:
            attn = torch.softmax(attn, dim=-1)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = self.drop_path(x)

        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block with Window Attention + MLP + LN
    """

    def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=2.0, drop_path=0.0, bn_momentum=0.2):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)

        self.attn = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            drop_path=drop_path,
            bn_momentum=bn_momentum
        )

        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(0.0)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        B, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = self._generate_mask(H, W, x.device)
        else:
            shifted_x = x
            attn_mask = None

        x_windows = self._window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = self._window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = shortcut + self.drop_path(x)

        shortcut2 = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut2 + self.drop_path(x)

        return x

    def _window_partition(self, x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows

    def _window_reverse(self, windows, window_size, H, W):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def _generate_mask(self, H, W, device):
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = self._window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class DetailEnhanceBlock(nn.Module):
    """
    高分辨率细节增强模块：
    - 输入：高分辨率特征，例如 E1 (640x640, 24 通道)
    - 内部：两层 3x3 卷积 + SwitchNorm + ReLU，带残差连接
    - 可选：通道注意力（这里使用简化的 SE 注意力）
    - 输出：与输入同尺寸的细节增强特征，用于与原 E1 融合
    """
    def __init__(self, in_channels, out_channels, reduction=4):
        super(DetailEnhanceBlock, self).__init__()
        mid_channels = out_channels

        # 基础卷积 + 残差
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = SwitchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = SwitchNorm2d(mid_channels)

        # 如果 in_channels != out_channels，则用 1x1 做通道对齐
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = None

        # 简化的 SE 通道注意力
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(out_channels, out_channels // reduction, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 残差主分支
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # shortcut
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out = out + shortcut
        out = self.relu(out)

        # SE 通道注意力
        w = self.global_pool(out)
        w = self.fc1(w)
        w = self.relu(w)
        w = self.fc2(w)
        w = self.sigmoid(w)

        out = out * w
        return out


class SemanticDEBGate(nn.Module):
    """
    DEB 语义门控模块：
    用深层语义特征 semantic_feat 生成空间门控图，
    对高分辨率细节特征 detail_feat 进行抑制或保留。
    """
    def __init__(self, detail_ch, semantic_ch, inter_ch=None, bn_momentum=0.2):
        super(SemanticDEBGate, self).__init__()

        if inter_ch is None:
            inter_ch = detail_ch

        # 先把深层语义特征通道压到和 detail_feat 一样
        self.semantic_proj = nn.Sequential(
            nn.Conv2d(semantic_ch, detail_ch, kernel_size=1, bias=False),
            SwitchNorm2d(detail_ch, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )

        # 根据 detail + semantic 生成门控图
        # 输出 1 通道，表示空间位置的重要性
        self.gate_conv = nn.Sequential(
            nn.Conv2d(detail_ch * 2, inter_ch, kernel_size=3, padding=1, bias=False),
            SwitchNorm2d(inter_ch, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True)
        )

        self.sigmoid = nn.Sigmoid()

        # 初始化时不要过度抑制 DEB，gate 初始约为 sigmoid(1)=0.73
        nn.init.constant_(self.gate_conv[-1].bias, 1.0)

    def forward(self, detail_feat, semantic_feat):
        """
        detail_feat:   DEB 输出特征，例如 enhanced_e1 / enhanced_e2
        semantic_feat: 深层语义特征，例如 out3
        """

        # 先压通道，再上采样，省显存
        semantic_feat = self.semantic_proj(semantic_feat)

        semantic_feat = F.interpolate(
            semantic_feat,
            size=detail_feat.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        gate = self.gate_conv(torch.cat([detail_feat, semantic_feat], dim=1))
        gate = self.sigmoid(gate)

        detail_feat = detail_feat * gate

        return detail_feat



class Post2d(nn.Module):
    def __init__(self, n_in, n_out, stride = 1):
        super(Post2d, self).__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(inplace = True)
        # self.conv2 = nn.Conv2d(n_out, n_out, kernel_size = 3, padding = 1)
        # self.bn2 = nn.BatchNorm2d(n_out)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.relu(out)
        return out



class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Sparsemax(nn.Module):
    """
    Sparsemax activation function.
    这是实现稀疏注意力的核心，它会将低概率的背景权重直接置为 0，而不是像 Softmax/Sigmoid 那样保留极小值。
    """

    def __init__(self, dim=None):
        super(Sparsemax, self).__init__()
        self.dim = -1 if dim is None else dim

    def forward(self, input):
        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, step=1, device=input.device, dtype=input.dtype).view(1,
                                                                                                                     -1)
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim=dim)
        is_gt = bound > cumulative_sum_zs
        k = torch.max(is_gt * range, dim, keepdim=True)[0]
        zs_sparse = is_gt * zs
        tau = (torch.sum(zs_sparse, dim=dim, keepdim=True) - 1) / k
        tau = tau.expand_as(input)

        self.output = torch.max(torch.zeros_like(input), input - tau)

        output = self.output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)
        return output


class Sparse_SN_CS_Parallel_Attention_block(nn.Module):
    """
    Sparse Attention Block with SwitchNorm and Parallel Structure.
    保留了原模型的 SwitchNorm 和 Channel+Spatial 并行结构。
    改进点：将 Spatial Attention 部分的 Sigmoid 替换为 Sparsemax。
    """

    def __init__(self, F_g, F_x, F_int):
        super(Sparse_SN_CS_Parallel_Attention_block, self).__init__()
        reduction_ratio = 2
        F_int = F_g  # 保持原代码逻辑，覆盖了传入的 F_int

        # --- Channel Attention 分支 (保持不变) ---
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x // reduction_ratio),
            nn.ReLU(),
            nn.Linear(F_x // reduction_ratio, F_x)
        )
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_g // reduction_ratio),
            nn.ReLU(),
            nn.Linear(F_g // reduction_ratio, F_x)  # 注意这里最后输出维度映射回 F_x
        )

        # --- Spatial Attention 分支 (SwitchNorm + 稀疏化) ---
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            SwitchNorm2d(F_int)  # [保留 SwitchNorm]
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_x, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            SwitchNorm2d(F_int)  # [保留 SwitchNorm]
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            SwitchNorm2d(1),  # [保留 SwitchNorm]
            # [XSimple Modified]:
            # 删除了这里的 nn.Sigmoid()。
            # 因为我们需要原始的 logits 输出来进行 Sparsemax 竞争。
        )

        self.relu = nn.ReLU(inplace=True)

        # [XSimple Modified]: 引入 Sparsemax
        # dim=2 指的是在 flattened 后的 feature list 上进行竞争
        self.sparsemax = Sparsemax(dim=2)

    def forward(self, g, x):
        # 1. Channel-wise attention (保持原逻辑不变)
        # 这部分负责筛选哪些 filter 对 MA 检测更重要
        avg_pool_x = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)

        avg_pool_g = F.avg_pool2d(g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)

        channel_att_sum = channel_att_x + channel_att_g
        # 通道注意力维持 Sigmoid 即可，因为不需要空间稀疏，只需要通道权重
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale

        # 2. Spatial-wise attention (核心修改部分)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)

        psi = self.psi(psi)  # 输出 shape: [Batch, 1, H, W]，此处是未归一化的 logits

        # [XSimple Modified]: 应用稀疏注意力
        b, c, h, w = psi.size()
        # 展平 H*W 维度，让所有像素点参与竞争
        psi_flat = psi.view(b, c, -1)

        # Sparsemax 会将非显著区域（背景）的权重强制截断为 0
        att_weights = self.sparsemax(psi_flat)

        # 恢复形状
        att_weights = att_weights.view(b, c, h, w)

        # 使用稀疏权重图对原始特征进行加权
        x_after_spacial = x * att_weights

        # 3. 融合输出
        out = self.relu(x_after_spacial + x_after_channel)
        return out


class SN_CS_Parallel_Attention_block(nn.Module):
    """
    Attention Block
    """
    def __init__(self, F_g, F_x, F_int):
        super(SN_CS_Parallel_Attention_block, self).__init__()
        reduction_ratio = 2
        F_int = F_g
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x // reduction_ratio),
            nn.ReLU(),
            nn.Linear(F_x // reduction_ratio, F_x)
        )
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_g // reduction_ratio),
            nn.ReLU(),
            nn.Linear(F_g // reduction_ratio, F_x)
        )
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            SwitchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_x, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            SwitchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            SwitchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # channel-wise attention
        avg_pool_x = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))

        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d( g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = channel_att_x + channel_att_g
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale

        # spacial-wise attention
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        x_after_spacial = x * psi
        out = self.relu(x_after_spacial + x_after_channel)
        return out

class CS_Parallel_Attention_block(nn.Module):
    """
    Attention Block
    """
    def __init__(self, F_g, F_x, F_int):
        super(CS_Parallel_Attention_block, self).__init__()
        reduction_ratio = 2
        F_int = F_g
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x // reduction_ratio),
            nn.ReLU(),
            nn.Linear(F_x // reduction_ratio, F_x)
        )
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_g // reduction_ratio),
            nn.ReLU(),
            nn.Linear(F_g // reduction_ratio, F_x)
        )
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            SwitchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_x, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            SwitchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            SwitchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # channel-wise attention
        avg_pool_x = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))

        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d( g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = channel_att_x + channel_att_g
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale

        # spacial-wise attention
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        x_after_spacial = x * psi
        out = self.relu(x_after_spacial + x_after_channel)
        return out

class CS_Attention_block(nn.Module):
    """
    Attention Block
    """
    def __init__(self, F_g, F_x, F_int):
        super(CS_Attention_block, self).__init__()
        reduction_ratio = 2
        # F_int = F_g
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x // reduction_ratio),
            nn.ReLU(),
            nn.Linear(F_x // reduction_ratio, F_x)
        )
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_g // reduction_ratio),
            nn.ReLU(),
            nn.Linear(F_g // reduction_ratio, F_x)
        )
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_x, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # channel-wise attention
        avg_pool_x = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        # print avg_pool_x.Size()
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d( g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = channel_att_x + channel_att_g
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale

        # spacial-wise attention
        g1 = self.W_g(g)
        x1 = self.W_x(x_after_channel)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x_after_channel * psi
        return out



class FCAttention_block(nn.Module):
    """
    Attention Block
    """
    def __init__(self, F_g, F_x1, F_x2, F_x3, F_x4, F_int):
        super(FCAttention_block, self).__init__()

        bn_momentum = 0.1

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int,momentum=bn_momentum)
        )

        self.W_x1 = nn.Sequential(
            nn.Conv2d(F_x1, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int,momentum=bn_momentum)
        )
        self.W_x2 = nn.Sequential(
            nn.Conv2d(F_x2, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int,momentum=bn_momentum)
        )
        self.W_x3 = nn.Sequential(
            nn.Conv2d(F_x3, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int,momentum=bn_momentum)
        )
        self.W_x4 = nn.Sequential(
            nn.Conv2d(F_x4, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int,momentum=bn_momentum)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1,momentum=bn_momentum),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x1, x2, x3, x4):
        g1 = self.W_g(g)
        x11 = self.W_x1(x1)
        x22 = self.W_x2(x2)
        x33 = self.W_x3(x3)
        x44 = self.W_x4(x4)
        psi = self.relu(g1 + x11 + x22 + x33 + x44)
        psi = self.psi(psi)
        x1  = x1 * psi
        x2  = x2 * psi
        x3  = x3 * psi
        x4  = x4 * psi
        return x1, x2, x3, x4



class Attention_block(nn.Module):
    """
    Attention Block
    """
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


# norm_layer = partial(SynchronizedBatchNorm2d, momentum=0.1)



class SN_PostRes2d(nn.Module):
    def __init__(self, n_in, n_out, stride = 1):
        super(SN_PostRes2d, self).__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = SwitchNorm2d(n_out)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(n_out, n_out, kernel_size = 3, padding = 1)
        self.bn2 = SwitchNorm2d(n_out)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv2d(n_in, n_out, kernel_size = 1, stride = stride),
                SwitchNorm2d(n_out))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class PostRes2d(nn.Module):
    def __init__(self, n_in, n_out, stride = 1):
        super(PostRes2d, self).__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(n_out, n_out, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(n_out)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv2d(n_in, n_out, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(n_out))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        return out
    
class PostRes(nn.Module):
    def __init__(self, n_in, n_out, stride = 1,bn_momentum=0.2):
        super(PostRes, self).__init__()
        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm3d(num_features=n_out,momentum=bn_momentum)
        self.relu = nn.LeakyReLU(inplace = True)
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm3d(num_features=n_out,momentum=bn_momentum)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv3d(n_in, n_out, kernel_size = 1, stride = stride),
                nn.BatchNorm3d(num_features=n_out,momentum=bn_momentum))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        return out

class CU(nn.Module):

    def __init__(self, n_in, n_out, stride = 1,bn_momentum=0.2):
        super(CU, self).__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(num_features=n_out,momentum=bn_momentum)
        self.relu = nn.LeakyReLU(inplace = True)
        self.conv2 = nn.Conv2d(n_out, n_out, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(num_features=n_out,momentum=bn_momentum)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out

class Upsample_1(nn.Module):
    def __init__(self,n_in,n_out,up_rate,bn_momentum=0.2):
        super(Upsample_1,self).__init__()

        self.conv=nn.Conv2d(n_in,n_out,kernel_size=1)
        self.bn=nn.BatchNorm2d(n_out,momentum=bn_momentum)
        self.relu=nn.ReLU(inplace=True)
        self.scale_factor=up_rate

    def forward(self,x):
        f=self.scale_factor
        out=self.conv(x)
        out=self.bn(out)
        out=self.relu(out)
        out=interpolate(input=out,scale_factor=(f,f),mode="bilinear",align_corners=True)
        return out

class SELayer(nn.Module):
    def __init__(self,channel,reduction=16):
        super(SELayer,self).__init__()

        self.avg_pool=nn.AdaptiveAvgPool3d(1)
        self.fc=nn.Sequential(
            nn.Linear(channel,channel//reduction,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction,channel,bias=False),
            nn.Sigmoid()
        )
    
    def forward(self,x):
        b,c,_,_,_=x.size()
        y=self.avg_pool(x).view(b,c)
        y=self.fc(y).view(b,c,1,1,1)
        return x*y.expand_as(x)
         

class SEResNetLayer(nn.Module):

    def __init__(self, n_in, n_out, stride = 1,bn_momentum=0.2):
        super(SEResNetLayer, self).__init__()

        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm3d(num_features=n_out,momentum=bn_momentum)
        self.relu = nn.LeakyReLU(inplace = True)
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm3d(num_features=n_out,momentum=bn_momentum)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv3d(n_in, n_out, kernel_size = 1, stride = stride),
                nn.BatchNorm3d(num_features=n_out,momentum=bn_momentum))
        else:
            self.shortcut = None

        self.avg_pool=nn.AdaptiveAvgPool3d(1)
        self.fc=nn.Sequential(
            nn.Linear(channel,channel//reduction,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction,channel,bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        b,c,_,_,_=out.size()
        y=self.avg_pool(out).view(b,c)
        y=self.fc(y).view(b,c,1,1,1)
        scale=out*y.expand_as(x)

        scale += residual
        scale =self.relu(scale)
        return scale

class Rec3(nn.Module):
    def __init__(self, n0, n1, n2, n3, p = 0.0, integrate = True):
        super(Rec3, self).__init__()
        
        self.block01 = nn.Sequential(
            nn.Conv3d(n0, n1, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm3d(n1),
            nn.ReLU(inplace = True),
            nn.Conv3d(n1, n1, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n1))

        self.block11 = nn.Sequential(
            nn.Conv3d(n1, n1, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n1),
            nn.ReLU(inplace = True),
            nn.Conv3d(n1, n1, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n1))
        
        self.block21 = nn.Sequential(
            nn.ConvTranspose3d(n2, n1, kernel_size = 2, stride = 2),
            nn.BatchNorm3d(n1),
            nn.ReLU(inplace = True),
            nn.Conv3d(n1, n1, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n1))
 
        self.block12 = nn.Sequential(
            nn.Conv3d(n1, n2, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm3d(n2),
            nn.ReLU(inplace = True),
            nn.Conv3d(n2, n2, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n2))
        
        self.block22 = nn.Sequential(
            nn.Conv3d(n2, n2, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n2),
            nn.ReLU(inplace = True),
            nn.Conv3d(n2, n2, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n2))
        
        self.block32 = nn.Sequential(
            nn.ConvTranspose3d(n3, n2, kernel_size = 2, stride = 2),
            nn.BatchNorm3d(n2),
            nn.ReLU(inplace = True),
            nn.Conv3d(n2, n2, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n2))
 
        self.block23 = nn.Sequential(
            nn.Conv3d(n2, n3, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm3d(n3),
            nn.ReLU(inplace = True),
            nn.Conv3d(n3, n3, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n3))

        self.block33 = nn.Sequential(
            nn.Conv3d(n3, n3, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n3),
            nn.ReLU(inplace = True),
            nn.Conv3d(n3, n3, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n3))

        self.relu = nn.ReLU(inplace = True)
        self.p = p
        self.integrate = integrate

    def forward(self, x0, x1, x2, x3):
        if self.p > 0 and self.training:
            coef = torch.bernoulli((1.0 - self.p) * torch.ones(8))
            out1 = coef[0] * self.block01(x0) + coef[1] * self.block11(x1) + coef[2] * self.block21(x2)
            out2 = coef[3] * self.block12(x1) + coef[4] * self.block22(x2) + coef[5] * self.block32(x3)
            out3 = coef[6] * self.block23(x2) + coef[7] * self.block33(x3)
        else:
            out1 = (1 - self.p) * (self.block01(x0) + self.block11(x1) + self.block21(x2))
            out2 = (1 - self.p) * (self.block12(x1) + self.block22(x2) + self.block32(x3))
            out3 = (1 - self.p) * (self.block23(x2) + self.block33(x3))

        if self.integrate:
            out1 += x1
            out2 += x2
            out3 += x3

        return x0, self.relu(out1), self.relu(out2), self.relu(out3)

def hard_mining(neg_output, neg_labels, num_hard):
    _, idcs = torch.topk(neg_output, min(num_hard, len(neg_output)))
    neg_output = torch.index_select(neg_output, 0, idcs)
    neg_labels = torch.index_select(neg_labels, 0, idcs)
    return neg_output, neg_labels

class Loss(nn.Module):
    def __init__(self, num_hard = 0):
        super(Loss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCELoss() # 二分类交叉熵损失函数``
        self.regress_loss = nn.SmoothL1Loss()
        self.num_hard = num_hard # =2

    def forward(self, output, labels, train = True):
        # labels.shape = [batch_size,32,32,32,3,5]
        batch_size = labels.size(0)
        #print('labels.shape = ',labels.shape)
        output = output.view(-1, 5)
        labels = labels.view(-1, 5)
        #print('after reshape,labels.shape = ',labels.shape)
        
        pos_idcs = labels[:, 0] > 0.5
        #print('pos_idcs = ',np.where(pos_idcs))
        pos_idcs = pos_idcs.unsqueeze(1).expand(pos_idcs.size(0), 5)
        pos_output = output[pos_idcs].view(-1, 5) # 过滤出标签是正例的网络的output（因为只有标签是正例的才参与计算）
        pos_labels = labels[pos_idcs].view(-1, 5) # 过滤出标签是正例的label

        neg_idcs = labels[:, 0] < -0.5
        neg_output = output[:, 0][neg_idcs] # 只有label是反例的输出的class部分才参与计算,其余地方需要回归的值不参与计算
        neg_labels = labels[:, 0][neg_idcs]
        
        if self.num_hard > 0 and train:
            neg_output, neg_labels = hard_mining(neg_output, neg_labels, self.num_hard * batch_size)
        neg_prob = self.sigmoid(neg_output)

        #classify_loss = self.classify_loss(
         #   torch.cat((pos_prob, neg_prob), 0),
          #  torch.cat((pos_labels[:, 0], neg_labels + 1), 0))
        if len(pos_output)>0:
            pos_prob = self.sigmoid(pos_output[:, 0])
            pz, ph, pw, pd = pos_output[:, 1], pos_output[:, 2], pos_output[:, 3], pos_output[:, 4]
            lz, lh, lw, ld = pos_labels[:, 1], pos_labels[:, 2], pos_labels[:, 3], pos_labels[:, 4]

            regress_losses = [
                self.regress_loss(pz, lz),
                self.regress_loss(ph, lh),
                self.regress_loss(pw, lw),
                self.regress_loss(pd, ld)]
            regress_losses_data = [l.data[0] for l in regress_losses]
            classify_loss = 0.5 * self.classify_loss(
            pos_prob, pos_labels[:, 0]) + 0.5 * self.classify_loss(
            neg_prob, neg_labels + 1)
            pos_correct = (pos_prob.data >= 0.5).sum() # tp数
            pos_total = len(pos_prob) # 正例总数，即tp+fn

        else:
            regress_losses = [0,0,0,0]
            classify_loss =  0.5 * self.classify_loss(
            neg_prob, neg_labels + 1)
            pos_correct = 0
            pos_total = 0
            regress_losses_data = [0,0,0,0]
        classify_loss_data = classify_loss.data[0]

        loss = classify_loss
        for regress_loss in regress_losses:
            loss += regress_loss

        neg_correct = (neg_prob.data < 0.5).sum() # tn数
        neg_total = len(neg_prob) # 反例总数，即tn+fp

        return [loss, classify_loss_data] + regress_losses_data + [pos_correct, pos_total, neg_correct, neg_total] # 损失函数返回的格式是[总的损失，分类损失，回归损失1,2,3,4，tp,tp+fn,tn,tn+fp]

class GetPBB(object):
    def __init__(self, config):
        self.stride = config['stride']
        self.anchors = np.asarray(config['anchors'])

    def __call__(self, output,thresh = -3, ismask=False):
        stride = self.stride
        anchors = self.anchors
        output = np.copy(output)
        offset = (float(stride) - 1) / 2
        output_size = output.shape
        oz = np.arange(offset, offset + stride * (output_size[0] - 1) + 1, stride)
        oh = np.arange(offset, offset + stride * (output_size[1] - 1) + 1, stride)
        ow = np.arange(offset, offset + stride * (output_size[2] - 1) + 1, stride)
        
        output[:, :, :, :, 1] = oz.reshape((-1, 1, 1, 1)) + output[:, :, :, :, 1] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 2] = oh.reshape((1, -1, 1, 1)) + output[:, :, :, :, 2] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 3] = ow.reshape((1, 1, -1, 1)) + output[:, :, :, :, 3] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 4] = np.exp(output[:, :, :, :, 4]) * anchors.reshape((1, 1, 1, -1))
        mask = output[..., 0] > thresh # 因为sigmoid是在Loss中计算的，而测试的时候没有计算Loss，所以这里的thresh是-3，相当于只取得分大于sigmoid(-3)的部分。
        xx,yy,zz,aa = np.where(mask)
        
        output = output[xx,yy,zz,aa]
        if ismask:
            return output,[xx,yy,zz,aa]
        else:
            return output

        #output = output[output[:, 0] >= self.conf_th] 
        #bboxes = nms(output, self.nms_th)

def iou(box0, box1): #计算iou
    
    r0 = box0[3] / 2
    s0 = box0[:3] - r0
    e0 = box0[:3] + r0

    r1 = box1[3] / 2
    s1 = box1[:3] - r1
    e1 = box1[:3] + r1

    overlap = []
    for i in range(len(s0)):
        overlap.append(max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])))

    intersection = overlap[0] * overlap[1] * overlap[2]
    union = box0[3] * box0[3] * box0[3] + box1[3] * box1[3] * box1[3] - intersection
    return intersection / union


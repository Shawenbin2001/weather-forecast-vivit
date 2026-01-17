import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_1d_sincos_pos_embed(embed_dim, length):
    # returns (length, embed_dim)
    position = torch.arange(length).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
    pe = torch.zeros(length, embed_dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def _get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w):
    # split embed dim between H and W
    assert embed_dim % 2 == 0, "embed_dim must be even for 2D sincos"
    emb_h = embed_dim // 2
    emb_w = embed_dim - emb_h
    pe_h = _get_1d_sincos_pos_embed(emb_h, grid_h)  # (grid_h, emb_h)
    pe_w = _get_1d_sincos_pos_embed(emb_w, grid_w)  # (grid_w, emb_w)
    # outer sum to make grid
    pe = torch.zeros(grid_h * grid_w, embed_dim)
    for i in range(grid_h):
        for j in range(grid_w):
            pe[i * grid_w + j, :] = torch.cat([pe_h[i], pe_w[j]], dim=0)
    return pe  # (grid_h*grid_w, embed_dim)


class VideoViT(nn.Module):
    """
    Video ViT with SEPARATED spatial-temporal attention (先空间、后时间)
    Input shape: (B, T, C, H, W)
    - Spatial Transformer: 对每个时间步的单帧计算空间注意力
    - Temporal Transformer: 对空间处理后的帧特征计算时间注意力
    """
    def __init__(
        self,
        in_ch=3,
        embed_dim=512,
        patch_size=(16, 16),
        spatial_depth=4,  # 空间Transformer的层数
        temporal_depth=2, # 时间Transformer的层数
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.embed_dim = embed_dim
        self.patch_h, self.patch_w = patch_size

        # 1. Patch投影（单帧）
        self.patch_proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

        # 2. 空间Transformer（处理单帧内的patch注意力）
        spatial_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, 
            dim_feedforward=int(embed_dim * mlp_ratio), 
            dropout=dropout, activation='gelu',
            batch_first=True  # 设为True，输入格式为 (B, S, D)，更直观
        )
        self.spatial_encoder = nn.TransformerEncoder(spatial_encoder_layer, num_layers=spatial_depth)

        # 3. 时间Transformer（处理帧之间的注意力）
        temporal_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout, activation='gelu',
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_encoder_layer, num_layers=temporal_depth)

        # 4. 解码器：将token还原为像素
        self.patch_recon = nn.Linear(embed_dim, patch_size[0] * patch_size[1] * in_ch)
        self.conv_last = nn.Conv2d(in_ch, in_ch, kernel_size=(7,7), stride=(1,1), padding=(3,3))

    def _pad_input(self, x):
        # 补齐输入尺寸到patch_size的整数倍（可选，原代码注释了，这里保留）
        b, t, c, h, w = x.shape
        ph, pw = self.patch_h, self.patch_w
        pad_h = (-(h) % ph)
        pad_w = (-(w) % pw)
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0)
        x = x.view(b * t, c, h, w)
        x = F.pad(x, (0, pad_w, 0, pad_h))
        new_h = h + pad_h
        new_w = w + pad_w
        x = x.view(b, t, c, new_h, new_w)
        return x, (pad_h, pad_w)

    def forward(self, x):
        """
        Forward流程：
        1. Patch投影 → 2. 空间注意力（单帧） → 3. 帧特征聚合 → 4. 时间注意力 → 5. 解码重构
        """
        b, t, c, h, w = x.shape
        device = x.device
        
        # Step 1: Patch投影（B*T, C, H, W）→ (B*T, embed_dim, n_h, n_w)
        xt = x.view(b * t, c, h, w)  # 合并B和T维度
        patches = self.patch_proj(xt)  # (B*T, embed_dim, n_h, n_w)
        n_h, n_w = patches.shape[2], patches.shape[3]
        n_patches = n_h * n_w
        
        # Step 2: 整理空间token格式 (B*T, n_patches, embed_dim)
        spatial_tokens = patches.flatten(2).transpose(1, 2)  # (B*T, n_patches, embed_dim)
        
        # Step 3: 空间位置编码 + 空间Transformer（只算单帧内的patch注意力）
        pos_spatial = _get_2d_sincos_pos_embed(self.embed_dim, n_h, n_w).to(device)  # (n_patches, D)
        spatial_tokens = spatial_tokens + pos_spatial.unsqueeze(0)  # 加位置编码
        spatial_encoded = self.spatial_encoder(spatial_tokens)  # (B*T, n_patches, D)
        
        # Step 4: 聚合单帧特征（取所有patch的均值，作为该帧的全局特征）
        frame_features = spatial_encoded  # (B*T, D) → 每个帧的全局特征
        frame_features = frame_features.view(b*n_patches, t, self.embed_dim)  # (B*np, T, D)
        
        # Step 5: 时间位置编码 + 时间Transformer（只算帧之间的注意力）
        pos_temporal = _get_1d_sincos_pos_embed(self.embed_dim, t).to(device)  # (T, D)
        temporal_tokens = frame_features + pos_temporal.unsqueeze(0)  # 加时间位置编码
        temporal_encoded = self.temporal_encoder(temporal_tokens)  # (B*np, T, D)
        
        # Step 6: 把时间增强后的帧特征广播回所有patch（恢复空间维度）
        # (B, T, D) → (B, T, n_patches, D) → (B*T, n_patches, D)
        #enhanced_tokens = temporal_encoded.unsqueeze(2).expand(-1, -1, n_patches, -1)
        enhanced_tokens = temporal_encoded.reshape(b*t, n_patches, self.embed_dim)
        
        # Step 7: 解码重构像素
        patches_vec = self.patch_recon(enhanced_tokens)  # (B*T, n_patches, ph*pw*C)
        # 重新排列为图像格式
        patches_vec = patches_vec.view(b * t, n_patches, self.in_ch, self.patch_h, self.patch_w)
        patches_vec = patches_vec.view(b * t, n_h, n_w, self.in_ch, self.patch_h, self.patch_w)
        patches_vec = patches_vec.permute(0, 3, 1, 4, 2, 5)  # (B*T, C, n_h, ph, n_w, pw)
        recon = patches_vec.contiguous().view(b * t, self.in_ch, n_h * self.patch_h, n_w * self.patch_w)
        #print("recon shape before conv_last:", recon.shape)
        recon = self.conv_last(recon).reshape(b, self.in_ch, t, n_h * self.patch_h, n_w * self.patch_w)  # (B, C, T, H, W)
        #print("recon shape after conv_last:", recon.shape)
        
        # 裁剪回原尺寸（如果有padding）
        #recon = recon.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        return recon


if __name__ == "__main__":
    # 测试分离式时空注意力的VideoViT
    model = VideoViT(
        in_ch=3, 
        embed_dim=256, 
        patch_size=(8, 8), 
        spatial_depth=4,  # 空间注意力层数
        temporal_depth=4, # 时间注意力层数
        num_heads=4
    )
    x = torch.randn(2, 3, 3, 160, 280)  # B=2, T=5, C=3, H=160, W=280
    y = model(x)
    print("input shape:", x.shape)
    print("output shape:", y.shape)
    assert y.shape == x.shape, "输入输出尺寸不一致！"
    print("测试通过：输入输出尺寸匹配")
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class AdaptiveUpsample(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * (scale_factor**2), 3, padding=1, bias=False)
        self.shuffle = nn.PixelShuffle(scale_factor)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.shuffle(self.conv(x)))

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim*mlp_ratio), dropout=dropout)
    def forward(self, x):
        res = x
        x, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + res
        return self.mlp(self.norm2(x)) + x

class CrossModalityFusionWithTransformer(nn.Module):
    def __init__(self, in_ch_hsi, in_ch_sar, mid_ch=64, out_ch=64, trans_depth=1):
        super().__init__()
        self.hsi_conv = nn.Sequential(nn.Conv2d(in_ch_hsi, mid_ch, 3, padding=1, bias=False), nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True))
        self.sar_conv = nn.Sequential(nn.Conv2d(in_ch_sar, mid_ch, 3, padding=1, bias=False), nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True))
        self.attn_gen = nn.Sequential(nn.Conv2d(mid_ch*2, mid_ch, 1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(mid_ch, 1, 1), nn.Sigmoid())
        self.fuse_conv = nn.Sequential(nn.Conv2d(mid_ch*2, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.transformers = nn.Sequential(*[TransformerBlock(dim=out_ch, num_heads=4) for _ in range(trans_depth)])

    def forward(self, hsi, sar):
        fh = self.hsi_conv(hsi)
        fs = self.sar_conv(sar)
        cat = torch.cat([fh, fs], dim=1)
        attn_map = self.attn_gen(cat)
        fs_attn = fs * attn_map
        fused = torch.cat([fh, fs_attn], dim=1)
        out = self.fuse_conv(fused)
        B, C, H, W = out.shape
        x = out.flatten(2).transpose(1, 2)
        x = self.transformers(x)
        return x.transpose(1, 2).view(B, C, H, W), attn_map

class MultiSourceClassifier(nn.Module):
    def __init__(self, num_classes, scale=2, up_ch=64, mid_ch=64, out_ch=64, trans_depth=1, hsi_ch=30, sar_ch=4):
        super().__init__()
        self.hsi_up = AdaptiveUpsample(hsi_ch, up_ch, scale_factor=scale)
        self.sar_up = AdaptiveUpsample(sar_ch, up_ch, scale_factor=scale)
        self.fusion = CrossModalityFusionWithTransformer(in_ch_hsi=up_ch, in_ch_sar=up_ch, mid_ch=mid_ch, out_ch=out_ch, trans_depth=trans_depth)
        self.gate_fc = nn.Linear(out_ch * 2, out_ch)
        self.classifier = nn.Sequential(nn.Linear(out_ch, 256), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(256, num_classes))

    def forward(self, hsi, sar):
        uh = self.hsi_up(hsi)
        us = self.sar_up(sar)
        fused, attn_map = self.fusion(uh, us)
        B, C, H, W = fused.shape
        center_feat = fused[:, :, H//2, W//2]
        context_feat = F.adaptive_avg_pool2d(fused, 1).view(B, C)
        cat = torch.cat([center_feat, context_feat], dim=1)
        gate = torch.sigmoid(self.gate_fc(cat))
        combined = gate * center_feat + (1 - gate) * context_feat
        return self.classifier(combined)

class ResNetSAREncoder(nn.Module):
    """
    基于 ResNet-18 的 SAR 编码器，具有更强的特征提取能力。
    使用 torchvision 的预训练权重初始化 (如果可用),或从头训练。
    """
    def __init__(self, in_ch=4, feat_dim=256, pretrained=False):
        super().__init__()
        # 使用 torchvision 的 resnet18 作为 backbone
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            if pretrained and ResNet18_Weights is not None:
                # 使用预训练权重
                resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                print("[INFO] ResNetSAREncoder: 使用 ImageNet 预训练权重")
            else:
                resnet = resnet18(weights=None)
                print("[INFO] ResNetSAREncoder: 从头开始训练")
        except ImportError:
            from torchvision.models import resnet18
            resnet = resnet18(pretrained=pretrained)
            print(f"[INFO] ResNetSAREncoder: pretrained={pretrained}")
        
        # 修改第一层卷积以适配 4 通道 SAR 输入
        old_conv1 = resnet.conv1
        resnet.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 复制预训练权重 (正确处理多通道)
        if pretrained:
            if in_ch == 3:
                # 3 通道直接复制
                resnet.conv1.weight.data.copy_(old_conv1.weight.data)
            elif in_ch == 4:
                # 4 通道：将 RGB 权重平均扩展到 4 通道
                # 方法：对 RGB 三通道在 dim=1 上求平均，然后扩展到 4 通道
                rgb_mean = old_conv1.weight.mean(dim=1, keepdim=True)  # [64, 1, 7, 7]
                new_weight = rgb_mean.expand(-1, 4, -1, -1) / 3.0  # 平均分配到 4 通道
                resnet.conv1.weight.data.copy_(new_weight)
                print(f"[INFO] ResNetSAREncoder: RGB 权重平均扩展到 4 通道")
            elif in_ch < 3:
                # <3 通道：只取前 in_ch 个通道
                with torch.no_grad():
                    resnet.conv1.weight[:, :in_ch, :, :] = old_conv1.weight[:, :in_ch, :, :]
            else:
                # >4 通道：复制 RGB 后，其余通道用平均值
                with torch.no_grad():
                    resnet.conv1.weight[:, :3, :, :] = old_conv1.weight
                    for c in range(3, in_ch):
                        resnet.conv1.weight[:, c:c+1, :, :] = old_conv1.weight.mean(dim=1, keepdim=True) / 3.0
        
        # 移除原始的全连接层，替换为特征投影头
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        
        # 特征投影头 (512 -> feat_dim)
        self.proj = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # 稍微增大 dropout 防止过拟合
            nn.Linear(512, feat_dim)
        )
        
    def forward(self, x):
        x = self.backbone(x).view(x.size(0), -1)  # [B, 512]
        return F.normalize(self.proj(x), dim=1)


class PolarimetricSAREncoder(nn.Module):
    """
    专门为 PolSAR 设计的编码器，考虑极化散射特性。
    核心创新:
    1. 极化通道注意力：学习 hh/hv/vh/vv 的重要性权重
    2. 相干矩阵特征：模拟极化分解的物理意义
    3. 多尺度融合：捕获不同尺度的散射机制
    """
    def __init__(self, in_ch=4, feat_dim=256):
        super().__init__()
        # ========== 1. 极化通道注意力 ==========
        # PolSAR 的 4 个通道不是独立的，存在物理关联
        self.pol_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, in_ch // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // 2, in_ch),
            nn.Sigmoid()
        )
        
        # ========== 2. 浅层特征提取 (保持细节) ==========
        self.shallow_conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # ========== 3. 极化相干块 (模拟相干矩阵) ==========
        # 用 1x1 卷积学习通道间的相干性
        self.coh_block = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # ========== 4. 多尺度特征提取 ==========
        # 不同大小的卷积核捕获不同散射机制
        self.multi_scale = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=k, padding=k//2, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ) for k in [3, 5, 7]  # 3×3, 5×5, 7×7
        ])
        
        # ========== 5. 深层特征压缩 ==========
        self.deep_compress = nn.Sequential(
            nn.Conv2d(64 * 3, 128, 3, padding=1, bias=False),  # 多尺度拼接后
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # ========== 6. 特征投影头 ==========
        self.proj = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(256, feat_dim)
        )
        
    def forward(self, x):
        # Step 1: 极化通道注意力加权
        pol_weights = self.pol_attention(x).unsqueeze(-1).unsqueeze(-1)
        x = x * pol_weights * 1.2  # 放大注意力效果
        
        # Step 2: 浅层特征
        x = self.shallow_conv(x)
        
        # Step 3: 极化相干块
        x_coh = self.coh_block(x)
        x = x + x_coh  # 残差连接
        
        # Step 4: 多尺度特征提取 + 拼接
        multi_feats = []
        for scale_conv in self.multi_scale:
            multi_feats.append(scale_conv(x))
        x_multi = torch.cat(multi_feats, dim=1)  # [B, 64*3, H, W]
        
        # Step 5: 深层特征压缩
        x = self.deep_compress(x_multi)
        x = x.view(x.size(0), -1)  # [B, 256]
        
        # Step 6: 投影 + L2 归一化
        return F.normalize(self.proj(x), dim=1)


# class SAREncoder(nn.Module):
#     def __init__(self, in_ch=4, feat_dim=256):
#         super().__init__()
#         self.conv1 = nn.Sequential(nn.Conv2d(in_ch, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2))
#         self.se1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 64), nn.Sigmoid())
#         self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2))
#         self.se2 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 128), nn.Sigmoid())
#         self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, 512, 3, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1))
#         self.proj = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.15), nn.Linear(512, feat_dim))
#     def forward(self, x):
#         x = self.conv1(x)
#         w1 = self.se1(x).unsqueeze(-1).unsqueeze(-1)
#         x = x * w1 * 1.2
#         x = self.conv2(x)
#         w2 = self.se2(x).unsqueeze(-1).unsqueeze(-1)
#         x = x * w2 * 1.2
#         x = self.conv3(x)
#         x = x.view(x.size(0), -1)
#         return F.normalize(self.proj(x), dim=1)

# class MultiModalEncoder(nn.Module):
#     def __init__(self, hsi_ch=30, sar_ch=1, feat_dim=256):
#         super().__init__()
#         self.hsi_conv = nn.Sequential(nn.Conv2d(hsi_ch, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1))
#         self.sar_conv = nn.Sequential(nn.Conv2d(sar_ch, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1))
#         self.cross_attn = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 512), nn.Sigmoid())
#         self.proj = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.15), nn.Linear(512, feat_dim))
#     def forward(self, hsi, sar):
#         fh = self.hsi_conv(hsi).view(hsi.size(0), -1)
#         fs = self.sar_conv(sar).view(sar.size(0), -1)
#         cat = torch.cat([fh, fs], dim=1)
#         attn = self.cross_attn(cat)
#         cat = cat * attn * 1.15
#         return F.normalize(self.proj(cat), dim=1)

class MultiModalEncoder(nn.Module):
    """
    由 MultiSourceClassifier 改造而来，作为纯粹的多源特征编码器。
    完全保留了原始 MultiSourceClassifier 的所有内部模块和数据流，
    仅移除了最后的分类头，并将门控融合后的特征向量进行投影和归一化后输出。
    输出格式与 SAREncoder 保持一致 (L2 归一化的特征向量)。
    """
    def __init__(self, hsi_ch=30, sar_ch=1, scale=2, up_ch=64, mid_ch=64, out_ch=64, trans_depth=1, feat_dim=256):
        super().__init__()
        # 1) 自适应放大（完全保留）
        self.hsi_up = AdaptiveUpsample(hsi_ch, up_ch, scale_factor=scale)
        self.sar_up = AdaptiveUpsample(sar_ch, up_ch, scale_factor=scale)
        
        # 2) 跨模态融合 + Transformer（完全保留）
        self.fusion = CrossModalityFusionWithTransformer(in_ch_hsi=up_ch, in_ch_sar=up_ch, 
                                                          mid_ch=mid_ch, out_ch=out_ch, 
                                                          trans_depth=trans_depth)
        
        # 3) 中心-上下文门控融合（完全保留）
        self.gate_fc = nn.Linear(out_ch * 2, out_ch)
        
        # 4) 替换：特征投影头（替代原来的分类头）
        # 将门控融合后的特征 (维度 out_ch) 投影到目标特征维度 (feat_dim)
        self.feature_projector = nn.Sequential(
            nn.Linear(out_ch, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(512, feat_dim)
        )
        
        # 保存关键参数，以便与 SAREncoder 等组件配合使用
        self.feat_dim = feat_dim

    def forward(self, hsi, sar):
        """
        输入:
            hsi: 高光谱图像 [B, hsi_ch, H, W]
            sar: SAR图像 [B, sar_ch, H, W]
        输出:
            归一化的特征向量 [B, feat_dim]，与 SAREncoder 输出格式一致
        """
        # Step 1 & 2: 上采样 -> 融合 (完全保留原始数据流)
        uh = self.hsi_up(hsi)              # [B, up_ch, H', W']
        us = self.sar_up(sar)              # [B, up_ch, H', W']
        fused, _ = self.fusion(uh, us)     # [B, out_ch, H', W']  (注意力图在此忽略)

        # Step 3: 中心 + 上下文特征提取与门控融合 (完全保留)
        B, C, H, W = fused.shape
        center_feat = fused[:, :, H//2, W//2]                   # [B, C]
        context_feat = F.adaptive_avg_pool2d(fused, 1).view(B, C)  # [B, C]
        cat = torch.cat([center_feat, context_feat], dim=1)        # [B, 2C]
        gate = torch.sigmoid(self.gate_fc(cat))                    # [B, C]
        combined = gate * center_feat + (1 - gate) * context_feat  # [B, C]  <- 核心融合特征

        # Step 4: 投影与归一化 (将分类任务改为特征编码任务)
        features = self.feature_projector(combined)  # [B, feat_dim]
        return F.normalize(features, dim=1)          # L2 归一化，与 SAREncoder 对齐

class MoCoEncoder(nn.Module):
    def __init__(self, base_encoder, feat_dim=128, queue_size=1024, momentum=0.99, temp=0.07):
        super().__init__()
        self.queue_size = queue_size
        self.m = momentum  # 提高momentum稳定训练
        self.temp = temp   # 稍微提高温度让梯度更平滑
        self.encoder_q = base_encoder
        self.encoder_k = copy.deepcopy(base_encoder)
        for p in self.encoder_k.parameters():
            p.requires_grad = False
        self.register_buffer("queue", torch.randn(feat_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update(self):
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data = self.m * p_k.data + (1 - self.m) * p_q.data

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_sz = keys.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_sz > self.queue_size:
            batch_sz = self.queue_size - ptr
            keys = keys[:batch_sz]
        self.queue[:, ptr:ptr + batch_sz] = keys.T
        self.queue_ptr[0] = (ptr + batch_sz) % self.queue_size

    def forward(self, x_q, x_k=None, is_multimodal=False, hsi_k=None, sar_k=None):
        if is_multimodal:
            q = self.encoder_q(x_q[0], x_q[1])
        else:
            q = self.encoder_q(x_q)
        with torch.no_grad():
            self._momentum_update()
            if is_multimodal and hsi_k is not None:
                k = self.encoder_k(hsi_k, sar_k)
            elif x_k is not None:
                k = self.encoder_k(x_k)
            else:
                k = q.detach()
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temp
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        self._dequeue_and_enqueue(k)
        return logits, labels, q

class CrossDomainContrastive(nn.Module):
    def __init__(self, src_encoder, tgt_encoder, feat_dim=256, queue_size=2048):
        super().__init__()
        self.src_moco = MoCoEncoder(src_encoder, feat_dim, queue_size)
        self.tgt_moco = MoCoEncoder(tgt_encoder, feat_dim, queue_size)
        self.cross_proj = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU(inplace=True), nn.Dropout(0.15), nn.Linear(feat_dim, feat_dim))
    def forward(self, src_data, tgt_data, is_src_multimodal=True):
        if is_src_multimodal:
            src_logits, src_labels, src_feat = self.src_moco((src_data['hsi'], src_data['sar']), is_multimodal=True, hsi_k=src_data.get('hsi_k', src_data['hsi']), sar_k=src_data.get('sar_k', src_data['sar']))
        else:
            src_logits, src_labels, src_feat = self.src_moco(src_data['sar'])
        tgt_logits, tgt_labels, tgt_feat = self.tgt_moco(tgt_data['sar'], tgt_data.get('sar_k'))
        src_proj = F.normalize(self.cross_proj(src_feat), dim=1)
        tgt_proj = F.normalize(self.cross_proj(tgt_feat), dim=1)
        return {'src_logits': src_logits, 'src_labels': src_labels, 'src_feat': src_feat, 'tgt_logits': tgt_logits, 'tgt_labels': tgt_labels, 'tgt_feat': tgt_feat, 'src_proj': src_proj, 'tgt_proj': tgt_proj}

class TransferEncoder(nn.Module):
    def __init__(self, in_ch=4, feat_dim=256):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(in_ch, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(256, feat_dim)
    def forward(self, x):
        x = self.backbone(x).view(x.size(0), -1)
        return F.normalize(self.fc(x), dim=1)


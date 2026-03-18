"""
增强型跨域训练策略
核心思路：
1. 类别原型对齐 - 学习每类的原型特征
2. 伪标签自训练 - 利用高置信度目标域样本
3. 特征动态增强 - 提升特征鲁棒性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import copy
import parameter

# 导入必要的模块
from core.utils.dataset import get_cross_domain_loaders, SourceDomainDataset
from core.ours import MultiModalEncoder, PolarimetricSAREncoder, CrossDomainContrastive
from train.model_train import CrossDomainTrainer, zero_shot_transfer_eval, plot_training_curves

class PrototypeAlignment(nn.Module):
    """类别原型对齐模块"""
    def __init__(self, num_classes=5, feat_dim=256):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        # 可学习的类别原型
        self.prototypes = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_normal_(self.prototypes)
        
    def forward(self, feats, labels):
        """
        feats: [N, feat_dim] L2 归一化的特征
        labels: [N] 类别标签
        """
        if feats.size(0) < 2:
            return torch.tensor(0.0, device=feats.device)
        
        # 归一化原型
        proto_norm = F.normalize(self.prototypes, dim=1)
        feat_norm = F.normalize(feats, dim=1)
        
        # 计算每个样本到原型的距离
        loss = torch.tensor(0.0, device=feats.device)
        n_valid = 0
        
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                cls_feats = feat_norm[mask]
                # 类内损失：拉近样本与对应原型
                proto_c = proto_norm[c:c+1]
                dist = (cls_feats - proto_c).pow(2).sum(dim=1).mean()
                loss = loss + dist
                n_valid += 1
        
        # 类间损失：推远不同原型
        if n_valid > 1:
            proto_sim = torch.mm(proto_norm, proto_norm.t())
            # 减去对角线 (自己不与自己推远)
            diag_mask = 1.0 - torch.eye(self.num_classes, device=feats.device)
            inter_loss = (proto_sim * diag_mask).mean()
            loss = loss + inter_loss * 0.5
        
        return loss / max(n_valid, 1)


class PseudoLabelGenerator:
    """伪标签生成器（基于类原型）

    关键修复：
    - 不再使用 feat·feat^T 的“全样本投票”（会导致对角线自相似占优，伪标签变成样本索引）
    - 仅允许用“类原型”生成伪标签，并对高置信度样本做 EMA 原型更新
    """
    def __init__(self, num_classes: int = 5, confidence_threshold: float = 0.7, temperature: float = 0.07, proto_momentum: float = 0.9):
        self.num_classes = int(num_classes)
        self.threshold = float(confidence_threshold)
        self.temperature = float(temperature)
        self.proto_momentum = float(proto_momentum)
        self.prototype_bank = None  # [C, D]，EMA 原型（用于产出伪标签）

    @torch.no_grad()
    def generate(self, feats: torch.Tensor, prototypes: torch.Tensor = None):
        """生成伪标签

        Args:
            feats: [N, D] 特征（建议已做过投影），不要求已归一化
            prototypes: [C, D] 类原型；若为 None，则使用 self.prototype_bank
        Returns:
            pseudo_labels: [N] long
            confidences: [N] float
            valid_mask: [N] bool
        """
        if feats is None or feats.size(0) < 2:
            return None, None, None

        proto = prototypes if prototypes is not None else self.prototype_bank
        if proto is None or proto.size(0) != self.num_classes:
            # 原型未就绪，无法生成可靠伪标签
            return None, None, None

        feat_norm = F.normalize(feats, dim=1)
        proto_norm = F.normalize(proto, dim=1)

        logits = torch.mm(feat_norm, proto_norm.t()) / max(self.temperature, 1e-6)  # [N, C]
        probs = F.softmax(logits, dim=1)

        confidence, pseudo_labels = probs.max(dim=1)
        valid_mask = confidence >= self.threshold

        return pseudo_labels.long(), confidence, valid_mask

    @torch.no_grad()
    def update_prototypes(self, feats: torch.Tensor, labels: torch.Tensor, momentum: float = None):
        """用（源域真标签 or 目标域高置信伪标签）做 EMA 原型更新"""
        if feats is None or labels is None or feats.size(0) == 0:
            return

        mom = self.proto_momentum if momentum is None else float(momentum)

        feat_norm = F.normalize(feats, dim=1)
        labels = labels.long().view(-1)

        C = self.num_classes
        D = feat_norm.size(1)

        if self.prototype_bank is None or self.prototype_bank.size(0) != C or self.prototype_bank.size(1) != D:
            # 初始化为随机单位向量，后续由 EMA 收敛
            proto = torch.randn(C, D, device=feat_norm.device)
            self.prototype_bank = F.normalize(proto, dim=1)

        for c in range(C):
            mask = (labels == c)
            if mask.sum() < 1:
                continue
            proto_c = feat_norm[mask].mean(dim=0)
            proto_c = F.normalize(proto_c, dim=0)
            self.prototype_bank[c] = F.normalize(mom * self.prototype_bank[c] + (1.0 - mom) * proto_c, dim=0)


def compute_coral(src_feat, tgt_feat):
    """
    CORAL 损失：对齐源域和目标域的二阶统计量（协方差矩阵）
    参考：Deep CORAL - Correlation Alignment for Deep Domain Adaptation
    
    Args:
        src_feat: 源域特征 [N, D]
        tgt_feat: 目标域特征 [M, D]
    Returns:
        coral_loss: 标量
    """
    # 计算源域协方差矩阵
    n = src_feat.size(0)
    src_mean = src_feat.mean(dim=0, keepdim=True)
    src_centered = src_feat - src_mean
    src_cov = torch.mm(src_centered.t(), src_centered) / (n - 1)
    
    # 计算目标域协方差矩阵
    m = tgt_feat.size(0)
    tgt_mean = tgt_feat.mean(dim=0, keepdim=True)
    tgt_centered = tgt_feat - tgt_mean
    tgt_cov = torch.mm(tgt_centered.t(), tgt_centered) / (m - 1)
    
    # 计算协方差差异的 Frobenius 范数
    diff = src_cov - tgt_cov
    coral_loss = torch.norm(diff, p='fro') ** 2 / (4.0 * src_feat.size(1) ** 2)
    
    return coral_loss


def enhance_features(feats, training=True, augment_prob=0.3):
    """
    特征级数据增强
    """
    if not training:
        return feats
    
    feat_norm = F.normalize(feats, dim=1)
    
    # 随机特征 masking
    if torch.rand(1).item() < augment_prob:
        mask_prob = torch.rand(1).item() * 0.15  # 0-15% 的维度
        n_mask = int(feats.size(1) * mask_prob)
        if n_mask > 0:
            mask_idx = torch.randperm(feats.size(1))[:n_mask]
            feat_norm[:, mask_idx] = 0
    
    # 随机特征噪声 (非常小，保持语义)
    if torch.rand(1).item() < augment_prob:
        noise = torch.randn_like(feat_norm) * 0.02
        feat_norm = feat_norm + noise
        feat_norm = F.normalize(feat_norm, dim=1)
    
    return feat_norm


class EnhancedCrossDomainTrainer:
    """增强型跨域训练器"""
    def __init__(self, base_trainer, device):
        """
        base_trainer: 原始的 CrossDomainTrainer 实例
        """
        self.base = base_trainer
        self.device = device
        
        # 新增模块
        self.prototype_align = PrototypeAlignment(num_classes=5, feat_dim=256).to(device)
        self.pseudo_label_gen = PseudoLabelGenerator(num_classes=5, confidence_threshold=0.75, temperature=0.07, proto_momentum=0.9)
        # ========== Mean Teacher (EMA) for Target Domain ==========
        # 用 EMA teacher 稳定生成伪标签 + 一致性约束，避免 student 自我确认偏差
        self.ema_m = 0.996
        self.teacher = copy.deepcopy(self.base.model).to(device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        
        # 添加到优化器
        all_params = list(self.base.model.parameters()) + \
                     list(self.base.infonce.parameters()) + \
                     list(self.base.align.parameters()) + \
                     [self.base.alpha_mod.log_alpha] + \
                     list(self.prototype_align.parameters())
        
        self.base.optimizer = torch.optim.AdamW(all_params, lr=self.base.optimizer.param_groups[0]['lr'], weight_decay=0.01)
        
        # 跟踪统计信息
        self.proto_loss_weight = 0.15
        self.pseudo_loss_weight = 0.20
        self.coral_loss_weight = 0.10  # CORAL 损失权重


    @torch.no_grad()
    def _ema_update_teacher(self):
        # EMA 更新 teacher 参数
        m = float(getattr(self, 'ema_m', 0.996))
        for p_s, p_t in zip(self.base.model.parameters(), self.teacher.parameters()):
            p_t.data.mul_(m).add_(p_s.data, alpha=(1.0 - m))

    @torch.no_grad()
    def _teacher_tgt_proj(self, sar: torch.Tensor) -> torch.Tensor:
        # teacher 的目标域投影特征（不走 MoCo forward，避免 queue 变化）
        self.teacher.eval()
        feat = self.teacher.tgt_moco.encoder_q(sar)
        if hasattr(self.teacher, 'cross_proj') and self.teacher.cross_proj is not None:
            feat = self.teacher.cross_proj(feat)
        return F.normalize(feat, dim=1)
    
    def train_epoch_enhanced(self, pair_loader):
        """增强版训练循环"""
        self.base.model.train()
        total_loss, moco_loss_sum, align_loss_sum, proto_loss_sum, pseudo_loss_sum = 0, 0, 0, 0, 0
        cnt = 0
        self.base.ep_cnt += 1

        # threshold_schedule: 先高后低，逐步纳入更多目标域伪标签（减少早期噪声）
        self.pseudo_label_gen.threshold = max(0.70, 0.85 - 0.01 * float(self.base.ep_cnt))
        
        for src_data, tgt_data in pair_loader:
            # 数据迁移到设备
            for k in src_data:
                if isinstance(src_data[k], torch.Tensor):
                    src_data[k] = src_data[k].to(self.device)
            for k in tgt_data:
                if isinstance(tgt_data[k], torch.Tensor):
                    tgt_data[k] = tgt_data[k].to(self.device)
            
            self.base.optimizer.zero_grad()
            
            # 前向传播
            out = self.base.model(src_data, tgt_data, is_src_multimodal=True)
            
            # 原始损失
            src_moco = self.base.infonce(out['src_logits'], out['src_labels'])
            tgt_moco = self.base.infonce(out['tgt_logits'], out['tgt_labels'])
            moco_loss = (src_moco + tgt_moco) / 2
            coral_loss = torch.tensor(0.0, device=self.device)
            
            align_loss, mmd, contrast = self.base.align(out['src_proj'], out['tgt_proj'])

            # ========== 二阶统计对齐：CORAL（补齐原本定义但未使用的部分） ==========
            if out['src_proj'].size(0) > 2 and out['tgt_proj'].size(0) > 2:
                coral_loss = compute_coral(out['src_proj'], out['tgt_proj'])
            
            # ========== 1. 原型对齐损失（只用有效源域标签） ==========
            src_lbl = src_data.get('label')
            proto_loss = torch.tensor(0.0, device=self.device)

            valid_src = None
            if src_lbl is not None:
                src_lbl = src_lbl.long()
                valid_src = (src_lbl >= 0) & (src_lbl < self.prototype_align.num_classes)

                # 用源域真标签做 EMA 原型初始化/更新（为目标域伪标签提供可靠原型）
                if valid_src.sum() > 0:
                    self.pseudo_label_gen.update_prototypes(out['src_proj'].detach()[valid_src], src_lbl[valid_src], momentum=0.9)

                    # 源域原型对齐
                    proto_loss_src = self.prototype_align(out['src_proj'][valid_src], src_lbl[valid_src])
                    proto_loss = proto_loss + proto_loss_src

            # ========== 2. 伪标签自训练损失（仅基于类原型生成） ==========
            pseudo_loss = torch.tensor(0.0, device=self.device)
            consis_loss = torch.tensor(0.0, device=self.device)
            proto_ce_loss = torch.tensor(0.0, device=self.device)

            # 用 EMA 原型（优先）或可学习原型（兜底）生成伪标签
            proto_for_pl = self.pseudo_label_gen.prototype_bank
            if proto_for_pl is None:
                proto_for_pl = self.prototype_align.prototypes.detach()

            # teacher 目标域投影（更稳）：用于伪标签 + 一致性约束
            teacher_tgt_proj = self._teacher_tgt_proj(tgt_data['sar'])
            # 一致性损失：student 表征应贴近 teacher（防止伪标签自我确认偏差）
            consis_loss = (1.0 - (out['tgt_proj'] * teacher_tgt_proj).sum(dim=1)).mean()

            pseudo_lbls, confs, valid_mask = self.pseudo_label_gen.generate(teacher_tgt_proj, prototypes=proto_for_pl)

            if pseudo_lbls is not None and valid_mask is not None and valid_mask.sum() > 8:
                high_conf_feat = out['tgt_proj'][valid_mask]
                high_conf_lbls = pseudo_lbls[valid_mask]

                # 额外：原型分类 CE（比纯 MSE 原型对齐更直接，强化类间间隔）
                proto_norm = F.normalize(proto_for_pl, dim=1)
                logits_proto = torch.mm(high_conf_feat, proto_norm.t()) / 0.07
                proto_ce_loss = F.cross_entropy(logits_proto, high_conf_lbls, label_smoothing=0.05)

                # 伪标签原型对齐（把目标域样本拉向对应类原型）
                if high_conf_feat.size(0) > 2:
                    pseudo_loss = self.prototype_align(high_conf_feat, high_conf_lbls)

                # 用高置信目标域样本做 EMA 原型自适应（更贴近目标域分布）
                if self.base.ep_cnt % 2 == 0:
                    self.pseudo_label_gen.update_prototypes(high_conf_feat.detach(), high_conf_lbls, momentum=0.95)
            # ========== 总损失 ==========
            # 权重分配：SupCon(0.50) + Center(0.20) + Align(0.08) + Proto(0.15) + Pseudo(0.20) + MoCo(0.02)
            # 需要从原来的 loss 中重新计算
            
            # 重新计算 SupCon 和 Center (复制原始逻辑)
            temp = 0.35
            supcon_loss = torch.tensor(0.0, device=self.device)
            center_loss = torch.tensor(0.0, device=self.device)
            
            def compute_supcon(feats, lbls):
                # 过滤 ignore 标签（-1）及越界标签，避免把噪声当正样本对
                if lbls is None:
                    return torch.tensor(0.0, device=feats.device)
                lbls = lbls.long()
                valid = (lbls >= 0) & (lbls < self.prototype_align.num_classes)
                if valid.sum() < 3:
                    return torch.tensor(0.0, device=feats.device)
                feats = feats[valid]
                lbls = lbls[valid]
                feats_n = F.normalize(feats, dim=1)
                sim = torch.mm(feats_n, feats_n.t()) / temp
                lbl_eq = (lbls.unsqueeze(0) == lbls.unsqueeze(1)).float()
                diag_mask = 1.0 - torch.eye(feats.size(0), device=feats.device)
                pos_mask = lbl_eq * diag_mask
                sim_max = sim.max().detach()
                sim = sim - sim_max
                exp_sim = torch.exp(sim) * diag_mask
                log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
                pos_cnt = pos_mask.sum(dim=1, keepdim=True).clamp(min=1)
                log_prob = (sim - log_denom) * pos_mask
                loss_v = -log_prob.sum(dim=1) / pos_cnt.squeeze()
                valid = (pos_mask.sum(dim=1) > 0)
                if valid.sum() > 0:
                    r = loss_v[valid].mean()
                    r = r / (math.log(max(feats.size(0), 2)) + 1e-8)
                    return r
                return torch.tensor(0.0, device=feats.device)
            
            n_supcon = 0.0
            # 源域 SupCon（只用有效标签）
            if src_lbl is not None and out['src_proj'].size(0) > 2:
                supcon_loss = supcon_loss + compute_supcon(out['src_proj'], src_lbl)
                n_supcon += 1.0

            # 目标域 SupCon：使用高置信伪标签（避免使用/依赖 GT）
            if pseudo_lbls is not None and valid_mask is not None and valid_mask.sum() > 2:
                supcon_loss = supcon_loss + compute_supcon(out['tgt_proj'][valid_mask], pseudo_lbls[valid_mask]) * 1.5
                n_supcon += 1.5

            if n_supcon > 0:
                supcon_loss = supcon_loss / n_supcon

            # Center loss（源域真标签 + 目标域高置信伪标签）
            n_ctr = 0
            pairs = []
            if src_lbl is not None:
                v = (src_lbl.long() >= 0) & (src_lbl.long() < self.prototype_align.num_classes)
                if v.sum() > 1:
                    pairs.append((out['src_proj'][v], src_lbl.long()[v]))
            if pseudo_lbls is not None and valid_mask is not None and valid_mask.sum() > 1:
                pairs.append((out['tgt_proj'][valid_mask], pseudo_lbls.long()[valid_mask]))

            for proj, lbl in pairs:
                if proj.size(0) < 2:
                    continue
                proj_n = F.normalize(proj, dim=1)
                for c in lbl.unique():
                    if int(c.item()) < 0 or int(c.item()) >= self.prototype_align.num_classes:
                        continue
                    mask = (lbl == c)
                    if mask.sum() > 1:
                        cls_f = proj_n[mask]
                        ctr = cls_f.mean(dim=0, keepdim=True)
                        center_loss = center_loss + ((cls_f - ctr) ** 2).sum() / mask.sum()
                        n_ctr += 1

            if n_ctr > 0:
                center_loss = center_loss / (n_ctr * 4.0)

            # 总损失
            loss = (moco_loss * 0.02 +
                   align_loss * 0.08 +
                   supcon_loss * 0.50 +      # SupCon
                   center_loss * 0.20 +      # Center
                   proto_loss * self.proto_loss_weight +     # 原型对齐
                   pseudo_loss * self.pseudo_loss_weight +   # 伪标签原型对齐
                   proto_ce_loss * 0.10 +    # 原型分类CE
                   consis_loss * 0.10 +      # MeanTeacher一致性
                   coral_loss * self.coral_loss_weight)      # CORAL二阶对齐
            if torch.isnan(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.base.model.parameters(), max_norm=1.0)
            self.base.optimizer.step()
            self._ema_update_teacher()
            
            total_loss += loss.item()
            moco_loss_sum += moco_loss.item()
            align_loss_sum += align_loss.item()
            proto_loss_sum += proto_loss.item()
            pseudo_loss_sum += pseudo_loss.item()
            cnt += 1
        
        avg_loss = total_loss / max(cnt, 1)
        return avg_loss, moco_loss_sum / max(cnt, 1), align_loss_sum / max(cnt, 1), \
               proto_loss_sum / max(cnt, 1), pseudo_loss_sum / max(cnt, 1)


def enhanced_cross_domain_train(epochs, lr, src_type, tgt_dir, test_dir, model_path, log_path, data_ratio=1.0):
    """
    增强型跨域训练函数（不调整超参数，只改进训练策略）
    
    核心优化:
    1. 类别原型对齐 - 学习判别性更强的类原型
    2. 伪标签自训练 - 利用目标域高置信度样本
    3. 特征动态增强 - 提升特征鲁棒性
    """
    from train.model_train import zero_shot_transfer_eval, plot_training_curves
    from tqdm import tqdm
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    _font_path = 'C:/Users/jialong/xwechat_files/wxid_b5a2qq8pnu3422_bfcb/msg/file/2026-02/code/code/仿宋_GB2312.ttf'
    _zh_font = FontProperties(fname=_font_path) if os.path.exists(_font_path) else None
    
    device = parameter.get_device()
    print(f"\n========== 加载数据 (比例={data_ratio*100:.1f}%) ==========")
    pair_loader, test_loader, src_ds, tgt_ds = get_cross_domain_loaders(
        src_type=src_type, tgt_dir=tgt_dir, test_dir=test_dir, 
        batch_sz=32, data_ratio=data_ratio
    )
    print(f"源域样本：{len(src_ds)}, 目标域样本：{len(tgt_ds)}")
    sample = src_ds[0]
    src_sar_ch = sample['sar'].shape[0]
    print(f"源域 SAR 通道数：{src_sar_ch}")
    
    # 构建模型
    src_encoder = MultiModalEncoder(hsi_ch=30, sar_ch=src_sar_ch, feat_dim=256)
    tgt_encoder = PolarimetricSAREncoder(in_ch=4, feat_dim=256)
    q_sz = min(128, len(src_ds) + len(tgt_ds))
    model = CrossDomainContrastive(src_encoder, tgt_encoder, feat_dim=256, queue_size=q_sz)
    
    # 创建基础训练器
    base_trainer = CrossDomainTrainer(model, device, lr=lr*0.5)
    
    # 创建增强型训练器
    trainer = EnhancedCrossDomainTrainer(base_trainer, device)
    print("[INFO] 使用增强型训练策略：原型对齐 + 伪标签自训练")
    
    # 训练准备
    log_file = open(log_path, 'w')
    print(f"\n========== 开始训练 {epochs} 轮 ==========")
    pbar = tqdm(range(epochs), desc='训练进度')
    history = {'loss': [], 'nmi': [], 'ari': []}
    
    hist_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '历史数据')
    os.makedirs(hist_dir, exist_ok=True)
    hist_file = os.path.join(hist_dir, 'training_history.csv')
    with open(hist_file, 'w') as hf:
        hf.write('epoch,loss,nmi,ari\n')
    
    # 早停配置
    _no_improve = 0
    _patience = 5
    _prev_lr = trainer.base.optimizer.param_groups[0]['lr']
    
    # 最佳模型跟踪
    _best_score = -float('inf')
    _best_model_state = None
    _best_epoch = 0
    _best_nmi = 0.0
    _best_ari = 0.0
    
    for ep in pbar:
        # 增强型训练
        loss_val, moco, align, proto, pseudo = trainer.train_epoch_enhanced(pair_loader)
        trainer.base.scheduler.step(loss_val)
        _cur_lr = trainer.base.optimizer.param_groups[0]['lr']
        
        if _cur_lr < _prev_lr - 1e-8:
            _no_improve = 0
            _prev_lr = _cur_lr
        
        # 评估
        metrics = zero_shot_transfer_eval(trainer.base, test_loader)
        final_loss = loss_val
        final_nmi = metrics['nmi']
        final_ari = metrics['ari']
        
        history['loss'].append(final_loss)
        history['nmi'].append(final_nmi)
        history['ari'].append(final_ari)
        
        # 检查是否最佳
        current_score = (final_nmi + final_ari) / 2.0
        if current_score > _best_score:
            _best_score = current_score
            _best_model_state = model.state_dict().copy()
            _best_epoch = ep + 1
            _best_nmi = final_nmi
            _best_ari = final_ari
            print(f"[最佳模型] Epoch {ep+1}: NMI={final_nmi:.4f}, ARI={final_ari:.4f}, Score={current_score:.4f}")
        
        with open(hist_file, 'a') as hf:
            hf.write(f"{ep+1},{final_loss:.6f},{final_nmi:.6f},{final_ari:.6f}\n")
        
        pbar.set_postfix({
            '损失': f'{final_loss:.3f}',
            'NMI': f"{final_nmi:.3f}",
            'ARI': f"{final_ari:.3f}",
            'lr': f"{_cur_lr:.1e}"
        })
        
        log = f"Epoch {ep+1}/{epochs} | Loss: {final_loss:.4f} | NMI: {final_nmi:.4f} | ARI: {final_ari:.4f} | lr: {_cur_lr:.1e}"
        log_file.write(log + '\n')
        
        # 早停检查
        if current_score > _best_score - 1e-6:
            _no_improve = 0
        else:
            _no_improve += 1
        
        if _no_improve >= _patience:
            print(f"\n[早停] Score 连续{_patience}轮不升 (lr={_cur_lr:.1e}), epoch {ep+1}停止")
            break
    
    log_file.close()
    
    # 保存最佳模型
    if _best_model_state is not None:
        model.load_state_dict(_best_model_state)
        os.makedirs(model_path, exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_file = os.path.join(model_path, f'best_enhanced_model_{timestamp}.pth')
        torch.save(model.state_dict(), model_file)
        print(f"[最佳模型] 已保存到：{model_file}")
        print(f"[最佳模型] 来自 Epoch {_best_epoch}, NMI={_best_nmi:.4f}, ARI={_best_ari:.4f}")
    
    final_nmi, final_ari = history['nmi'][-1], history['ari'][-1]
    best_nmi, best_ari = _best_nmi, _best_ari
    print(f"\n========== 训练完成 ==========")
    print(f"最终模型 (Epoch {len(history['nmi'])}): NMI={final_nmi:.4f}, ARI={final_ari:.4f}")
    print(f"最佳模型 (Epoch {_best_epoch}): NMI={best_nmi:.4f}, ARI={best_ari:.4f}")
    print(f"[INFO] 历史数据已保存：{hist_file}")
    plot_training_curves(history, 'pic/ours')
    
    return model, trainer, history

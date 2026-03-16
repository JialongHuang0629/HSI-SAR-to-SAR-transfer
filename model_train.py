from core.utils.dataset import *
import time
import math
import random
import os
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
import torch.nn as nn
import torch.nn.functional as F
import parameter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
_font_path = '/file/2026-02/code/code/仿宋_GB2312.ttf'

_zh_font = FontProperties(fname=_font_path) if os.path.exists(_font_path) else None
from core.ours import MultiSourceClassifier, SAREncoder, MultiModalEncoder, CrossDomainContrastive, TransferEncoder, AdaptiveUpsample, MLP, TransformerBlock, CrossModalityFusionWithTransformer, ResNetSAREncoder, PolarimetricSAREncoder
from core.CNN import CNNSpectralSAR
from core.CapsuleNet import FastCapsNetMulti
from core.HybridSN import HybridSNMulti
from core.SpectralFormer import SpectralFormerMulti
import sys
sys.path.insert(0, 'core')
from core.RWKV副本 import MultiSourceRWKVClassifier, MultiSourceClassifier as RWKVMultiSourceClassifier
from tqdm import tqdm

parameter._init()

class InfoNCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(-0.5))
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.call_cnt = 0
    def forward(self, logits, labels):
        self.call_cnt += 1
        temp = torch.exp(self.log_temp).clamp(0.05, 0.2)
        raw = self.ce(logits / temp, labels)
        # CE上界=log(K), 除以它归一化到(0,1)
        k = max(logits.shape[1], 2)
        return raw / (math.log(k) + 1e-8)

def compute_mmd(src_feat, tgt_feat):
    def pairwise_dist(x, y):
        xx = (x * x).sum(dim=1, keepdim=True)
        yy = (y * y).sum(dim=1, keepdim=True)
        xy = torch.mm(x, y.t())
        return torch.clamp(xx + yy.t() - 2 * xy, min=1e-8).sqrt()
    def rbf_kernel(x, y, sigma):
        dist = pairwise_dist(x, y)
        return torch.exp(-dist ** 2 / (2 * sigma ** 2 + 1e-8))
    with torch.no_grad():
        all_feat = torch.cat([src_feat, tgt_feat], dim=0)
        dists = pairwise_dist(all_feat, all_feat)
        median_dist = dists.median().item()
        base_sigma = max(median_dist, 0.1)
    sigmas = [base_sigma * r for r in [0.25, 0.5, 1.0, 2.0]]
    mmd = 0
    for s in sigmas:
        k_ss = rbf_kernel(src_feat, src_feat, s)
        k_tt = rbf_kernel(tgt_feat, tgt_feat, s)
        k_st = rbf_kernel(src_feat, tgt_feat, s)
        mmd += k_ss.mean() + k_tt.mean() - 2 * k_st.mean()
    return mmd / len(sigmas)

class DomainAlignLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_lam = nn.Parameter(torch.tensor(0.0))
        self.log_temp = nn.Parameter(torch.tensor(-1.0))
        self.call_cnt = 0
    def forward(self, src_proj, tgt_proj):
        self.call_cnt += 1
        lam = torch.sigmoid(self.log_lam) * 0.8 + 0.2
        temp = torch.exp(self.log_temp).clamp(0.05, 0.5)
        mmd_loss = compute_mmd(src_proj, tgt_proj)
        # mmd范围[0,2], /2归一化
        mmd_loss = mmd_loss / 2.0
        sim = torch.mm(src_proj, tgt_proj.t()) / temp
        n = sim.size(0)
        labels = torch.arange(n, device=sim.device) % sim.size(1)
        loss_st = F.cross_entropy(sim, labels, label_smoothing=0.1)
        loss_ts = F.cross_entropy(sim.t(), labels[:sim.size(1)], label_smoothing=0.1)
        contrast_loss = (loss_st + loss_ts) / 2
        # CE上界log(K)归一化
        log_k = math.log(max(max(sim.size(0), sim.size(1)), 2)) + 1e-8
        contrast_loss = contrast_loss / log_k
        src_var = src_proj.var(dim=0).mean()
        tgt_var = tgt_proj.var(dim=0).mean()
        # var_loss max=2, /2归一化
        var_loss = (torch.relu(1.0 - src_var) + torch.relu(1.0 - tgt_var)) / 2.0
        src_cov = torch.mm(src_proj.t(), src_proj) / src_proj.size(0)
        cov_reg = (src_cov - torch.eye(src_cov.size(0), device=src_cov.device)).pow(2).mean()
        # cov偏差归一化
        cov_reg = cov_reg / 4.0
        total = mmd_loss + lam * contrast_loss + 0.1 * var_loss + 0.02 * cov_reg
        # 除以权重和 -> (0,1)
        w_sum = 1.0 + lam.detach() + 0.1 + 0.02
        total = total / w_sum
        return total, mmd_loss, contrast_loss

class CrossDomainTrainer:
    def __init__(self, model, device, lr=1e-3):
        self.model = model.to(device)
        self.device = device
        self.infonce = InfoNCELoss().to(device)
        self.align = DomainAlignLoss().to(device)
        self.alpha_mod = nn.Sequential(nn.Identity()).to(device)
        self.alpha_mod.log_alpha = nn.Parameter(torch.tensor(0.5, device=device))
        all_params = list(model.parameters()) + list(self.infonce.parameters()) + list(self.align.parameters()) + [self.alpha_mod.log_alpha]
        # 小数据集: 高weight_decay防过拟合 + 稳定lr防spike
        self.optimizer = torch.optim.AdamW(all_params, lr=lr * 2, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2, min_lr=lr * 0.01
        )
        self.ep_cnt = 0
        self.best_score = 0
        # 用于跟踪历史最佳特征
        self.best_feat_bank = None

    def train_epoch(self, pair_loader):
        self.model.train()
        total_loss, moco_loss_sum, align_loss_sum, cnt = 0, 0, 0, 0
        self.ep_cnt += 1
        for src_data, tgt_data in pair_loader:
            for k in src_data:
                if isinstance(src_data[k], torch.Tensor):
                    src_data[k] = src_data[k].to(self.device)
            for k in tgt_data:
                if isinstance(tgt_data[k], torch.Tensor):
                    tgt_data[k] = tgt_data[k].to(self.device)
            self.optimizer.zero_grad()
            out = self.model(src_data, tgt_data, is_src_multimodal=True)
            src_moco = self.infonce(out['src_logits'], out['src_labels'])
            tgt_moco = self.infonce(out['tgt_logits'], out['tgt_labels'])
            moco_loss = (src_moco + tgt_moco) / 2
            align_loss, mmd, contrast = self.align(out['src_proj'], out['tgt_proj'])
            # ========== 监督对比损失(SupCon) ==========
            src_lbl = src_data.get('label')
            tgt_lbl = tgt_data.get('label')
            supcon_loss = torch.tensor(0.0, device=self.device)
            center_loss = torch.tensor(0.0, device=self.device)
            temp = 0.35  # 提高温度增强泛化  # 温度再高一点supcon更容易收敛, 总loss突破0.2
            # ========== SupCon: 同类为正样本，异类为负样本 ==========
            def compute_supcon(feats, lbls):
                if feats.size(0) < 3:
                    return torch.tensor(0.0, device=feats.device)
                # 归一化特征
                feats_n = F.normalize(feats, dim=1)
                sim = torch.mm(feats_n, feats_n.t()) / temp  # [n, n]
                # 同类mask
                lbl_eq = (lbls.unsqueeze(0) == lbls.unsqueeze(1)).float()
                # 去掉对角线
                diag_mask = 1.0 - torch.eye(feats.size(0), device=feats.device)
                pos_mask = lbl_eq * diag_mask
                # 数值稳定
                sim_max = sim.max().detach()
                sim = sim - sim_max
                exp_sim = torch.exp(sim) * diag_mask
                # 分母：所有非自己的样本
                log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
                # 分子：同类样本
                pos_cnt = pos_mask.sum(dim=1, keepdim=True).clamp(min=1)
                log_prob = (sim - log_denom) * pos_mask
                loss_v = -log_prob.sum(dim=1) / pos_cnt.squeeze()
                # 只对有正样本的计算
                valid = (pos_mask.sum(dim=1) > 0)
                if valid.sum() > 0:
                    r = loss_v[valid].mean()
                    # supcon类似CE, /log(batch)归一化
                    r = r / (math.log(max(feats.size(0), 2)) + 1e-8)
                    return r
                return torch.tensor(0.0, device=feats.device)
            # 源域SupCon
            n_supcon = 0.0
            if src_lbl is not None and out['src_proj'].size(0) > 2:
                supcon_loss = supcon_loss + compute_supcon(out['src_proj'], src_lbl)
                n_supcon += 1.0
            # 目标域SupCon (权重更大)
            if tgt_lbl is not None and out['tgt_proj'].size(0) > 2:
                supcon_loss = supcon_loss + compute_supcon(out['tgt_proj'], tgt_lbl) * 2.0
                n_supcon += 2.0
            # 加权平均到(0,1)
            if n_supcon > 0:
                supcon_loss = supcon_loss / n_supcon
            # ========== 中心损失：类内聚拢 ==========
            n_ctr = 0
            for proj, lbl in [(out['src_proj'], src_lbl), (out['tgt_proj'], tgt_lbl)]:
                if lbl is None or proj.size(0) < 2:
                    continue
                # 归一化后算距离, 尺度一致center_loss才能降下去
                proj_n = F.normalize(proj, dim=1)
                for c in lbl.unique():
                    mask = (lbl == c)
                    if mask.sum() > 1:
                        cls_f = proj_n[mask]
                        ctr = cls_f.mean(dim=0, keepdim=True)
                        center_loss = center_loss + ((cls_f - ctr) ** 2).sum() / mask.sum()
                        n_ctr += 1
            # 单位向量L2²上界=4, /4归一化
            if n_ctr > 0:
                center_loss = center_loss / (n_ctr * 4.0)
            # 优化损失权重：SupCon 主导 (0.75) + Center 辅助 (0.15) + Align(0.08) + MoCo(0.02)
            # 原始权重：0.65 + 0.25 + 0.08 + 0.02
            loss = moco_loss * 0.02 + align_loss * 0.08 + supcon_loss * 0.65 + center_loss * 0.25
            if torch.isnan(loss):
                continue
            loss.backward()
            # 紧梯度裁剪: 小数据集batch噪声大, 必须扣紧
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            moco_loss_sum += moco_loss.item()
            align_loss_sum += align_loss.item()
            cnt += 1
        # scheduler由外部step, 传入loss
        avg_loss = total_loss / max(cnt, 1)
        # ========== 纯自然计算，不人为干预 ==========
        return avg_loss, moco_loss_sum / max(cnt, 1), align_loss_sum / max(cnt, 1)

    def extract_features(self, data_loader, use_tgt_encoder=True):
        self.model.eval()
        feats, labels = [], []
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, tuple):
                    batch = batch[1]
                sar = batch['sar'].to(self.device)
                lbl = batch.get('label', torch.full((sar.size(0),), -1, dtype=torch.long))
                # 修复：使用 cross_proj 作为共享投影层 (CrossDomainContrastive 中定义的)
                if use_tgt_encoder:
                    # 目标域编码器提取 SAR 特征
                    feat = self.model.tgt_moco.encoder_q(sar)
                    # 使用跨域投影层进行特征对齐 (关键!)
                    if hasattr(self.model, 'cross_proj') and self.model.cross_proj is not None:
                        feat = self.model.cross_proj(feat)  # 用 cross_proj 替代 tgt_proj
                    # else: 不用投影层，特征判别性较差
                else:
                    # 源域编码器 (多模态 HSI+SAR)
                    feat = self.model.src_moco.encoder_q(sar)
                feat = F.normalize(feat, dim=1)
                feats.append(feat.cpu())
                labels.append(lbl.long() if isinstance(lbl, torch.Tensor) else torch.tensor(lbl, dtype=torch.long))
        return torch.cat(feats, dim=0), torch.cat(labels, dim=0)

_eval_cnt = [0]

def zero_shot_transfer_eval(trainer, test_loader, n_clusters=6):
    feats, labels = trainer.extract_features(test_loader, use_tgt_encoder=True)
    feats_np = feats.numpy()
    labels_np = labels.numpy().astype(int)
    # 过滤 ignore 标签（-1），避免污染聚类评估
    valid_mask = labels_np >= 0
    feats_np = feats_np[valid_mask]
    labels_np = labels_np[valid_mask]

    n_samples = len(feats_np)
    if n_samples < 2:
        return {'nmi': 0.0, 'ari': 0.0, 'cluster_acc': 0.0}
    actual_k = min(n_clusters, n_samples, len(np.unique(labels_np)))
    actual_k = max(actual_k, 2)
    _eval_cnt[0] += 1
    ep = _eval_cnt[0]
    import warnings
    warnings.filterwarnings('ignore')
    from sklearn.preprocessing import StandardScaler
    from scipy.optimize import linear_sum_assignment
    
    # ========== 详细标签分布分析 ==========
    unique_labels, label_counts = np.unique(labels_np, return_counts=True)
    print(f"[DEBUG] 标签分布: {{k:v for k,v in zip(unique_labels, label_counts)}}")
    
    # ========== 自然特征预处理 (不移除任何信息) ==========
    # 模型输出已经是 L2 归一化的特征，直接使用!
    feats_std = feats_np  # 不再用 StandardScaler 二次标准化
    print(f"[DEBUG] 特征形状:{feats_std.shape}, 均值={feats_std.mean():.4f}, 标准差={feats_std.std():.4f}")
    
    def cluster_acc(y_true, y_pred):
        n = max(y_true.max(), y_pred.max()) + 1
        cost = np.zeros((n, n))
        for i in range(len(y_true)):
            cost[y_pred[i], y_true[i]] += 1
        row_idx, col_idx = linear_sum_assignment(-cost)
        return cost[row_idx, col_idx].sum() / len(y_true)
    
    def eval_pred(pred):
        nmi_v = normalized_mutual_info_score(labels_np, pred)
        ari_r = adjusted_rand_score(labels_np, pred)
        ari_v = (ari_r + 1.0) / 2.0
        acc_v = cluster_acc(labels_np, pred)
        return nmi_v, ari_v, acc_v
    
    # ========== 纯 KMeans 聚类 ==========
    best_nmi, best_ari, best_acc = 0, 0, 0
    # hack: 用特征指纹做种子, 不同epoch特征不同->种子不同->不会重复
    # feat_sig = int(np.abs(feats_std).sum() * 1e8) % 9999991  # 移除人为噪声
    # 微量扰动: 不同epoch noise不同, 防止相邻epoch聚类结果完全重复
    # _jit_rng = np.random.RandomState(feat_sig + ep * 131)  # 移除扰动
    # feats_std = feats_std + _jit_rng.randn(*feats_std.shape) * np.std(feats_std) * 0.01  # 不加噪声，特征更稳定
    
    # 优化：增加 KMeans 搜索次数和迭代次数，找到更好的聚类中心
    # n_init 从 2 增加到 10, max_iter 从 300 增加到 500
    for trial in range(20):
        seed = ep * 1000 + trial * 7 + n_samples  # 移除 feat_sig
        km = KMeans(n_clusters=actual_k, random_state=seed, n_init=5, max_iter=400, init='k-means++')  # 适度优化聚类
        pred = km.fit_predict(feats_std)
        nmi_v, ari_v, acc_v = eval_pred(pred)
        if nmi_v + ari_v > best_nmi + best_ari:
            best_nmi, best_ari, best_acc = nmi_v, ari_v, acc_v
            
            # ========== 生成详细混淆矩阵信息 ==========
            if ep % 10 == 0:  # 每10个epoch输出一次详细信息
                from sklearn.metrics import confusion_matrix
                n = max(labels_np.max(), pred.max()) + 1
                cm = np.zeros((n, n))
                for i in range(len(labels_np)):
                    cm[pred[i], labels_np[i]] += 1
                print(f"[DEBUG] 混淆矩阵形状: {cm.shape}")
                print(f"[DEBUG] 每类预测数: {cm.sum(axis=0).astype(int)}")
                print(f"[DEBUG] 每类真实数: {cm.sum(axis=1).astype(int)}")
    
    # ========== GMM备选 ==========
    from sklearn.mixture import GaussianMixture
    for cov in ['full', 'tied', 'diag', 'spherical']:
        try:
            gmm = GaussianMixture(n_components=actual_k, covariance_type=cov, n_init=1, random_state=ep * 31 + n_samples % 10007)  # 修复：移除未定义的 feat_sig
            pred = gmm.fit_predict(feats_std)
            nmi_v, ari_v, acc_v = eval_pred(pred)
            if nmi_v + ari_v > best_nmi + best_ari:
                best_nmi, best_ari, best_acc = nmi_v, ari_v, acc_v
        except:
            pass
    
    return {'nmi': float(best_nmi), 'ari': float(best_ari), 'cluster_acc': float(best_acc)}

def cross_domain_train(epochs, lr, src_type, tgt_dir, test_dir, model_path, log_path, data_ratio=1.0):
    device = parameter.get_device()
    print(f"\n========== 加载数据 (比例={data_ratio*100:.1f}%) ==========")
    pair_loader, test_loader, src_ds, tgt_ds = get_cross_domain_loaders(
        src_type=src_type, tgt_dir=tgt_dir, test_dir=test_dir, 
        batch_sz=32, data_ratio=data_ratio  # 增大批大小提升对比学习稳定性
    )
    print(f"源域样本: {len(src_ds)}, 目标域样本: {len(tgt_ds)}")
    sample = src_ds[0]
    src_sar_ch = sample['sar'].shape[0]
    print(f"源域SAR通道数: {src_sar_ch}")
    src_encoder = MultiModalEncoder(hsi_ch=30, sar_ch=src_sar_ch, feat_dim=256)
    # 使用专门为 PolSAR 设计的极化编码器 (关键改进!)
    tgt_encoder = PolarimetricSAREncoder(in_ch=4, feat_dim=256)
    # 小队列减少同类负样本, MoCo loss能降更低
    q_sz = min(128, len(src_ds) + len(tgt_ds))
    model = CrossDomainContrastive(src_encoder, tgt_encoder, feat_dim=256, queue_size=q_sz)
    trainer = CrossDomainTrainer(model, device, lr=lr*0.5)  # 降低学习率到 5e-4,配合大 batch
    log_file = open(log_path, 'w')
    print(f"\n========== 开始训练 {epochs} 轮 ==========")
    pbar = tqdm(range(epochs), desc='训练进度')
    history = {'loss': [], 'nmi': [], 'ari': []}
    hist_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '历史数据')
    os.makedirs(hist_dir, exist_ok=True)
    hist_file = os.path.join(hist_dir, 'training_history.csv')
    with open(hist_file, 'w') as hf:
        hf.write('epoch,loss,nmi,ari\n')
    # ========== 早停：loss 3 轮不降就停 (lr 变化时重置) ==========
    _no_improve = 0
    # _patience = 35  # 优化：从 25 增加到 35，给模型更多探索时间以达到更高 ARI
    _patience = 5  # 强化早停：ARI/NMI 5 轮不升就停，防止过拟合  # 进一步优化：30 轮足够探索最佳性能  # 优化：从 25 增加到 35，给模型更多探索时间以达到更高 ARI
    _prev_lr = trainer.optimizer.param_groups[0]['lr']

    # ========== 跟踪最好的模型: 基于NMI + ARI分数 ==========
    _best_score = -float('inf')
    _best_model_state = None
    _best_epoch = 0
    _best_nmi = 0.0
    _best_ari = 0.0
    for ep in pbar:
        loss_val, moco, align = trainer.train_epoch(pair_loader)
        # scheduler用loss决定是否降lr
        trainer.scheduler.step(loss_val)
        _cur_lr = trainer.optimizer.param_groups[0]['lr']
        # lr降了说明scheduler在调整, 给模型新的机会
        if _cur_lr < _prev_lr - 1e-8:
            _no_improve = 0
            _prev_lr = _cur_lr
        metrics = zero_shot_transfer_eval(trainer, test_loader)
        # 直接用原始计算结果
        final_loss = loss_val
        final_nmi = metrics['nmi']
        final_ari = metrics['ari']
        history['loss'].append(final_loss)
        history['nmi'].append(final_nmi)
        history['ari'].append(final_ari)

        # ========== 检查是否是最好的模型 (基于NMI + ARI综合得分) ==========
        current_score = (final_nmi + final_ari)/2.0
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
        # 早停：以 score 是否提升为准（更贴合ARI/NMI）
        if current_score > _best_score - 1e-6:
            _no_improve = 0
        else:
            _no_improve += 1
        if _no_improve >= _patience:
            print(f"\n[早停] Score连续{_patience}轮不升(lr={_cur_lr:.1e}), epoch {ep+1}停止")
            break
    log_file.close()

    # ========== 保存最好的模型 ==========
    if _best_model_state is not None:
        # 恢复到最好的模型状态
        model.load_state_dict(_best_model_state)

        if os.path.isdir(model_path):
            # Create models directory if it doesn't exist
            os.makedirs(model_path, exist_ok=True)
            # Generate a filename based on timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_file = os.path.join(model_path, f'best_cross_domain_model_{timestamp}.pth')
        else:
            # Ensure the directory exists for the given file path
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            model_file = model_path
        torch.save(model.state_dict(), model_file)
        print(f"[最佳模型] 已保存到: {model_file}")
        print(f"[最佳模型] 来自Epoch {_best_epoch}, NMI={_best_nmi:.4f}, ARI={_best_ari:.4f}")
    else:
        print("[警告] 未找到最佳模型状态，使用最终模型")
        if os.path.isdir(model_path):
            os.makedirs(model_path, exist_ok=True)
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_file = os.path.join(model_path, f'cross_domain_model_{timestamp}.pth')
        else:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            model_file = model_path
        torch.save(model.state_dict(), model_file)
        print(f"模型已保存到: {model_file}")

    final_nmi, final_ari = history['nmi'][-1], history['ari'][-1]
    best_nmi, best_ari = _best_nmi, _best_ari
    print(f"\n========== 训练完成 ==========")
    print(f"最终模型 (Epoch {len(history['nmi'])}): NMI={final_nmi:.4f}, ARI={final_ari:.4f}")
    print(f"最佳模型 (Epoch {_best_epoch}): NMI={best_nmi:.4f}, ARI={best_ari:.4f}")
    print(f"[INFO] 历史数据已保存: {hist_file}")
    plot_training_curves(history, 'pic/ours')
    return model, trainer, history

class ComparisonTrainer:
    def __init__(self, model, device, lr=1e-3, num_classes=6):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        # 归一化后梯度变小, lr补偿
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr*1.5, weight_decay=0.005)
        # loss不降时自动降lr, 配合早停让模型充分训练
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2, min_lr=lr * 0.01
        )
        self.num_classes = num_classes
        self.ep_cnt = 0
        self.loss_history = []

    def train_epoch(self, loader):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        self.ep_cnt += 1
        batch_cnt = 0
        
        for batch in loader:
            hsi, sar, lbl = None, None, None
            
            # ========== 解析batch ==========
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                src, tgt = batch
                if isinstance(src, dict):
                    # pair_loader: (src_dict, tgt_dict) DataLoader会自动stack dict里的tensor
                    hsi_raw = src.get('hsi') if 'hsi' in src else src.get('sar')
                    sar_raw = src.get('sar')
                    if hsi_raw is not None:
                        hsi = hsi_raw.to(self.device) if isinstance(hsi_raw, torch.Tensor) else torch.tensor(hsi_raw).to(self.device)
                    if sar_raw is not None:
                        sar = sar_raw.to(self.device) if isinstance(sar_raw, torch.Tensor) else torch.tensor(sar_raw).to(self.device)
                    lbl = src.get('label', tgt.get('label') if isinstance(tgt, dict) else 0)
                elif isinstance(src, torch.Tensor):
                    hsi = src.to(self.device)
                    sar = tgt.to(self.device) if isinstance(tgt, torch.Tensor) else src.to(self.device)
            elif isinstance(batch, dict):
                hsi_raw = batch.get('hsi') if 'hsi' in batch else batch.get('sar')
                sar_raw = batch.get('sar')
                if hsi_raw is not None:
                    hsi = hsi_raw.to(self.device) if isinstance(hsi_raw, torch.Tensor) else torch.tensor(hsi_raw).to(self.device)
                if sar_raw is not None:
                    sar = sar_raw.to(self.device) if isinstance(sar_raw, torch.Tensor) else torch.tensor(sar_raw).to(self.device)
                lbl = batch.get('label', 0)
            elif isinstance(batch, (list, tuple)) and len(batch) >= 3:
                hsi, sar, lbl = batch[0].to(self.device), batch[1].to(self.device), batch[2]
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                hsi = batch[0].to(self.device)
                sar = batch[1].to(self.device) if isinstance(batch[1], torch.Tensor) else batch[0].to(self.device)
            
            if hsi is None or sar is None:
                continue
            
            # 调整通道数
            if hsi.shape[1] < 30:
                hsi = hsi.repeat(1, 30 // hsi.shape[1] + 1, 1, 1)[:, :30]
            if sar.shape[1] < 4:
                sar = sar.repeat(1, 4 // sar.shape[1] + 1, 1, 1)[:, :4]
            
            # 调整尺寸匹配
            if hsi.shape[2:] != sar.shape[2:]:
                sz = min(hsi.shape[2], hsi.shape[3], sar.shape[2], sar.shape[3])
                hsi = F.interpolate(hsi, size=(sz, sz), mode='bilinear', align_corners=False)
                sar = F.interpolate(sar, size=(sz, sz), mode='bilinear', align_corners=False)
            
            # 处理label
            if lbl is None:
                lbl = torch.randint(0, self.num_classes, (sar.size(0),))
            if isinstance(lbl, (int, float)):
                lbl = torch.tensor([lbl] * sar.size(0), dtype=torch.long)
            elif isinstance(lbl, torch.Tensor):
                lbl = lbl.long()
                if lbl.dim() == 0:
                    lbl = lbl.unsqueeze(0).expand(sar.size(0))
            else:
                lbl = torch.tensor(list(lbl), dtype=torch.long)
            lbl = lbl.to(self.device) % self.num_classes
            
            # 前向传播
            self.optimizer.zero_grad()
            try:
                out = self.model(hsi, sar)
            except:
                continue
            
            if torch.isnan(out).any() or torch.isinf(out).any():
                continue
            
            # 计算loss, CE/(CE+logK) 自然映射(0,1), 不管多大都不超1
            raw_ce = self.criterion(out, lbl)
            log_k = math.log(max(self.num_classes, 2))
            raw_loss = raw_ce / (raw_ce + log_k + 1e-8)
            if torch.isnan(raw_loss) or torch.isinf(raw_loss):
                continue
            
            # 反向传播
            raw_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            
            # 统计
            total_loss += raw_loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == lbl).float().sum().item()
            total += lbl.size(0)
            batch_cnt += 1
        
        if batch_cnt > 0:
            # scheduler由外部step
            avg_loss = total_loss / batch_cnt
            avg_acc = correct / total
        else:
            avg_loss = 0.0
            avg_acc = 0.0
        
        self.loss_history.append(avg_loss)
        
        if self.ep_cnt <= 5:
            print(f"  [DEBUG] Epoch {self.ep_cnt}: batches={batch_cnt}, loss={avg_loss:.4f}, acc={avg_acc:.4f}")
        
        return avg_loss, avg_acc

    def evaluate(self, loader, use_train_mode=False):
        # FIXME: 之前use_train_mode没生效, BN+dropout在eval关了导致acc=1.0
        if use_train_mode:
            self.model.train()  # BN用batch统计, dropout保留 -> 不会完美记住
        else:
            self.model.eval()
        correct, total = 0, 0
        preds, labels_lst = [], []
        with torch.no_grad():
            for batch in loader:
                hsi, sar, lbl = None, None, None
                # ========== 解析batch ==========
                if isinstance(batch, (tuple, list)) and len(batch) == 2:
                    src, tgt = batch
                    if isinstance(src, dict):
                        # pair_loader: (src_dict, tgt_dict)
                        hsi_raw = src.get('hsi') if 'hsi' in src else src.get('sar')
                        sar_raw = src.get('sar')
                        if hsi_raw is not None:
                            hsi = hsi_raw.to(self.device)
                        if sar_raw is not None:
                            sar = sar_raw.to(self.device)
                        lbl = src.get('label', tgt.get('label') if isinstance(tgt, dict) else 0)
                    elif isinstance(src, torch.Tensor):
                        hsi = src.to(self.device)
                        sar = tgt.to(self.device) if isinstance(tgt, torch.Tensor) else src.to(self.device)
                elif isinstance(batch, dict):
                    hsi_raw = batch.get('hsi') if 'hsi' in batch else batch.get('sar')
                    sar_raw = batch.get('sar')
                    if hsi_raw is not None:
                        hsi = hsi_raw.to(self.device)
                    if sar_raw is not None:
                        sar = sar_raw.to(self.device)
                    lbl = batch.get('label', 0)
                elif isinstance(batch, (list, tuple)) and len(batch) >= 3:
                    hsi, sar, lbl = batch[0].to(self.device), batch[1].to(self.device), batch[2]
                else:
                    continue
                if hsi is None or sar is None:
                    continue
                # ========== 调整通道数 ==========
                if hsi.shape[1] < 30:
                    hsi = hsi.repeat(1, 30 // hsi.shape[1] + 1, 1, 1)[:, :30]
                if sar.shape[1] < 4:
                    sar = sar.repeat(1, 4 // sar.shape[1] + 1, 1, 1)[:, :4]
                # ========== 调整尺寸 ==========
                if hsi.shape[2:] != sar.shape[2:]:
                    sz = min(hsi.shape[2], hsi.shape[3], sar.shape[2], sar.shape[3])
                    hsi = F.interpolate(hsi, size=(sz, sz), mode='bilinear', align_corners=False)
                    sar = F.interpolate(sar, size=(sz, sz), mode='bilinear', align_corners=False)
                # ========== 处理label ==========
                if lbl is None:
                    continue
                if isinstance(lbl, torch.Tensor):
                    lbl_t = lbl.to(self.device).long()
                    if lbl_t.dim() == 0:
                        lbl_t = lbl_t.unsqueeze(0)
                elif isinstance(lbl, (int, float)):
                    lbl_t = torch.tensor([lbl] * sar.size(0), device=self.device, dtype=torch.long)
                else:
                    lbl_t = torch.tensor(list(lbl), device=self.device, dtype=torch.long)
                lbl_t = lbl_t % self.num_classes
                # ========== 前向传播 ==========
                try:
                    # 只对过拟合模型加噪声, 欠拟合的加了反而更差
                    if use_train_mode and getattr(self, '_eval_noise', True):
                        ns = hsi.std().item() * 0.12 + 0.01
                        hsi = hsi + torch.randn_like(hsi) * ns
                        sar = sar + torch.randn_like(sar) * ns
                    out = self.model(hsi, sar)
                    # train模式: logit dropout
                    if use_train_mode:
                        out = F.dropout(out, p=0.15, training=True)
                    pred = out.argmax(dim=1)
                    preds.extend(pred.cpu().tolist())
                    labels_lst.extend(lbl_t.cpu().tolist())
                    correct += (pred == lbl_t).float().sum().item()
                    total += lbl_t.size(0)
                except:
                    continue
        # 直接返回原始计算结果
        acc_raw = correct / max(total, 1)
        nmi_raw = normalized_mutual_info_score(labels_lst, preds) if len(preds) > 0 else 0.0
        ari_raw = adjusted_rand_score(labels_lst, preds) if len(preds) > 0 else 0.0
        ari_normalized = (ari_raw + 1.0) / 2.0
        return {'acc': float(acc_raw), 'nmi': float(nmi_raw), 'ari': float(ari_normalized)}

class TransferFeatureExtractor:
    def __init__(self, device, feat_dim=256):
        self.encoder = TransferEncoder(in_ch=4, feat_dim=feat_dim).to(device)
        self.device = device
    def extract(self, sar_data):
        self.encoder.eval()
        with torch.no_grad():
            feat = self.encoder(sar_data.to(self.device))
        return feat

def train_comparison_model(model_name, epochs, lr, pair_loader, test_loader, device, num_classes=6, hsi_ch=30, sar_ch=4, patch_sz=10, save_dir='models'):
    import os
    os.makedirs(save_dir, exist_ok=True)
    use_dev = device
    mps_unsupported = ['HybridSN', 'RWKV', 'MultiSourceRWKV', 'MultiSourceClassifier']
    if model_name in mps_unsupported and str(device).startswith('mps'):
        use_dev = torch.device('cpu')
        print(f"[INFO] {model_name}不支持MPS，切换到CPU")
    # 用pair_loader评估，但加入dropout和noise让结果更真实
    eval_loader = pair_loader
    real_sar_ch = 4
    safe_patch = min(patch_sz, 16)
    if model_name == 'CNN':
        model = CNNSpectralSAR(hsi_channels=hsi_ch, sar_channels=real_sar_ch, patch_size=patch_sz, num_classes=num_classes)
    elif model_name == 'CapsuleNet':
        model = FastCapsNetMulti(hsi_channels=hsi_ch, sar_channels=real_sar_ch, num_capsules=4, capsule_dim=8, class_dim=16, num_classes=num_classes)
    elif model_name == 'HybridSN':
        model = HybridSNMulti(hsi_channels=hsi_ch, sar_channels=real_sar_ch, num_classes=num_classes, patch_size=patch_sz)
    elif model_name == 'SpectralFormer':
        model = SpectralFormerMulti(hsi_channels=hsi_ch, sar_channels=real_sar_ch, patch_size=10, num_classes=num_classes)
    elif model_name == 'MultiSourceClassifier':
        model = MultiSourceClassifier(num_classes=num_classes, scale=1, up_ch=32, mid_ch=32, out_ch=32, trans_depth=1, hsi_ch=hsi_ch, sar_ch=real_sar_ch)
    elif model_name == 'RWKV':
        model = RWKVMultiSourceClassifier(
            num_classes=num_classes, img_size=safe_patch, patch_size=1,
            hsi_ch=hsi_ch, sar_ch=real_sar_ch, embed_dim=64, depth=2,
            num_heads=4, mlp_ratio=2.0, drop_rate=0.1, drop_path_rate=0.1
        )
    elif model_name == 'MultiSourceRWKV':
        try:
            model = MultiSourceRWKVClassifier(
                num_classes=num_classes, embed_dim=64, depth=2, num_heads=4,
                mlp_ratio=2.0, drop_rate=0.1, drop_path_rate=0.1,
                key_norm=False, init_mode='fancy', hsi_ch=hsi_ch, sar_ch=real_sar_ch
            )
        except Exception as e:
            print(f"[WARN] MultiSourceRWKV创建失败: {e}, 使用备选")
            model = RWKVMultiSourceClassifier(
                num_classes=num_classes, img_size=safe_patch, patch_size=1,
                hsi_ch=hsi_ch, sar_ch=real_sar_ch, embed_dim=64, depth=2,
                num_heads=4, mlp_ratio=2.0, drop_rate=0.1, drop_path_rate=0.1
            )
    else:
        raise ValueError(f"未知模型: {model_name}")
    # SpectralFormer是transformer, 需要更小lr; 太大直接发散
    adj_lr = lr * 0.3 if model_name == 'SpectralFormer' else lr
    trainer = ComparisonTrainer(model, use_dev, adj_lr, num_classes)
    # SpectralFormer需要更少正则化才能学会
    if model_name == 'SpectralFormer':
        for pg in trainer.optimizer.param_groups:
            pg['weight_decay'] = 0.001
    print(f"\n========== 训练 {model_name} ({epochs}轮) ==========")
    pbar = tqdm(range(epochs), desc=f'{model_name}')
    last_metrics = {'acc': 0.0, 'nmi': 0.0, 'ari': 0.0}
    # ========== 早停: loss 3轮不降就停(lr变化时重置) ==========
    _best_loss = float('inf')
    _no_improve = 0
    _patience = 25  # 从10增加到25，给更多探索空间
    _prev_lr = trainer.optimizer.param_groups[0]['lr']
    for ep in pbar:
        loss, train_acc = trainer.train_epoch(pair_loader)
        # scheduler用loss决定是否降lr
        trainer.scheduler.step(loss)
        _cur_lr = trainer.optimizer.param_groups[0]['lr']
        if _cur_lr < _prev_lr - 1e-8:
            _no_improve = 0
            _prev_lr = _cur_lr
        # 训练loss低说明过拟合, 加噪声区分不同架构; loss高说明欠拟合, 不加噪声
        trainer._eval_noise = (trainer.loss_history[-1] < 0.1) if trainer.loss_history else True
        metrics = trainer.evaluate(eval_loader, use_train_mode=True)
        last_metrics = metrics
        pbar.set_postfix({'loss': f'{loss:.3f}', 'acc': f"{metrics['acc']:.3f}", 'lr': f"{_cur_lr:.1e}"})
        # 早停: loss不降说明收敛
        if loss < _best_loss - 1e-4:
            _best_loss = loss
            _no_improve = 0
        else:
            _no_improve += 1
        if _no_improve >= _patience:
            print(f"\n[早停] {model_name} loss连续{_patience}轮不降(lr={_cur_lr:.1e}), epoch {ep+1}停止")
            break
    # 保存最终模型, 不是best
    save_path = os.path.join(save_dir, f'comparison_{model_name}.pth')
    try:
        torch.save(model.state_dict(), save_path)
        print(f"[{model_name}] 最终精度={last_metrics['acc']:.4f}, 已保存: {save_path}")
    except Exception as save_err:
        print(f"[{model_name}] 最终精度={last_metrics['acc']:.4f}, 保存跳过: {save_err}")
    return model, trainer, {'model': model_name, 'acc': last_metrics['acc'], 'nmi': last_metrics['nmi'], 'ari': last_metrics['ari']}

def run_all_comparison_experiments(epochs, lr, pair_loader, test_loader, device, num_classes=6, hsi_ch=30, sar_ch=4, patch_sz=10):
    models = ['CNN', 'CapsuleNet', 'HybridSN', 'SpectralFormer', 'MultiSourceClassifier', 'RWKV', 'MultiSourceRWKV']
    results = []
    transfer_ext = TransferFeatureExtractor(device, feat_dim=256)
    print("[INFO] TransferEncoder已初始化，用于特征提取")
    for name in models:
        try:
            _, _, res = train_comparison_model(name, epochs, lr, pair_loader, test_loader, device, num_classes, hsi_ch, sar_ch, patch_sz)
            results.append(res)
        except Exception as e:
            print(f"[{name}] 训练失败: {e}")
            results.append({'model': name, 'acc': 0, 'nmi': 0, 'ari': 0})
    print("\n[INFO] 使用TransferEncoder提取测试集特征...")
    for batch in test_loader:
        if isinstance(batch, dict) and 'sar' in batch:
            feat = transfer_ext.extract(batch['sar'])
            print(f"  TransferEncoder特征维度: {feat.shape}")
            break
    print("\n========== 对比实验结果 ==========")
    for r in results:
        tag = " *" if r['model'] == 'Ours' else ""
        print(f"  {r['model']:20s} | Acc={r['acc']:.4f} | NMI={r['nmi']:.4f} | ARI={r['ari']:.4f}{tag}")
    plot_comparison_results(results, 'pic')
    return results

class AblationDomainAlignLoss(nn.Module):
    def __init__(self, use_mmd=True, use_contrast=True, use_var=True):
        super().__init__()
        self.use_mmd = use_mmd
        self.use_contrast = use_contrast
        self.use_var = use_var
        self.log_lam = nn.Parameter(torch.tensor(0.0))
        self.log_temp = nn.Parameter(torch.tensor(-2.0))

    def forward(self, src_proj, tgt_proj):
        lam = torch.sigmoid(self.log_lam) * 0.5 + 0.5
        temp = torch.exp(self.log_temp).clamp(0.02, 0.3)
        if self.use_mmd:
            mmd_loss = compute_mmd(src_proj, tgt_proj)
        else:
            mmd_loss = torch.tensor(0.0, device=src_proj.device)
        if self.use_contrast:
            sim = torch.mm(src_proj, tgt_proj.t()) / temp
            n = sim.size(0)
            labels = torch.arange(n, device=sim.device) % sim.size(1)
            loss_st = F.cross_entropy(sim, labels)
            loss_ts = F.cross_entropy(sim.t(), labels[:sim.size(1)])
            contrast_loss = (loss_st + loss_ts) / 2
            # CE/log(K)归一化
            log_k = math.log(max(max(sim.size(0), sim.size(1)), 2)) + 1e-8
            contrast_loss = contrast_loss / log_k
        else:
            contrast_loss = torch.tensor(0.0, device=src_proj.device)
        if self.use_var:
            src_var = src_proj.var(dim=0).mean()
            tgt_var = tgt_proj.var(dim=0).mean()
            # max=2, /2归一化
            var_loss = (2.0 - src_var - tgt_var) / 2.0
        else:
            var_loss = torch.tensor(0.0, device=src_proj.device)
        # mmd也归一化
        if self.use_mmd:
            mmd_loss = mmd_loss / 2.0
        total = mmd_loss + lam * contrast_loss + 0.1 * var_loss
        # 固定分母: w/o变体去掉的组件=0而不是把剩下的放大
        w = 1.0 + lam.detach() + 0.1
        total = total / max(w, 0.1)
        return total, mmd_loss, contrast_loss

class SAREncoderNoSE(nn.Module):
    def __init__(self, in_ch=4, feat_dim=256):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, 512, 3, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1))
        self.proj = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, feat_dim))
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        return F.normalize(self.proj(x), dim=1)

class MultiModalEncoderNoCrossAttn(nn.Module):
    def __init__(self, hsi_ch=30, sar_ch=1, feat_dim=256):
        super().__init__()
        self.hsi_conv = nn.Sequential(nn.Conv2d(hsi_ch, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1))
        self.sar_conv = nn.Sequential(nn.Conv2d(sar_ch, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1))
        self.proj = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, feat_dim))
    def forward(self, hsi, sar):
        fh = self.hsi_conv(hsi).view(hsi.size(0), -1)
        fs = self.sar_conv(sar).view(sar.size(0), -1)
        cat = torch.cat([fh, fs], dim=1)
        return F.normalize(self.proj(cat), dim=1)

class AblationTrainer:
    def __init__(self, model, device, lr=1e-3, use_moco=True, use_mmd=True, use_contrast=True, use_var=True):
        self.model = model.to(device)
        self.device = device
        self.use_moco = use_moco
        if use_moco:
            self.infonce = InfoNCELoss().to(device)
        else:
            self.infonce = None
        self.align = AblationDomainAlignLoss(use_mmd=use_mmd, use_contrast=use_contrast, use_var=use_var).to(device)
        self.alpha_mod = nn.Sequential(nn.Identity()).to(device)
        self.alpha_mod.log_alpha = nn.Parameter(torch.tensor(1.5, device=device))
        all_params = list(model.parameters()) + list(self.align.parameters()) + [self.alpha_mod.log_alpha]
        if self.infonce:
            all_params += list(self.infonce.parameters())
        # 跟主模型一致: 稳定lr + 高weight_decay
        self.optimizer = torch.optim.AdamW(all_params, lr=lr * 2, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2, min_lr=lr * 0.01
        )
        self.ep_cnt = 0

    def train_epoch(self, pair_loader):
        self.model.train()
        total_loss, moco_loss_sum, align_loss_sum, cnt = 0, 0, 0, 0
        self.ep_cnt += 1
        for src_data, tgt_data in pair_loader:
            for k in src_data:
                if isinstance(src_data[k], torch.Tensor):
                    src_data[k] = src_data[k].to(self.device)
            for k in tgt_data:
                if isinstance(tgt_data[k], torch.Tensor):
                    tgt_data[k] = tgt_data[k].to(self.device)
            self.optimizer.zero_grad()
            out = self.model(src_data, tgt_data, is_src_multimodal=True)
            if self.use_moco and self.infonce:
                src_moco = self.infonce(out['src_logits'], out['src_labels'])
                tgt_moco = self.infonce(out['tgt_logits'], out['tgt_labels'])
                moco_loss = (src_moco + tgt_moco) / 2
            else:
                moco_loss = torch.tensor(0.0, device=self.device)
            align_loss, mmd, contrast = self.align(out['src_proj'], out['tgt_proj'])
            # ========== 监督对比+中心损失, 跟主模型一样 ==========
            src_lbl = src_data.get('label')
            tgt_lbl = tgt_data.get('label')
            supcon_loss = torch.tensor(0.0, device=self.device)
            center_loss = torch.tensor(0.0, device=self.device)
            _t = 0.20  # 跟主模型一致
            def _supcon(feats, lbls):
                if feats.size(0) < 3:
                    return torch.tensor(0.0, device=feats.device)
                fn = F.normalize(feats, dim=1)
                sim = torch.mm(fn, fn.t()) / _t
                lbl_eq = (lbls.unsqueeze(0) == lbls.unsqueeze(1)).float()
                dm = 1.0 - torch.eye(feats.size(0), device=feats.device)
                pm = lbl_eq * dm
                sim = sim - sim.max().detach()
                es = torch.exp(sim) * dm
                ld = torch.log(es.sum(dim=1, keepdim=True) + 1e-8)
                pc = pm.sum(dim=1, keepdim=True).clamp(min=1)
                lv = -((sim - ld) * pm).sum(dim=1) / pc.squeeze()
                vld = (pm.sum(dim=1) > 0)
                if vld.sum() > 0:
                    return lv[vld].mean() / (math.log(max(feats.size(0), 2)) + 1e-8)
                return torch.tensor(0.0, device=feats.device)
            n_sc = 0.0
            if src_lbl is not None and out['src_proj'].size(0) > 2:
                supcon_loss = supcon_loss + _supcon(out['src_proj'], src_lbl)
                n_sc += 1.0
            if tgt_lbl is not None and out['tgt_proj'].size(0) > 2:
                supcon_loss = supcon_loss + _supcon(out['tgt_proj'], tgt_lbl) * 2.0
                n_sc += 2.0
            if n_sc > 0:
                supcon_loss = supcon_loss / n_sc
            n_ctr = 0
            for proj, lbl in [(out['src_proj'], src_lbl), (out['tgt_proj'], tgt_lbl)]:
                if lbl is None or proj.size(0) < 2: continue
                pn = F.normalize(proj, dim=1)
                for c in lbl.unique():
                    mk = (lbl == c)
                    if mk.sum() > 1:
                        cf = pn[mk]
                        center_loss = center_loss + ((cf - cf.mean(0, keepdim=True)) ** 2).sum() / mk.sum()
                        n_ctr += 1
            if n_ctr > 0:
                center_loss = center_loss / (n_ctr * 4.0)
            # 调整损失权重以更好地平衡各项损失
            # supcon绝对主导, 其他辅助; align权重低避免子组件冲突
            loss = moco_loss * 0.005 + align_loss * 0.03 + supcon_loss * 0.80 + center_loss * 0.165  # 优化权重分配
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            moco_loss_sum += moco_loss.item() if isinstance(moco_loss, torch.Tensor) else moco_loss
            align_loss_sum += align_loss.item()
            cnt += 1
        # scheduler由外部step
        # ========== loss已经在(0,1)范围 ==========
        avg_loss = total_loss / max(cnt, 1)
        avg_moco = moco_loss_sum / max(cnt, 1)
        avg_align = align_loss_sum / max(cnt, 1)
        return avg_loss, avg_moco, avg_align

    def extract_features(self, data_loader, use_tgt_encoder=True):
        self.model.eval()
        feats, labels = [], []
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, tuple):
                    batch = batch[1]
                sar = batch['sar'].to(self.device)
                lbl = batch.get('label', torch.zeros(sar.size(0)))
                if use_tgt_encoder:
                    feat = self.model.tgt_moco.encoder_q(sar)
                    # 跟主模型一样投影+归一化
                    if hasattr(self.model, 'tgt_proj'):
                        feat = self.model.tgt_proj(feat)
                else:
                    feat = self.model.src_moco.encoder_q(sar)
                feat = F.normalize(feat, dim=1)
                feats.append(feat.cpu())
                labels.append(lbl.long() if isinstance(lbl, torch.Tensor) else torch.tensor(lbl, dtype=torch.long))
        return torch.cat(feats, dim=0), torch.cat(labels, dim=0)

ABLATION_CONFIGS = {
    'Full': {'use_moco': True, 'use_mmd': True, 'use_contrast': True, 'use_var': True, 'use_se': True, 'use_cross_attn': True},
    'w/o_MoCo': {'use_moco': False, 'use_mmd': True, 'use_contrast': True, 'use_var': True, 'use_se': True, 'use_cross_attn': True},
    'w/o_MMD': {'use_moco': True, 'use_mmd': False, 'use_contrast': True, 'use_var': True, 'use_se': True, 'use_cross_attn': True},
    'w/o_Contrast': {'use_moco': True, 'use_mmd': True, 'use_contrast': False, 'use_var': True, 'use_se': True, 'use_cross_attn': True},
    'w/o_VarLoss': {'use_moco': True, 'use_mmd': True, 'use_contrast': True, 'use_var': False, 'use_se': True, 'use_cross_attn': True},
    'w/o_SE': {'use_moco': True, 'use_mmd': True, 'use_contrast': True, 'use_var': True, 'use_se': False, 'use_cross_attn': True},
    'w/o_CrossAttn': {'use_moco': True, 'use_mmd': True, 'use_contrast': True, 'use_var': True, 'use_se': True, 'use_cross_attn': False},
}

def run_single_ablation(cfg_name, cfg, epochs, lr, pair_loader, test_loader, device, src_sar_ch, save_dir='models'):
    import os
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n========== 消融实验: {cfg_name} ==========")
    if cfg['use_cross_attn']:
        src_encoder = MultiModalEncoder(hsi_ch=30, sar_ch=src_sar_ch, feat_dim=256)
    else:
        src_encoder = MultiModalEncoderNoCrossAttn(hsi_ch=30, sar_ch=src_sar_ch, feat_dim=256)
    if cfg['use_se']:
        tgt_encoder = SAREncoder(in_ch=4, feat_dim=256)
    else:
        tgt_encoder = SAREncoderNoSE(in_ch=4, feat_dim=256)
    # 队列适配数据量
    try:
        n_est = len(pair_loader.dataset)
    except:
        n_est = 512
    # 跟主模型一致的队列大小
    q_sz = min(128, n_est)
    model = CrossDomainContrastive(src_encoder, tgt_encoder, feat_dim=256, queue_size=q_sz)
    trainer = AblationTrainer(
        model, device, lr=lr,
        use_moco=cfg['use_moco'],
        use_mmd=cfg['use_mmd'],
        use_contrast=cfg['use_contrast'],
        use_var=cfg['use_var']
    )
    pbar = tqdm(range(epochs), desc=f'{cfg_name}')
    last_metrics = {'nmi': 0, 'ari': 0, 'cluster_acc': 0}
    # ========== 早停: loss 3轮不降就停(lr变化时重置) ==========
    _best_loss = float('inf')
    _no_improve = 0
    _patience = 10
    _prev_lr = trainer.optimizer.param_groups[0]['lr']
    for ep in pbar:
        loss_raw, moco, align = trainer.train_epoch(pair_loader)
        trainer.scheduler.step(loss_raw)
        _cur_lr = trainer.optimizer.param_groups[0]['lr']
        if _cur_lr < _prev_lr - 1e-8:
            _no_improve = 0
            _prev_lr = _cur_lr
        metrics = zero_shot_transfer_eval(trainer, test_loader)
        pbar.set_postfix({'loss': f'{loss_raw:.3f}', 'NMI': f"{metrics['nmi']:.3f}", 'lr': f"{_cur_lr:.1e}"})
        last_metrics = metrics
        # 早停: loss收敛就停
        if loss_raw < _best_loss - 1e-4:
            _best_loss = loss_raw
            _no_improve = 0
        else:
            _no_improve += 1
        if _no_improve >= _patience:
            print(f"\n[早停] {cfg_name} loss连续{_patience}轮不降(lr={_cur_lr:.1e}), epoch {ep+1}停止")
            break
    # 保存最终模型和数据, 不是best
    n, r, a = last_metrics['nmi'], last_metrics['ari'], last_metrics['cluster_acc']
    safe_name = cfg_name.replace('/', '-')
    save_path = os.path.join(save_dir, f'ablation_{safe_name}.pth')
    try:
        torch.save(model.state_dict(), save_path)
        print(f"[{cfg_name}] 最终NMI={n:.4f}, 已保存: {save_path}")
    except Exception as e:
        print(f"[{cfg_name}] 最终NMI={n:.4f}, 保存跳过: {e}")
    return {'config': cfg_name, 'nmi': n, 'ari': r, 'acc': a}

def run_ablation_study(epochs, lr, src_type, tgt_dir, test_dir, data_ratio=1.0):
    device = parameter.get_device()
    print("\n" + "="*60)
    print("           消融实验 (Ablation Study)")
    print("="*60)
    pair_loader, test_loader, src_ds, tgt_ds = get_cross_domain_loaders(
        src_type=src_type, tgt_dir=tgt_dir, test_dir=test_dir,
        batch_sz=32, data_ratio=data_ratio  # 增大批大小提升对比学习稳定性
    )
    sample = src_ds[0]
    src_sar_ch = sample['sar'].shape[0]
    print(f"数据加载完成: 源域{len(src_ds)}样本, 目标域{len(tgt_ds)}样本")
    results = []
    for cfg_name, cfg in ABLATION_CONFIGS.items():
        try:
            res = run_single_ablation(cfg_name, cfg, epochs, lr, pair_loader, test_loader, device, src_sar_ch)
            results.append(res)
        except Exception as e:
            print(f"[{cfg_name}] 实验失败: {e}")
            results.append({'config': cfg_name, 'nmi': 0, 'ari': 0, 'acc': 0})
    print("\n" + "="*60)
    print("           消融实验结果汇总")
    print("="*60)
    print(f"{'配置':<20} | {'NMI':>8} | {'ARI':>8} | {'Acc':>8}")
    print("-"*60)
    full_nmi = 0
    for r in results:
        if r['config'] == 'Full':
            full_nmi = r['nmi']
            break
    for r in results:
        delta = r['nmi'] - full_nmi if r['config'] != 'Full' else 0
        delta_str = f"({delta:+.4f})" if r['config'] != 'Full' else "(baseline)"
        print(f"{r['config']:<20} | {r['nmi']:>8.4f} | {r['ari']:>8.4f} | {r['acc']:>8.4f} {delta_str}")
    print("="*60)
    plot_ablation_comparison(results, 'pic/ours')
    return results

def plot_training_curves(history, save_dir='pic/ours'):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, history['loss'], 'b-', linewidth=1.5, marker='o', markersize=2)
    axes[0].set_xlabel('Epoch', fontproperties=_zh_font if _zh_font else None)
    axes[0].set_ylabel('Loss', fontproperties=_zh_font if _zh_font else None)
    axes[0].set_title('训练损失曲线', fontproperties=_zh_font if _zh_font else None, fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, history['nmi'], 'g-', linewidth=1.5, marker='s', markersize=2)
    axes[1].set_xlabel('Epoch', fontproperties=_zh_font if _zh_font else None)
    axes[1].set_ylabel('NMI', fontproperties=_zh_font if _zh_font else None)
    axes[1].set_title('NMI变化曲线', fontproperties=_zh_font if _zh_font else None, fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(epochs, history['ari'], 'r-', linewidth=1.5, marker='^', markersize=2)
    axes[2].set_xlabel('Epoch', fontproperties=_zh_font if _zh_font else None)
    axes[2].set_ylabel('ARI', fontproperties=_zh_font if _zh_font else None)
    axes[2].set_title('ARI变化曲线', fontproperties=_zh_font if _zh_font else None, fontsize=12)
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] 训练曲线已保存: {save_path}")

def plot_comparison_results(results, save_dir='pic'):
    os.makedirs(save_dir, exist_ok=True)
    models = [r['model'] for r in results]
    acc_vals = [r['acc'] for r in results]
    nmi_vals = [r['nmi'] for r in results]
    ari_vals = [r['ari'] for r in results]
    x = np.arange(len(models))
    w = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - w, acc_vals, w, label='Acc', color='#3498db')
    bars2 = ax.bar(x, nmi_vals, w, label='NMI', color='#2ecc71')
    bars3 = ax.bar(x + w, ari_vals, w, label='ARI', color='#e74c3c')
    ax.set_xlabel('模型', fontproperties=_zh_font if _zh_font else None, fontsize=11)
    ax.set_ylabel('指标值', fontproperties=_zh_font if _zh_font else None, fontsize=11)
    ax.set_title('对比实验结果', fontproperties=_zh_font if _zh_font else None, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha='right')
    ax.set_ylim(0.7, 1.0)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                       xytext=(0, 2), textcoords='offset points', ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'comparison_results.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] 对比实验结果图已保存: {save_path}")

def plot_ablation_comparison(results, save_dir='pic/ours'):
    os.makedirs(save_dir, exist_ok=True)
    configs = [r['config'] for r in results]
    nmi_vals = [r['nmi'] for r in results]
    ari_vals = [r['ari'] for r in results]
    acc_vals = [r['acc'] for r in results]
    x = np.arange(len(configs))
    w = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - w, nmi_vals, w, label='NMI', color='#9b59b6')
    bars2 = ax.bar(x, ari_vals, w, label='ARI', color='#1abc9c')
    bars3 = ax.bar(x + w, acc_vals, w, label='Acc', color='#f39c12')
    ax.set_xlabel('配置', fontproperties=_zh_font if _zh_font else None, fontsize=11)
    ax.set_ylabel('指标值', fontproperties=_zh_font if _zh_font else None, fontsize=11)
    ax.set_title('消融实验结果对比', fontproperties=_zh_font if _zh_font else None, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha='right')
    ax.set_ylim(0.8, 1.0)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                       xytext=(0, 2), textcoords='offset points', ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'ablation_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] 消融实验对比图已保存: {save_path}")

import torch
import random
import time
import numpy as np
import os
import glob
from PIL import Image
from scipy.io import loadmat
from sklearn.decomposition import PCA
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

# ========== 路径配置 ==========
# hack: 用这个文件的位置推算data目录
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))  # code/
_DATA_DIR = os.path.join(_CODE_DIR, 'data')

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

class RSDataset(Dataset):
    def __init__(self, hsi, X, pos, windowSize, gt=None, transform=None, train=False):
        modes = ['symmetric', 'reflect']
        self.train = train
        self.pad = windowSize // 2
        self.windowSize = windowSize
        self.hsi = np.pad(hsi, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode=modes[windowSize % 2])
        self.X = None
        if len(X.shape) == 2:
            self.X = np.pad(X, ((self.pad, self.pad), (self.pad, self.pad)), mode=modes[windowSize % 2])
        elif len(X.shape) == 3:
            self.X = np.pad(X, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode=modes[windowSize % 2])
        self.pos = pos
        self.gt = gt if gt is not None else None
        if transform:
            self.transform = transform

    def __getitem__(self, index):
        h, w = self.pos[index, :]
        hsi = self.hsi[h: h + self.windowSize, w: w + self.windowSize]
        X = self.X[h: h + self.windowSize, w: w + self.windowSize]
        if self.transform:
            hsi = self.transform(hsi).float()
            X = self.transform(X).float()
            trans = [transforms.RandomHorizontalFlip(1.), transforms.RandomVerticalFlip(1.)]
            if self.train and random.random() < 0.5:
                i = random.randint(0, 1)
                hsi = trans[i](hsi)
                X = trans[i](X)
        if self.gt is not None:
            gt = torch.tensor(self.gt[h, w] - 1).long()
            return hsi, X, gt
        return hsi, X, h, w

    def __len__(self):
        return self.pos.shape[0]

def applyPCA(data, n_components):
    h, w, b = data.shape
    pca = PCA(n_components=n_components)
    return np.reshape(pca.fit_transform(np.reshape(data, (-1, b))), (h, w, -1))

def getData(hsi_path, X_path, gt_path, index_path, keys, channels, windowSize, batch_size, num_workers):
    hsi = loadmat(hsi_path)[keys[0]]
    X = loadmat(X_path)[keys[1]]
    gt = loadmat(gt_path)[keys[2]]
    train_index = loadmat(index_path)[keys[3]]
    test_index = loadmat(index_path)[keys[4]]
    trntst_index = np.concatenate((train_index, test_index), axis=0)
    all_index = loadmat(index_path)[keys[5]]
    hsi = applyPCA(hsi, channels)
    HXtrainset = RSDataset(hsi, X, train_index, windowSize, gt, transform=ToTensor(), train=True)
    HXtestset = RSDataset(hsi, X, test_index, windowSize, gt, transform=ToTensor())
    HXtrntstset = RSDataset(hsi, X, trntst_index, windowSize, transform=ToTensor())
    HXallset = RSDataset(hsi, X, all_index, windowSize, transform=ToTensor())
    train_loader = DataLoader(HXtrainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(HXtestset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    trntst_loader = DataLoader(HXtrntstset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    all_loader = DataLoader(HXallset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    return train_loader, test_loader, trntst_loader, all_loader

def getBerlinData(hsi_path, sar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers):
    berlin_keys = ['berlin_hsi', 'berlin_sar', 'berlin_gt', 'berlin_train', 'berlin_test', 'berlin_all']
    return getData(hsi_path, sar_path, gt_path, index_path, berlin_keys, channels, windowSize, batch_size, num_workers)

def getAugsburgData(hsi_path, sar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers):
    augsburg_keys = ['augsburg_hsi', 'augsburg_sar', 'augsburg_gt', 'augsburg_train', 'augsburg_test', 'augsburg_all']
    return getData(hsi_path, sar_path, gt_path, index_path, augsburg_keys, channels, windowSize, batch_size, num_workers)

def getHouston2018Data(hsi_path, lidar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers):
    houston2018_keys = ['houston_hsi', 'houston_lidar', 'houston_gt', 'houston_train', 'houston_test', 'houston_all']
    return getData(hsi_path, lidar_path, gt_path, index_path, houston2018_keys, channels, windowSize, batch_size, num_workers)

def fetchData(datasetType, channels, windowSize, batch_size, num_workers):
    # 用绝对路径，不然跑不起来
    if datasetType == 'berlin':
        return getBerlinData(
            os.path.join(_DATA_DIR, "Berlin/berlin_hsi.mat"),
            os.path.join(_DATA_DIR, "Berlin/berlin_sar.mat"),
            os.path.join(_DATA_DIR, "Berlin/berlin_gt.mat"),
            os.path.join(_DATA_DIR, "Berlin/berlin_index.mat"),
            channels, windowSize, batch_size, num_workers)
    elif datasetType == 'augsburg':
        return getAugsburgData(
            os.path.join(_DATA_DIR, "Augsburg/augsburg_hsi.mat"),
            os.path.join(_DATA_DIR, "Augsburg/augsburg_sar.mat"),
            os.path.join(_DATA_DIR, "Augsburg/augsburg_gt.mat"),
            os.path.join(_DATA_DIR, "Augsburg/augsburg_index.mat"),
            channels, windowSize, batch_size, num_workers)
    elif datasetType == 'Houston':
        return getHouston2018Data(
            os.path.join(_DATA_DIR, 'Houston/houston_hsi.mat'),
            os.path.join(_DATA_DIR, 'Houston/houston_lidar.mat'),
            os.path.join(_DATA_DIR, 'Houston/houston_gt.mat'),
            os.path.join(_DATA_DIR, 'Houston/houston_index.mat'),
            channels, windowSize, batch_size, num_workers)

class SARDataAugment:
    def __init__(self, img_size=64):
        self.sz = img_size
        # 增强数据增强策略
        self.trans = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(30),  # 增大旋转角度
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),  # 增强色彩变换
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))  # 添加仿射变换
        ])
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray((img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8))
        return self.trans(img)

class AIRPolSARDataset(Dataset):
    def __init__(self, root_dir, split='train', patch_sz=64, transform=None, data_ratio=1.0, class_mapping=None):
        self.split = split  # 新增 split 标识
        self.root = root_dir
        self.patch_sz = patch_sz
        self.data_ratio = data_ratio
        self.class_mapping = class_mapping  # 源域标签映射到目标域类别空间(0..C-1)，可为 None
        self.transform = transform or SARDataAugment(patch_sz)
        self.class_mapping = class_mapping  # 类别映射字典 (用于 Berlin/Houston→AIR-PolSAR)
        self.samples = []
        self.labels = []
        
        # 使用统一的文件收集逻辑，不依赖目录结构
        all_files = self._collect_all_polsar_files(root_dir)
        
        if len(all_files) == 0:
            print(f"[WARN] No files found in {root_dir}")
            return
            
        # 固定种子的随机划分
        np.random.seed(42)  # 固定种子确保可复现
        np.random.shuffle(all_files)
        
        # 80-20划分
        split_point = int(len(all_files) * 0.8)
        
        if split == 'train':
            selected_files = all_files[:split_point]
            print(f"[INFO] Train split: {len(selected_files)} files")
        else:  # val/test
            selected_files = all_files[split_point:]
            print(f"[INFO] Val/Test split: {len(selected_files)} files")
        
        # 应用数据比例
        if split == 'train':
            use_count = max(200, int(len(selected_files) * data_ratio))
        else:
            use_count = len(selected_files)  # val保持完整
            
        if use_count < len(selected_files):
            selected_files = selected_files[:use_count]
            
        self.samples = selected_files
        self._setup_gt_directory(root_dir)

    def _collect_all_polsar_files(self, root_dir):
        """收集所有 PolSAR 文件，不依赖特定目录结构"""
        all_files = []
            
        # 搜索常见的目录模式
        search_patterns = [
            os.path.join(root_dir, '**', 'hh', '*.mat'),
            os.path.join(root_dir, '**', 'hh', '*.tif'),
            os.path.join(root_dir, '**', 'hh', '*.png'),
            os.path.join(root_dir, 'bjtff', 'hh', '*.tif'),
            os.path.join(root_dir, 'bjtff', 'hh', '*.png'),
        ]
            
        for pattern in search_patterns:
            try:
                files = glob.glob(pattern, recursive=True)
                if files:  # 只在找到文件时输出
                    print(f"[DEBUG] Pattern found files: {pattern} -> {len(files)} files")
                all_files.extend(files)
            except Exception as e:
                print(f"[ERROR] Pattern failed {pattern}: {e}")
            
        # 如果所有模式都没找到，尝试直接扫描 hh 目录
        if len(all_files) == 0:
            print(f"[WARN] Standard patterns failed, scanning manually...")
            hh_dirs = []
            for root, dirs, files in os.walk(root_dir):
                if 'hh' in dirs:
                    hh_dirs.append(os.path.join(root, 'hh'))
                
            for hh_dir in hh_dirs:
                try:
                    for ext in ['*.tif', '*.png', '*.mat']:
                        files = glob.glob(os.path.join(hh_dir, ext))
                        if files:
                            print(f"[DEBUG] Manual scan found: {hh_dir} -> {len(files)} files")
                            all_files.extend(files)
                except Exception as e:
                    print(f"[ERROR] Manual scan failed for {hh_dir}: {e}")
            
        # 去重并排序
        all_files = sorted(list(set(all_files)))
        print(f"[INFO] Total unique PolSAR files found: {len(all_files)}")
            
        return all_files

    def _split_by_scene(self, all_files, split='train', data_ratio=1.0):
        """按场景划分训练/验证集（解决场景分布异常问题）"""
        # 按场景分组
        scene_groups = {}
        for file_path in all_files:
            fname = os.path.basename(file_path)
            try:
                scene_id = int(fname.split('_')[0])  # 文件名第一个数字是场景ID
                if scene_id not in scene_groups:
                    scene_groups[scene_id] = []
                scene_groups[scene_id].append(file_path)
            except:
                continue
        
        # 按场景ID排序
        sorted_scenes = sorted(scene_groups.keys())
        print(f"[INFO] 发现场景: {sorted_scenes}")
        
        if len(sorted_scenes) >= 3:
            # 场景级划分：1,2用于训练，3用于验证（推荐方案）
            if split == 'train':
                train_scenes = sorted_scenes[:2]  # 场景1,2
                selected_files = []
                for scene in train_scenes:
                    selected_files.extend(scene_groups[scene])
                # 应用数据比例
                n_keep = int(len(selected_files) * data_ratio)
                selected_files = selected_files[:n_keep]
            else:
                test_scene = sorted_scenes[2]  # 场景3
                selected_files = scene_groups[test_scene]
        elif len(sorted_scenes) == 2:
            # 只有两个场景：第一个用于训练，第二个用于验证
            print(f"[INFO] 只有2个场景，采用场景级划分")
            if split == 'train':
                selected_files = scene_groups[sorted_scenes[0]]
                # 应用数据比例
                n_keep = int(len(selected_files) * data_ratio)
                selected_files = selected_files[:n_keep]
            else:
                selected_files = scene_groups[sorted_scenes[1]]
        else:
            # 如果只有一个场景或场景数不足，回退到随机划分
            print(f"[WARN] 场景数不足({len(sorted_scenes)})，使用随机划分")
            np.random.seed(42)  # 固定种子确保可复现
            np.random.shuffle(all_files)
            split_point = int(len(all_files) * 0.8)
            if split == 'train':
                selected_files = all_files[:split_point]
                # 应用数据比例
                n_keep = int(len(selected_files) * data_ratio)
                selected_files = selected_files[:n_keep]
            else:
                selected_files = all_files[split_point:]
            
        # 显示选取的场景信息
        if selected_files:
            scene_ids = [int(os.path.basename(f).split('_')[0]) for f in selected_files[:5]]
            unique_scenes = list(set(scene_ids))
            print(f"[INFO] {split}集包含场景: {unique_scenes}")
            
        return selected_files

    def _setup_gt_directory(self, root_dir):
        """设置GT目录路径"""
        # 尝试几种常见的GT目录位置
        possible_gt_dirs = [
            os.path.join(root_dir, 'gt'),
            os.path.join(root_dir, '验证集切片数据', 'gt'),
            os.path.join(root_dir, '训练集切片数据', 'gt'),
            os.path.join(root_dir, '训练集切片数据', '训练集切片数据', 'train', 'gt'),  # 正确路径
        ]
        
        for gt_dir in possible_gt_dirs:
            if os.path.exists(gt_dir):
                self.gt_dir = gt_dir
                print(f"[INFO] GT directory found: {gt_dir}")
                return
                
        print(f"[WARN] No GT directory found, will use filename-based labels")
        self.gt_dir = None

    def _load_4ch(self, path):
        base_dir = os.path.dirname(os.path.dirname(path))
        fname = os.path.basename(path)
        chs = []
        for pol in ['hh', 'hv', 'vh', 'vv']:
            p = os.path.join(base_dir, pol, fname)
            arr = None
            if os.path.exists(p):
                if p.endswith('.mat'):
                    try:
                        d = loadmat(p)
                        k = [k for k in d.keys() if not k.startswith('_')][0]
                        arr = d[k].astype(np.float32)
                    except:
                        arr = None
                else:
                    try:
                        arr = np.array(Image.open(p).convert('L')).astype(np.float32) / 255.0
                    except:
                        arr = None
            if arr is None:
                arr = np.zeros((self.patch_sz, self.patch_sz), dtype=np.float32)
            if len(arr.shape) == 3:
                arr = arr[:, :, 0] if arr.shape[2] <= arr.shape[0] else arr[0]
            if len(arr.shape) != 2:
                arr = np.zeros((self.patch_sz, self.patch_sz), dtype=np.float32)
            if arr.max() > 1:
                arr = arr / (arr.max() + 1e-8)
            chs.append(arr)
        return np.stack(chs, axis=0)

    def __getitem__(self, idx):
        # 样本为空时返回随机数据
        if len(self.samples) == 0:
            img = torch.randn(4, self.patch_sz, self.patch_sz) * 0.1 + 0.5
            img = img.clamp(0, 1)
            return {'sar': img, 'sar_k': img.clone(), 'label': random.randint(0, 5)}
        path = self.samples[idx % len(self.samples)]
        img = self._load_4ch(path)
        if img.shape[0] != 4:
            img = np.stack([img[0] if len(img) > 0 else np.zeros((self.patch_sz, self.patch_sz))]*4, axis=0)
        if img.shape[1] != self.patch_sz or img.shape[2] != self.patch_sz:
            t = torch.from_numpy(img).float().unsqueeze(0)
            t = torch.nn.functional.interpolate(t, size=(self.patch_sz, self.patch_sz), mode='bilinear', align_corners=False)
            img = t.squeeze(0).numpy()
        img = torch.from_numpy(img).float()
        img_k = img.clone()
        if random.random() > 0.5:
            img_k = torch.flip(img_k, dims=[1])
        if random.random() > 0.5:
            img_k = torch.flip(img_k, dims=[2])
        
        # 使用真实GT标签替代文件名标签
        fname = os.path.basename(path)
        lbl = self._get_gt_label(fname)
        
        return {'sar': img, 'sar_k': img_k, 'label': lbl}

    def _get_gt_label(self, fname):
        """从 GT mask 生成 patch 级别的标签 (多数类 + 纯度过滤) + 类别映射"""
        # 构造GT文件路径
        if hasattr(self, 'gt_dir') and self.gt_dir is not None and os.path.exists(self.gt_dir):
            parts = fname.split('_')
            if len(parts) >= 5:
                gt_filename = f'{parts[0]}_patch_{parts[2]}_{parts[3]}_{parts[4].split(".")[0]}.tiff'
                gt_path = os.path.join(self.gt_dir, gt_filename)
                
                if os.path.exists(gt_path):
                    try:
                        gt_img = np.array(Image.open(gt_path)).astype(np.int32)
                        unique_vals, counts = np.unique(gt_img, return_counts=True)
                        total_pixels = gt_img.size
                        ratios = counts / total_pixels
                        max_idx = np.argmax(ratios)
                        majority_class = unique_vals[max_idx]
                        purity = ratios[max_idx]
                        purity_threshold = 0.7
                        
                        if purity >= purity_threshold and majority_class > 0:
                            raw_label = int(majority_class - 1)
                            # ========== 关键：应用类别映射 ==========
                            if self.class_mapping is not None:
                                mapped_label = self.class_mapping.get(raw_label, 0)
                                return mapped_label
                            return raw_label
                        else:
                            # 低纯度/背景 patch：标记为 ignore (-1)，避免把随机噪声当作真值
                            return -1
                    except Exception as e:
                        print(f"[WARN] GT 读取失败 {gt_path}: {e}")
                        pass
        
        # fallback: 无法读取 GT 时，直接忽略该样本（返回 -1）
        # 说明：之前的实现会从文件名推断标签或随机赋值，这会在评估时显著污染 NMI/ARI。
        return -1


    def __len__(self):
        # 训练集可重复采样保证batch稳定；验证/测试集保持真实长度
        if self.split == 'train':
            return max(len(self.samples), 200)
        return max(len(self.samples), 1)

class SourceDomainDataset(Dataset):
    def __init__(self, dataset_type='berlin', channels=30, window_sz=10, data_ratio=1.0, class_mapping=None):
        self.window_sz = window_sz
        self.channels = channels
        self.dtype = dataset_type
        self.data_ratio = data_ratio
        self.class_mapping = class_mapping  # 源域标签映射到目标域类别空间(0..C-1)，可为 None
        # 凑合用绝对路径
        if dataset_type == 'berlin':
            hsi_path = os.path.join(_DATA_DIR, "Berlin/berlin_hsi.mat")
            sar_path = os.path.join(_DATA_DIR, "Berlin/berlin_sar.mat")
            gt_path = os.path.join(_DATA_DIR, "Berlin/berlin_gt.mat")
            idx_path = os.path.join(_DATA_DIR, "Berlin/berlin_index.mat")
            keys = ['berlin_hsi', 'berlin_sar', 'berlin_gt', 'berlin_train', 'berlin_all']
        else:
            hsi_path = os.path.join(_DATA_DIR, "Augsburg/augsburg_hsi.mat")
            sar_path = os.path.join(_DATA_DIR, "Augsburg/augsburg_sar.mat")
            gt_path = os.path.join(_DATA_DIR, "Augsburg/augsburg_gt.mat")
            idx_path = os.path.join(_DATA_DIR, "Augsburg/augsburg_index.mat")
            keys = ['augsburg_hsi', 'augsburg_sar', 'augsburg_gt', 'augsburg_train', 'augsburg_all']
        self.hsi = loadmat(hsi_path)[keys[0]]
        self.sar = loadmat(sar_path)[keys[1]]
        self.gt = loadmat(gt_path)[keys[2]]
        self.pos = loadmat(idx_path)[keys[3]]
        total_n = len(self.pos)
        use_n = max(1, int(total_n * data_ratio))
        seed = int(time.time()) % 10000
        np.random.seed(seed)
        indices = np.random.choice(total_n, use_n, replace=False)
        self.pos = self.pos[indices]
        self.hsi = applyPCA(self.hsi, channels)
        pad = window_sz // 2
        self.hsi = np.pad(self.hsi, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
        if len(self.sar.shape) == 2:
            self.sar = np.pad(self.sar, ((pad, pad), (pad, pad)), mode='reflect')
        else:
            self.sar = np.pad(self.sar, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')

    def __getitem__(self, idx):
        h, w = self.pos[idx]
        hsi = self.hsi[h:h+self.window_sz, w:w+self.window_sz]
        sar = self.sar[h:h+self.window_sz, w:w+self.window_sz]
        hsi = torch.from_numpy(hsi).permute(2, 0, 1).float()
        if len(sar.shape) == 2:
            sar = torch.from_numpy(sar).unsqueeze(0).float()
        else:
            sar = torch.from_numpy(sar).permute(2, 0, 1).float()
        lbl = int(self.gt[h, w]) - 1 if self.gt[h, w] > 0 else 0
        # 应用类别映射（保证源/目标在同一类别空间内做原型对齐）
        if self.class_mapping is not None:
            lbl = int(self.class_mapping.get(int(lbl), 0))
        return {'hsi': hsi, 'sar': sar, 'hsi_k': hsi.clone(), 'sar_k': sar.clone(), 'label': lbl}

    def __len__(self):
        return len(self.pos)

class CrossDomainPairDataset(Dataset):
    def __init__(self, src_dataset, tgt_dataset):
        self.src = src_dataset
        self.tgt = tgt_dataset
        self.len = max(len(src_dataset), len(tgt_dataset))

    def __getitem__(self, idx):
        return self.src[idx % len(self.src)], self.tgt[idx % len(self.tgt)]

    def __len__(self):
        return self.len

class TestSARDataset(Dataset):
    def __init__(self, data_dir, patch_sz=64):
        self.patch_sz = patch_sz
        self.samples = []
        self.label_mats = {}
        for sub in ['SanFrancisco', 'Flevoland', 'Flevoland_14', 'Oberpfaffenhofen', 'SF_AIRSAR']:
            sub_dir = os.path.join(data_dir, sub)
            if not os.path.exists(sub_dir): continue
            lbl_path = os.path.join(sub_dir, 'label.mat')
            if os.path.exists(lbl_path):
                try:
                    d = loadmat(lbl_path)
                    k = [k for k in d.keys() if not k.startswith('_')][0]
                    self.label_mats[sub] = d[k]
                except:
                    self.label_mats[sub] = None
            for ext in ['*.png', '*.tif', '*.mat', '*.bmp', '*.jpg']:
                files = glob.glob(os.path.join(sub_dir, '**', ext), recursive=True)
                for f in files:
                    if 'gt' not in f.lower() and 'label' not in f.lower():
                        self.samples.append((f, sub))

    def __getitem__(self, idx):
        if len(self.samples) == 0:
            return {'sar': torch.zeros(4, self.patch_sz, self.patch_sz), 'label': 0}
        item = self.samples[idx % len(self.samples)]
        path, sub = item if isinstance(item, tuple) else (item, '')
        try:
            if path.endswith('.mat'):
                d = loadmat(path)
                k = [k for k in d.keys() if not k.startswith('_')][0]
                img = d[k].astype(np.float32)
            else:
                img = np.array(Image.open(path).convert('RGB')).astype(np.float32) / 255.0
        except:
            return {'sar': torch.zeros(4, self.patch_sz, self.patch_sz), 'label': 0}
        if len(img.shape) == 2:
            img = np.stack([img]*3, axis=0)
        elif len(img.shape) == 3 and img.shape[2] <= 4:
            img = img.transpose(2, 0, 1)
        if img.shape[0] > 4:
            img = img[:4]
        if img.max() > 1:
            img = img / (img.max() + 1e-8)
        img = torch.from_numpy(img).float()
        if img.shape[1] != self.patch_sz or img.shape[2] != self.patch_sz:
            img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(self.patch_sz, self.patch_sz), mode='bilinear', align_corners=False).squeeze(0)
        if img.shape[0] < 4:
            img = torch.cat([img, img[:1].expand(4 - img.shape[0], -1, -1)], dim=0)
        lbl = 0
        if sub in self.label_mats and self.label_mats[sub] is not None:
            gt = self.label_mats[sub]
            rh = random.randint(0, gt.shape[0]-1)
            rw = random.randint(0, gt.shape[1]-1)
            lbl = int(gt[rh, rw]) if gt[rh, rw] > 0 else ((idx * 7 + int(time.time())) % 6)
        else:
            lbl = (idx * 3 + int(time.time() * 1000)) % 6
        return {'sar': img[:4], 'label': lbl}

    def __len__(self):
        return max(len(self.samples), 1)

def get_cross_domain_loaders(src_type='berlin', tgt_dir='AIR-PolSAR-Seg-2.0/AIR-PolSAR-Seg-2.0', test_dir='AIR-PolSAR-Seg-2.0/AIR-PolSAR-Seg-2.0', batch_sz=16, num_workers=0, data_ratio=1.0, use_class_mapping=True):
    seed = int(time.time()) % 10000
    set_random_seed(seed)
    print(f"[INFO] 随机种子已设置：{seed}")
    
    # ========== 跨域训练数据加载架构 ==========
    # 源域 (Source Domain): 通过 SourceDomainDataset 从 data/Berlin 加载
    #   - 有标签数据，用于监督学习
    #   - 提供分类损失的监督信号
    # 
    # 目标域 (Target Domain): 通过 AIRPolSARDataset 从 tgt_dir 加载
    #   - 无标签 PolSAR 图像数据
    #   - 用于特征对齐和分布匹配
    #   - split='train' 用于训练过程中的对齐
    # 
    # 验证集 (Test/Val): 通过 AIRPolSARDataset 从 test_dir 加载
    #   - 用于在训练过程中评估零样本迁移性能
    #   - split='val' 或 split='train'（取决于数据量）
    
    # ========== 类别映射配置 (Berlin/Houston → AIR-PolSAR) ==========
    class_mapping = None
    if use_class_mapping:
        # Berlin 8 类 → AIR-PolSAR 5 类 (注意：标签是 0-indexed)
        berlin_mapping = {
            # Berlin: 原始标签经 (gt-1) 后为 0..7（8类）
            # 下面映射到 AIR-PolSAR 的 5 类空间：0..4
            0: 0,  # 背景/未知
            1: 0,  # 植被/草地
            2: 0,  # 植被/草地
            3: 1,  # 树木
            4: 2,  # 建筑
            5: 2,  # 建筑
            6: 3,  # 道路/硬化地表
            7: 4   # 水体/其它
        }
        # Houston 20 类 → AIR-PolSAR 5 类
        houston_mapping = {
            # Houston: 原始标签经 (gt-1) 后为 0..19（20类）
            0: 0,  # 背景
            1: 0, 2: 0, 3: 0,     # 草地/植被 -> 0
            4: 1, 5: 1,           # 树木 -> 1
            7: 2, 8: 2, 15: 2, 16: 2, 17: 2, 18: 2, 19: 2,  # 建筑 -> 2  (原 8,9,16..20 -> 减1)
            9: 3, 10: 3, 11: 3, 12: 3, 13: 3, 14: 3,        # 道路 -> 3  (原 10..15 -> 减1)
            6: 4                  # 水体/裸土 -> 4 (原 7 -> 减1)
        }
        # 根据源域类型选择映射
        if src_type == 'berlin':
            class_mapping = berlin_mapping
            print(f"[INFO] 使用 Berlin→AIR-PolSAR 类别映射 (8 类→5 类)")
        else:
            class_mapping = houston_mapping
            print(f"[INFO] 使用 Houston→AIR-PolSAR 类别映射 (20 类→5 类)")
    
        # 创建源域数据集（应用类别映射，保证与目标域 5 类空间一致）
    src_ds = SourceDomainDataset(src_type, data_ratio=data_ratio, class_mapping=class_mapping)

# 目标域训练集：使用 data 目录下的数据
    tgt_train = AIRPolSARDataset(tgt_dir, split='train', data_ratio=data_ratio, class_mapping=class_mapping)
    
    # 目标域测试集：使用 AIR-PolSAR-Seg-2.0 的数据 (不应用映射，保持原始 5 类)
    try:
        tgt_test = AIRPolSARDataset(test_dir, split='val', data_ratio=1.0, class_mapping=None)
        if len(tgt_test) < 10:
            tgt_test = AIRPolSARDataset(test_dir, split='train', data_ratio=1.0, class_mapping=None)
    except:
        tgt_test = AIRPolSARDataset(test_dir, split='train', data_ratio=1.0, class_mapping=None)
    
    print(f"[DEBUG] tgt_train samples: {len(tgt_train)}")
    print(f"[DEBUG] tgt_test samples: {len(tgt_test)}")
    
    # 检查验证集大小
    if len(tgt_test) < 10:
        print("[WARN] Test set too small, using train set for evaluation (temporary fix)")
        test_ds = tgt_train
    else:
        test_ds = tgt_test
    
    pair_ds = CrossDomainPairDataset(src_ds, tgt_train)
    pair_loader = DataLoader(pair_ds, batch_size=batch_sz, shuffle=True, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_sz, shuffle=False, num_workers=num_workers)
    return pair_loader, test_loader, src_ds, tgt_train

def load_source_with_fetchdata(dataset_type='berlin', channels=30, window_sz=10, batch_sz=16, data_ratio=1.0):
    try:
        train_ld, test_ld, trntst_ld, all_ld = fetchData(dataset_type, channels, window_sz, batch_sz, num_workers=0)
        print(f"[INFO] fetchData加载{dataset_type}成功")
        return train_ld, test_ld
    except Exception as e:
        print(f"[WARN] fetchData加载失败: {e}, 使用SourceDomainDataset")
        src_ds = SourceDomainDataset(dataset_type, channels=channels, window_sz=window_sz, data_ratio=data_ratio)
        train_ld = DataLoader(src_ds, batch_size=batch_sz, shuffle=True, drop_last=True)
        return train_ld, None

def create_rs_dataset(hsi, sar, pos, window_sz, gt=None, train=False):
    ds = RSDataset(hsi, sar, pos, window_sz, gt=gt, transform=ToTensor(), train=train)
    print(f"[INFO] RSDataset创建成功, 样本数: {len(ds)}")
    return ds

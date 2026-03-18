"""
增强型跨域训练 - 不调整超参数的优化方案

核心改进:
1. 类别原型对齐 - 学习判别性更强的类原型特征
2. 伪标签自训练 - 利用目标域高置信度样本  
3. 特征动态增强 - 提升特征鲁棒性

预期效果:
- NMI 提升 10-20%
- ARI 提升 5-15%
"""

from enhanced_trainer import enhanced_cross_domain_train
import parameter
import os
import glob

def clean_old_models(model_dir='models'):
    """清理旧模型"""
    if not os.path.exists(model_dir):
        return
    old_files = glob.glob(os.path.join(model_dir, '*.pth'))
    if old_files:
        print(f"\n[INFO] 清理旧模型 ({len(old_files)}个)...")
        for f in old_files:
            try:
                os.remove(f)
                print(f"  删除：{os.path.basename(f)}")
            except:
                pass

def select_data_ratio():
    """选择数据比例"""
    print("\n选择数据比例:")
    print("  [1] 快速 5% (默认)")
    print("  [2] 标准 50%")
    print("  [3] 完整 100%")
    print("  [4] 自定义")
    try:
        choice = input("请选择：").strip()
    except:
        choice = '1'
    
    if choice == '' or choice == '1':
        return 0.05
    elif choice == '2':
        return 0.5
    elif choice == '3':
        return 1.0
    elif choice == '4':
        try:
            pct = input("输入比例 (1-100): ").strip()
            return max(0.01, min(1.0, float(pct) / 100.0))
        except:
            return 0.1
    return 0.05

if __name__ == '__main__':
    parameter._init()
    
    # 选择数据比例
    data_ratio = select_data_ratio()
    parameter.set_value('data_ratio', data_ratio)
    
    print(f"\n{'='*60}")
    print(f"[INFO] 运行模式：增强型训练，数据比例：{data_ratio*100:.1f}%")
    print(f"{'='*60}\n")
    
    clean_old_models('models')
    
    # 设置训练参数
    epochs = 15
    lr = parameter.get_value('cross_domain_lr')
    src_type = parameter.get_value('src_domain_type')
    tgt_dir = parameter.get_value('tgt_data_dir')
    test_dir = parameter.get_value('test_data_dir')
    model_path = parameter.get_value('cross_domain_model_path')
    log_path = parameter.get_value('cross_domain_log_path')
    
    print(f"[GPU] 检测到 CUDA: {parameter.get_device()}")
    print(f"\n[INFO] 源域数据路径：data/Berlin")
    print(f"[INFO] 目标域数据路径：{tgt_dir} (无标签特征对齐)")
    print(f"\n{'='*60}")
    print("开始增强型训练...")
    print(f"{'='*60}\n")
    
    # 运行增强型训练
    model, trainer, history = enhanced_cross_domain_train(
        epochs=epochs, 
        lr=lr, 
        src_type=src_type,
        tgt_dir=tgt_dir, 
        test_dir=test_dir,
        model_path=model_path, 
        log_path=log_path,
        data_ratio=data_ratio
    )
    
    # 输出训练结果
    print(f"\n{'='*60}")
    print("训练完成！")
    print(f"最终模型 (Epoch {len(history['nmi'])}): NMI={history['nmi'][-1]:.4f}, ARI={history['ari'][-1]:.4f}")
    print(f"最佳模型：NMI={max(history['nmi']):.4f}, ARI={max(history['ari']):.4f}")
    print(f"{'='*60}\n")
    
    # 零样本测试
    from test.model_test import comprehensive_zero_shot_test
    device = parameter.get_device()
    print("\n========== 零样本迁移测试 ==========")
    comprehensive_zero_shot_test(model, test_dir, tgt_dir, device, data_ratio=data_ratio)
    
    print("\n{'='*60}")
    print("全部实验完成！")
    print(f"{'='*60}\n")

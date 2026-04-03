import sys
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# ================= 0. 动态路径识别 =================
current_dir = os.getcwd()
if os.path.basename(current_dir) == 'notebooks':
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
else:
    project_root = os.path.abspath(current_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

DATA_DIR = os.path.join(project_root, 'data')
MODELS_DIR = os.path.join(project_root, 'models')

# 【核心修复点】: 直接从我们写好的 train.py 中导入 MoE 模型和数据集类
from train import SimpleMoE, CustomDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 1. 评估主流程 =================
def main():
    test_file = os.path.join(DATA_DIR, 'Testing_set.csv')
    model_path = os.path.join(MODELS_DIR, 'model.pth')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    
    # 这里的参数必须与 train.py 训练时保持绝对一致
    HIDDEN_DIM = 39
    NUM_EXPERTS = 5

    if not os.path.exists(model_path):
        print(f"❌ 找不到模型文件: {model_path}，请先运行 train.py！")
        return
        
    print(f"📦 正在加载测试数据: {test_file}")
    try:
        # is_train=False 表示不重新拟合 Scaler，而是读取之前存好的 scaler_path
        test_dataset = CustomDataset(test_file, is_train=False, scaler_path=scaler_path)
    except FileNotFoundError:
        print(f"❌ 找不到文件。请确保 {test_file} 和 {scaler_path} 都存在！")
        return

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    input_dim = test_dataset.X.shape[1]
    
    # 【核心修复点】: 初始化正确的 SimpleMoE 架构，而不是普通 KAN
    model = SimpleMoE(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_experts=NUM_EXPERTS).to(DEVICE)
    
    # 加载权重现在会完美契合！
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    criterion = nn.MSELoss()
    total_loss = 0.0
    
    all_preds = []
    all_targets = []
    
    print("🔍 开始评估模型...")
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            outputs = model(batch_X)
            
            # 对齐形状
            if outputs.shape != batch_y.shape:
                outputs = outputs.view_as(batch_y)
                
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            
    avg_loss = total_loss / len(test_loader)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # 计算评估指标
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    mae = np.mean(np.abs(all_preds - all_targets))
    
    # 计算 R2 Score (为了不额外引入库，手写计算公式)
    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    print("\n📊 =============== 测试集评估报告 ===============")
    print(f"平均 Loss (MSE)   : {avg_loss:.4f}")
    print(f"均方根误差 (RMSE) : {rmse:.4f}")
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print(f"决定系数 (R2)     : {r2:.4f}")
    print("=================================================\n")
    
    # 将预测结果保存，方便后续画图对比
    result_df = pd.DataFrame({
        'True_Values': all_targets.flatten(),
        'Predictions': all_preds.flatten()
    })
    result_path = os.path.join(DATA_DIR, 'predictions_test.csv')
    result_df.to_csv(result_path, index=False)
    print(f"💾 测试集详细预测结果已保存至: {result_path}")

if __name__ == "__main__":
    main()
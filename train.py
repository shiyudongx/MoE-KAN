import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# ================= 0. 动态路径识别 (永不报错的路径管理) =================
current_dir = os.getcwd()
if os.path.basename(current_dir) == 'notebooks':
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
else:
    project_root = os.path.abspath(current_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

DATA_DIR = os.path.join(project_root, 'data')
MODELS_DIR = os.path.join(project_root, 'models')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# 导入核心的 KAN 层
from src.efficient_kan.kan import KAN

# ================= 1. MoE-KAN 模型结构 =================
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.fc1 = KAN([input_dim, hidden_dim, output_dim])
        
    def forward(self, x):
        return self.fc1(x)

class MoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts=5):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(input_dim, num_experts)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.gate.bias, 0.0)

    def forward(self, x):
        gate_scores = self.gate(x)
        gate_probs = F.softmax(gate_scores, dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        output = torch.einsum('be,beo->bo', gate_probs, expert_outputs)
        return output

class SimpleMoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts, output_dim=1):
        super(SimpleMoE, self).__init__()
        self.moe = MoE(input_dim, hidden_dim, output_dim, num_experts)
        
    def forward(self, x):
        output = self.moe(x)
        return output.squeeze(-1)

# ================= 2. 数据集与全局配置 =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self, csv_file, is_train=True, scaler_path=None):
        df = pd.read_csv(csv_file)
        
        # 【核心修复点】: 不再需要硬编码列名，自动提取前 N-1 列为特征，最后一列为目标
        X_raw = df.iloc[:, :-1].values.astype(np.float32)
        y_raw = df.iloc[:, -1].values.astype(np.float32)
        
        # 自动标准化处理 (对 KAN 网络非常重要)
        if is_train:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(X_raw)
            if scaler_path:
                joblib.dump(self.scaler, scaler_path)
                print(f"💾 标准化工具 (Scaler) 已保存至: {scaler_path}")
        else:
            self.scaler = joblib.load(scaler_path)
            self.X = self.scaler.transform(X_raw)
            
        self.y = y_raw
        if len(self.y.shape) == 1:
            self.y = self.y.reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# ================= 3. 主训练流程 =================
def main():
    print(f"🚀 正在使用计算设备: {DEVICE}")
    
    # 路径配置
    train_file = os.path.join(DATA_DIR, 'Training_set.csv')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    model_save_path = os.path.join(MODELS_DIR, 'model.pth')
    
    # 超参数配置 (使用你之前提供的最佳参数)
    BATCH_SIZE = 64
    EPOCHS = 200
    LEARNING_RATE = 0.033
    HIDDEN_DIM = 39
    NUM_EXPERTS = 5

    print(f"📦 正在加载训练数据: {train_file}")
    
    # 初始化数据集，这会自动保存 scaler.pkl 供以后的 SHAP 和评估使用
    train_dataset = CustomDataset(train_file, is_train=True, scaler_path=scaler_path)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    input_dim = train_dataset.X.shape[1]
    
    # 初始化 MoE-KAN 模型
    model = SimpleMoE(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_experts=NUM_EXPERTS).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("🔥 开始训练 MoE-KAN 网络...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # 确保预测值和真实值的形状对齐
            if outputs.shape != batch_y.shape:
                outputs = outputs.view_as(batch_y)
                
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # 梯度裁剪，防止 KAN 网络训练早期出现梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        # 每隔 20 轮打印一次进度
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:03d}/{EPOCHS}] | 训练 Loss (MSE): {avg_loss:.4f}")
            
    # 训练结束，保存最终的模型权重
    torch.save(model.state_dict(), model_save_path)
    print("=" * 50)
    print(f"✅ 训练完毕！")
    print(f"📄 模型参数已保存至: {model_save_path}")
    print("=" * 50)

if __name__ == "__main__":
    main()
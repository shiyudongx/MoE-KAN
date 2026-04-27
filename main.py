import sys
import os
import random
import copy
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from torch.utils.data import Dataset, DataLoader

# ================= 0. 动态路径识别 =================
current_dir = os.getcwd()

if os.path.basename(current_dir) == "notebooks":
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
else:
    project_root = os.path.abspath(current_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

DATA_DIR = os.path.join(project_root, "data")
MODELS_DIR = os.path.join(project_root, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# 你的 KAN 路径
from src.efficient_kan import KAN

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")


# ================= 1. 固定随机种子 =================
def set_all_seeds(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ================= 2. 模型结构 =================
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.fc1 = KAN([input_dim, hidden_dim, output_dim])

    def forward(self, x):
        return self.fc1(x)


class MoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts=4):
        super(MoE, self).__init__()

        self.num_experts = num_experts

        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim)
            for _ in range(num_experts)
        ])

        self.gate = nn.Linear(input_dim, num_experts)

        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.gate.bias, 0.0)

    def forward(self, x, return_gate_weights=False):
        gate_scores = self.gate(x)
        gate_probs = F.softmax(gate_scores, dim=-1)

        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts],
            dim=1
        )

        output = torch.einsum("be,beo->bo", gate_probs, expert_outputs)

        if return_gate_weights:
            return output, gate_probs

        return output


class SimpleMoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts, output_dim=1):
        super(SimpleMoE, self).__init__()
        self.moe = MoE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_experts=num_experts
        )

    def forward(self, x, return_gate_weights=False):
        if return_gate_weights:
            output, gate_probs = self.moe(x, return_gate_weights=True)
            return output.squeeze(-1), gate_probs
        else:
            output = self.moe(x)
            return output.squeeze(-1)


# ================= 3. 测试集 Dataset =================
class TestDataset(Dataset):
    def __init__(self, csv_file, scaler_path):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"找不到测试文件: {csv_file}")

        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"找不到 scaler 文件: {scaler_path}")

        df = pd.read_csv(csv_file)

        if df.isnull().values.any():
            print("⚠️ 测试集中存在空值，已自动删除含空值的行")
            df = df.dropna()

        data = df.values

        self.X_np = data[:, :-1]
        self.y_np = data[:, -1]

        scaler = joblib.load(scaler_path)
        self.X_scaled = scaler.transform(self.X_np)

        self.X = torch.FloatTensor(self.X_scaled)
        self.y = torch.FloatTensor(self.y_np)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ================= 4. 训练数据读取与预处理 =================
def load_and_process_train_data(train_file, seed=42):
    set_all_seeds(seed)

    if not os.path.exists(train_file):
        raise FileNotFoundError(f"找不到训练文件: {train_file}")

    df_full = pd.read_csv(train_file)

    if df_full.isnull().values.any():
        print("⚠️ 训练集中存在空值，已自动删除含空值的行")
        df_full = df_full.dropna()

    print(f"成功读取训练数据: {train_file}")
    print(f"训练数据 Shape: {df_full.shape}")

    train_data, val_data = train_test_split(
        df_full.values,
        test_size=0.2,
        random_state=seed
    )

    X_train_np = train_data[:, :-1]
    y_train_np = train_data[:, -1]

    X_val_np = val_data[:, :-1]
    y_val_np = val_data[:, -1]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_np)
    X_val_scaled = scaler.transform(X_val_np)

    X_train = torch.FloatTensor(X_train_scaled).to(DEVICE)
    y_train = torch.FloatTensor(y_train_np).to(DEVICE)

    X_val = torch.FloatTensor(X_val_scaled).to(DEVICE)
    y_val = torch.FloatTensor(y_val_np).to(DEVICE)

    return X_train, y_train, X_val, y_val, scaler


# ================= 5. 训练函数 =================
def train_model(params, X_train, y_train, X_val, y_val, seed=42, show_plots=True):
    set_all_seeds(seed)

    hidden_dim = int(params["hidden_dim"])
    num_experts = int(params["num_experts"])
    learning_rate = float(params["learning_rate"])
    batch_size = int(params["batch_size"])
    num_epochs = int(params.get("num_epochs", 200))

    input_dim = X_train.shape[1]

    model = SimpleMoE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_experts=num_experts
    ).to(DEVICE)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=10,
        factor=0.5
    )

    train_losses = []
    val_losses = []

    best_val_r2 = -float("inf")
    best_model_state = None

    print("\n====== 开始训练 ======")
    print("当前参数:", params)

    for epoch in range(num_epochs):
        model.train()

        permutation = torch.randperm(X_train.size(0))

        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i + batch_size]

            batch_X = X_train[indices]
            batch_y = y_train[indices]

            optimizer.zero_grad()

            outputs = model(batch_X)

            if outputs.shape != batch_y.shape:
                outputs = outputs.view_as(batch_y)

            loss = criterion(outputs, batch_y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)

        model.eval()

        with torch.no_grad():
            val_outputs = model(X_val)

            if val_outputs.shape != y_val.shape:
                val_outputs = val_outputs.view_as(y_val)

            val_loss = criterion(val_outputs, y_val).item()
            val_losses.append(val_loss)

            y_pred_val = val_outputs.cpu().numpy()
            y_true_val = y_val.cpu().numpy()

            val_r2 = r2_score(y_true_val, y_pred_val)

            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_model_state = copy.deepcopy(model.state_dict())

        scheduler.step(val_loss)

        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val R2: {val_r2:.4f}"
            )

    print("\n训练结束，加载验证集表现最佳的模型...")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        print("⚠️ 未找到最佳模型，使用最后一轮模型")

    model.eval()

    with torch.no_grad():
        train_outputs = model(X_train)

        if train_outputs.shape != y_train.shape:
            train_outputs = train_outputs.view_as(y_train)

        train_r2 = r2_score(
            y_train.cpu().numpy(),
            train_outputs.cpu().numpy()
        )

    print("\n📊 =============== 训练/验证结果 ===============")
    print(f"Train R2 : {train_r2:.4f}")
    print(f"Val R2   : {best_val_r2:.4f}")
    print("==============================================\n")

    if show_plots:
        plt.figure(figsize=(8, 4))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.title("Training Process")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    return model, train_r2, best_val_r2


# ================= 6. 保存训练集和验证集预测结果 =================
def save_train_val_predictions(model, X_train, y_train, X_val, y_val, train_r2, val_r2):
    model.eval()

    with torch.no_grad():
        train_pred = model(X_train).cpu().numpy()
        train_real = y_train.cpu().numpy()

        val_pred = model(X_val).cpu().numpy()
        val_real = y_val.cpu().numpy()

    df_train = pd.DataFrame({
        "True_Value": train_real.flatten(),
        "Predicted_Value": train_pred.flatten(),
        "Dataset": "Training"
    })

    df_val = pd.DataFrame({
        "True_Value": val_real.flatten(),
        "Predicted_Value": val_pred.flatten(),
        "Dataset": "Validation"
    })

    df_all = pd.concat([df_train, df_val], ignore_index=True)

    csv_path = os.path.join(DATA_DIR, "prediction_train_val.csv")
    df_all.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"💾 训练集和验证集预测结果已保存至: {csv_path}")

    plt.figure(figsize=(6, 6))

    plt.scatter(
        train_real,
        train_pred,
        alpha=0.4,
        label=f"Train (R2={train_r2:.3f})",
        s=30
    )

    plt.scatter(
        val_real,
        val_pred,
        alpha=0.7,
        label=f"Val (R2={val_r2:.3f})",
        marker="s",
        s=40
    )

    all_vals = np.concatenate([
        train_real.flatten(),
        val_real.flatten(),
        train_pred.flatten(),
        val_pred.flatten()
    ])

    min_val = np.min(all_vals)
    max_val = np.max(all_vals)

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "k--",
        lw=2,
        label="Ideal Fit"
    )

    plt.title("Prediction: Train vs Validation")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# ================= 7. 测试函数 =================
def evaluate_on_test(test_file, model_path, scaler_path, hidden_dim, num_experts):
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型文件: {model_path}")
        return

    if not os.path.exists(test_file):
        print(f"❌ 找不到测试文件: {test_file}")
        return

    if not os.path.exists(scaler_path):
        print(f"❌ 找不到 scaler 文件: {scaler_path}")
        return

    print(f"\n📦 正在加载测试数据: {test_file}")

    test_dataset = TestDataset(
        csv_file=test_file,
        scaler_path=scaler_path
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False
    )

    input_dim = test_dataset.X.shape[1]

    model = SimpleMoE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_experts=num_experts
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE)
    )

    model.eval()

    criterion = nn.MSELoss()

    total_loss = 0.0
    num_batches = 0

    all_preds = []
    all_targets = []

    print("🔍 开始测试集评估...")

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            outputs = model(batch_X)

            if outputs.shape != batch_y.shape:
                outputs = outputs.view_as(batch_y)

            loss = criterion(outputs, batch_y)

            total_loss += loss.item()
            num_batches += 1

            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / num_batches

    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()

    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    mae = np.mean(np.abs(all_preds - all_targets))

    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)

    if ss_tot == 0:
        r2 = np.nan
        print("⚠️ 测试集 y 值方差为 0，R2 无法计算")
    else:
        r2 = 1 - ss_res / ss_tot

    print("\n📊 =============== 测试集评估报告 ===============")
    print(f"平均 Loss (MSE)   : {avg_loss:.4f}")
    print(f"均方根误差 (RMSE) : {rmse:.4f}")
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print(f"决定系数 (R2)     : {r2:.4f}")
    print("=================================================\n")

    result_df = pd.DataFrame({
        "True_Values": all_targets,
        "Predictions": all_preds
    })

    result_path = os.path.join(DATA_DIR, "predictions_test.csv")
    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")

    print(f"💾 测试集详细预测结果已保存至: {result_path}")

    plt.figure(figsize=(6, 6))

    plt.scatter(
        all_targets,
        all_preds,
        alpha=0.7,
        label=f"Test (R2={r2:.3f})",
        s=40
    )

    min_val = min(np.min(all_targets), np.min(all_preds))
    max_val = max(np.max(all_targets), np.max(all_preds))

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "k--",
        lw=2,
        label="Ideal Fit"
    )

    plt.title("Prediction: Test Set")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# ================= 8. 主程序 =================
def main():
    seed = 42
    set_all_seeds(seed)

    train_file = os.path.join(DATA_DIR, "your_train.csv")
    test_file = os.path.join(DATA_DIR, "your_test.csv")

    model_path = os.path.join(MODELS_DIR, "model.pth")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")

    params = {
        "hidden_dim": 39,
        "num_experts": 5,
        "learning_rate": 0.033052046662679385,
        "batch_size": 64,
        "num_epochs": 200
    }

    # ========== 1. 读取并处理训练数据 ==========
    X_train, y_train, X_val, y_val, scaler = load_and_process_train_data(
        train_file=train_file,
        seed=seed
    )

    # ========== 2. 训练模型 ==========
    best_model, train_r2, val_r2 = train_model(
        params=params,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        seed=seed,
        show_plots=True
    )

    # ========== 3. 保存模型和 scaler ==========
    torch.save(best_model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)

    print(f"✅ 模型已保存至: {model_path}")
    print(f"✅ 标准化工具已保存至: {scaler_path}")

    # ========== 4. 保存训练集和验证集预测 ==========
    save_train_val_predictions(
        model=best_model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        train_r2=train_r2,
        val_r2=val_r2
    )

    # ========== 5. 验证保存后的模型是否正常 ==========
    print("\n------ 最终验证：加载保存的模型 ------")

    loaded_model = SimpleMoE(
        input_dim=X_train.shape[1],
        hidden_dim=int(params["hidden_dim"]),
        num_experts=int(params["num_experts"])
    ).to(DEVICE)

    loaded_model.load_state_dict(
        torch.load(model_path, map_location=DEVICE)
    )

    loaded_model.eval()

    with torch.no_grad():
        check_out = loaded_model(X_val)

        if check_out.shape != y_val.shape:
            check_out = check_out.view_as(y_val)

        check_r2 = r2_score(
            y_val.cpu().numpy(),
            check_out.cpu().numpy()
        )

    print(f"保存模型在验证集上的实际 R2: {check_r2:.4f}")

    if abs(check_r2 - val_r2) < 1e-4:
        print("✅ 验证成功：保存的模型就是最佳模型！")
    else:
        print("❌ 验证失败：保存的模型与报告结果不一致！")

    # ========== 6. 测试集评估 ==========
    evaluate_on_test(
        test_file=test_file,
        model_path=model_path,
        scaler_path=scaler_path,
        hidden_dim=int(params["hidden_dim"]),
        num_experts=int(params["num_experts"])
    )


if __name__ == "__main__":
    main()
import numpy as np
import os
import torch
import datetime
import gc

from action_model import  ( train_model, test_model,ActionModel_ours,ActionModel_V1,ActionModel_V2,ActionModel_V3,ActionModel_V4, ActionModel_V5,
                            ActionModel_V6,ActionModel_V7,ActionModel_V8,ActionModel_V9,ActionModel_V10,
                                ActionModel_V11, ActionModel_V12,ActionModel_V13,ActionModel_V14,ActionModel_V15,ActionModel_V16
                            )

from dataloader import get_dataloaders

from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    cohen_kappa_score, matthews_corrcoef,
    average_precision_score, top_k_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_dir = '/home/ubuntu22/PDD/Datensatz'
if not os.path.exists(base_dir):
    raise FileNotFoundError(f"数据集路径不存在: {base_dir}")

batch_size = 32
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="ours",
                    choices=['ours','V0','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16'],
                    help="选择要使用的模型版本")
parser.add_argument("--epochs", type=int, default=120,
                    help="训练轮数 (默认 120)")
args = parser.parse_args()
num_epochs = args.epochs
model_dict = {

    'ours': ActionModel_ours,
    'V1': ActionModel_V1,
    'V2': ActionModel_V2,
    'V3': ActionModel_V3,
    'V4': ActionModel_V4,
    'V5': ActionModel_V5,
    'V6': ActionModel_V6,
    'V7': ActionModel_V7,
    'V8': ActionModel_V8,
    'V9': ActionModel_V9,
    'V10': ActionModel_V10,
    'V11': ActionModel_V11,
    'V12': ActionModel_V12,
    'V13': ActionModel_V13,
    'V14': ActionModel_V14,
    'V15': ActionModel_V15,
    'V16': ActionModel_V16,

}

if args.model not in model_dict:
    raise ValueError(f"未知模型: {args.model}，可选: {list(model_dict.keys())}")
model = model_dict[args.model]().to(device)
model_name = args.model

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"🧠 使用模型: {args.model}")
print(f"🧠 模型总参数数目: {total_params:,}")
print(f"🎯 可训练参数数目: {trainable_params:,}")

if args.model == "ours":
    feature_folder = "dinov2_reg_base_map_16x16_768"
elif args.model == "V1":
    feature_folder = "swin_v2_map"
elif args.model == "V3":
    feature_folder = "vgg16_map"
elif args.model == "V2":
    feature_folder = "vit_base_map_14x14_768"
elif args.model == "V15":
    feature_folder = "deit_base_map_14x14_768"
elif args.model == "V16":
    feature_folder = "effnet_b0_map"

else:
    feature_folder = "dinov2_reg_base_map_16x16_768"

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"🧠 模型总参数数目: {total_params:,}")
print(f"🎯 可训练参数数目: {trainable_params:,}")
print("📊 加载完整数据集...")
dataloaders, distributions = get_dataloaders(
    base_dir, batch_size=batch_size, window_size=5, step=2, max_samples=None, feature_folder=feature_folder
)

for phase in ['train', 'val', 'test']:
    print(f"📊 数据集 {phase} 样本数: {len(dataloaders[phase].dataset)}")

train_dist = distributions['train']
total = sum(train_dist.values())
class_weights = [total / train_dist.get(i, 1) for i in range(3)]
w = torch.tensor(class_weights, dtype=torch.float32)
w = w / w.mean()
print(f"⚖️ 类别权重: {w}")

print("🚀 开始训练...")
save_dir = os.path.join("weights", model_name, timestamp)
os.makedirs(save_dir, exist_ok=True)
train_model(
    model,
    dataloaders,
    w,
    num_epochs=num_epochs,
    lr=5e-5,
    save_dir=save_dir,
    model_name=model_name
)
model_path = os.path.join(save_dir, f"final_model_{model_name}.pth")
torch.save(model.state_dict(), model_path)
print(f"✅ 模型训练完成，已保存到 {model_path}")

print("🚀 开始测试...")
test_dataloader = dataloaders['test']

best_model_path = os.path.join(save_dir, f"best_{model_name}.pth")
model.load_state_dict(torch.load(best_model_path, map_location=device))
print(f"✅ 已加载最佳模型权重: {best_model_path}")

all_preds, all_labels, all_probs = test_model(model, test_dataloader)
os.makedirs(save_dir, exist_ok=True)
report_path = os.path.join(save_dir, f"report_{model_name}.txt")

report_dict = classification_report(
    all_labels, all_preds, target_names=['call', 'normal', 'eye'], output_dict=True
)

report_text = classification_report(
    all_labels, all_preds, target_names=['call', 'normal', 'eye']
)
print(report_text)

overall_acc = report_dict['accuracy']
print(f"Overall Accuracy: {overall_acc:.4f}")

class_metrics = {}
for cls in ['call', 'normal', 'eye']:
    class_metrics[cls] = {
        "precision": report_dict[cls]["precision"],
        "recall": report_dict[cls]["recall"],
        "f1-score": report_dict[cls]["f1-score"],
        "support": report_dict[cls]["support"],
    }

auc_macro = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
auc_per_class = roc_auc_score(all_labels, all_probs, multi_class='ovr', average=None)
ap_per_class = average_precision_score(np.eye(3)[all_labels], all_probs, average=None)
top2_acc = top_k_accuracy_score(all_labels, all_probs, k=2)
kappa = cohen_kappa_score(all_labels, all_preds)
mcc = matthews_corrcoef(all_labels, all_preds)
conf_matrix = confusion_matrix(all_labels, all_preds)

with open(report_path, "w") as f:
    f.write("📊 Classification Report (sklearn):\n")
    f.write(report_text + "\n")
    f.write("\n📊 Extracted Metrics:\n")
    for cls, metrics in class_metrics.items():
        f.write(
            f"{cls}: "
            f"Precision={metrics['precision']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"F1={metrics['f1-score']:.4f}, "
            f"Support={metrics['support']}\n"
        )
    f.write(f"\nOverall Accuracy: {overall_acc:.4f}\n")
    f.write(f"AUC (macro): {auc_macro:.4f}\n")
    f.write(f"AUC per class: {auc_per_class.tolist()}\n")
    f.write(f"Average Precision per class: {ap_per_class.tolist()}\n")
    f.write(f"Top-2 Accuracy: {top2_acc:.4f}\n")
    f.write(f"Cohen's Kappa: {kappa:.4f}\n")
    f.write(f"Matthews Corrcoef: {mcc:.4f}\n")
    f.write("Confusion Matrix:\n")
    f.write(str(conf_matrix) + "\n")

print(f"✅ 指标已写入 {report_path}")

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=['call', 'normal', 'eye'],
            yticklabels=['call', 'normal', 'eye'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.tight_layout()
heatmap_path = os.path.join(save_dir, f"confusion_matrix_{model_name}.png")
plt.savefig(heatmap_path)
plt.close()
print(f"✅ 混淆矩阵图已保存到 {heatmap_path}")

# -*- coding: utf-8 -*-
"""resnet.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1SR218ItZ55RFj3OImPNUwGVi4lDO1Q_K
"""

# 🧠 ResNet18 기반 Binary Classification (Accuracy + Precision + Recall + F1 출력)

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import classification_report

# ✅ 1. 경로 & 장치 설정
data_dir = '/content/drive/MyDrive/split_dataset'  # train/, test/ 포함된 폴더
train_dir = os.path.join(data_dir, 'train')
test_dir  = os.path.join(data_dir, 'test')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 2. 전처리 + 데이터 로딩
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(train_dir, transform=transform)
test_data  = datasets.ImageFolder(test_dir, transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=32, shuffle=False)

print(f"✔️ 학습 데이터 수: {len(train_data)}, 테스트 데이터 수: {len(test_data)}")
print(f"✔️ 클래스 매핑: {train_data.class_to_idx}")  # 예: {'defect': 0, 'no_defect': 1}

# ✅ 3. 모델 정의 (ResNet18 + 1 output)
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 1)  # binary classification
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ✅ 4. 학습 함수
def train_one_epoch(model, loader):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()

    acc = correct / total
    print(f"📘 Train Loss: {total_loss:.4f} | Acc: {acc:.4f}")


# ✅ 5. 평가 함수 (정확도 + 정밀도 + 재현율 + F1)
def evaluate_metrics(model, loader, tag='Test'):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).int().squeeze(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"\n📊 {tag} Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['defect', 'no_defect']))

# ✅ 6. 학습 루프 + 테스트
for epoch in range(10):
    print(f"\n🔁 Epoch {epoch+1}")
    train_one_epoch(model, train_loader)
    evaluate_metrics(model, test_loader, tag="Test")

# ✅ 7. 모델 저장 (선택)
torch.save(model.state_dict(), '/content/drive/MyDrive/resnet18_base_20.pt')

# 🧠 ResNet18 기반 Binary Classification (Accuracy + Precision + Recall + F1 출력)

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import classification_report

# ✅ 1. 경로 & 장치 설정
data_dir = '/content/drive/MyDrive/split_dataset_aug'  # train/, test/ 포함된 폴더
train_dir = os.path.join(data_dir, 'train')
test_dir  = os.path.join(data_dir, 'test')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 2. 전처리 + 데이터 로딩
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(train_dir, transform=transform)
test_data  = datasets.ImageFolder(test_dir, transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=32, shuffle=False)

print(f"✔️ 학습 데이터 수: {len(train_data)}, 테스트 데이터 수: {len(test_data)}")
print(f"✔️ 클래스 매핑: {train_data.class_to_idx}")  # 예: {'defect': 0, 'no_defect': 1}

# ✅ 3. 모델 정의 (ResNet18 + 1 output)
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 1)  # binary classification
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ✅ 4. 학습 함수
def train_one_epoch(model, loader):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()

    acc = correct / total
    print(f"📘 Train Loss: {total_loss:.4f} | Acc: {acc:.4f}")


# ✅ 5. 평가 함수 (정확도 + 정밀도 + 재현율 + F1)
def evaluate_metrics(model, loader, tag='Test'):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).int().squeeze(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"\n📊 {tag} Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['defect', 'no_defect']))

# ✅ 6. 학습 루프 + 테스트
for epoch in range(10):
    print(f"\n🔁 Epoch {epoch+1}")
    train_one_epoch(model, train_loader)
    evaluate_metrics(model, test_loader, tag="Test")

# ✅ 7. 모델 저장 (선택)
torch.save(model.state_dict(), '/content/drive/MyDrive/resnet18_aug_20.pt')

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import classification_report

# ✅ 1. 경로 & 장치 설정
data_dir = '/content/drive/MyDrive/split_dataset'  # train/, test/ 포함된 폴더
train_dir = os.path.join(data_dir, 'train')
test_dir  = os.path.join(data_dir, 'test')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 2. 전처리 + 데이터 로딩
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

#train_data = datasets.ImageFolder(train_dir, transform=transform)
test_data  = datasets.ImageFolder(test_dir, transform=transform)
#train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=32, shuffle=False)

# 모델 로드
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load('/content/drive/MyDrive/resnet18_base.pt'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

y_true = []
y_scores = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()

        y_scores.extend(probs)
        y_true.extend(labels.numpy())

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  # 대각선 기준선
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Test Set)')
plt.legend(loc='lower right')
plt.grid()
plt.show()

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# ✅ 경로 & 장치 설정
data_dir = '/content/drive/MyDrive/split_dataset'
test_dir = os.path.join(data_dir, 'test')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ✅ 테스트 데이터 로딩
test_data = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# ✅ 모델 로드
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load('/content/drive/MyDrive/resnet18_base.pt'))
model = model.to(device)
model.eval()

# ✅ ROC 계산용 예측 수행 (진행 상황 출력)
y_true = []
y_scores = []

print(f"\n🚀 ROC Curve 계산 시작 (총 {len(test_loader)} 배치)\n")

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):
        print(f"  🔄 배치 {batch_idx + 1}/{len(test_loader)} 처리 중...")

        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()

        y_scores.extend(probs)
        y_true.extend(labels.numpy())

# ✅ ROC Curve 및 AUC 계산
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# ✅ 시각화
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  # 대각선 기준선
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Test Set)')
plt.legend(loc='lower right')
plt.grid()
plt.show()

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# ✅ 경로 & 장치 설정
data_dir = '/content/drive/MyDrive/split_dataset'
test_dir = os.path.join(data_dir, 'test')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ✅ 테스트 데이터 로딩
test_data = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# ✅ 모델 로드
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load('/content/drive/MyDrive/resnet18_aug.pt'))
model = model.to(device)
model.eval()

# ✅ ROC 계산용 예측 수행 (진행 상황 출력)
y_true = []
y_scores = []

print(f"\n🚀 ROC Curve 계산 시작 (총 {len(test_loader)} 배치)\n")

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):
        print(f"  🔄 배치 {batch_idx + 1}/{len(test_loader)} 처리 중...")

        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()

        y_scores.extend(probs)
        y_true.extend(labels.numpy())

# ✅ ROC Curve 및 AUC 계산
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# ✅ 시각화
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  # 대각선 기준선
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Test Set)')
plt.legend(loc='lower right')
plt.grid()
plt.show()

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# ✅ 1. 장치 및 경로 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = '/content/drive/MyDrive/split_dataset'
test_dir = os.path.join(data_dir, 'test')
model_path = '/content/drive/MyDrive/resnet18_base.pt'

# ✅ 2. 전처리 및 데이터 로딩
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ✅ 3. 모델 불러오기
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

# ✅ 4. 테스트셋 예측 (확률 및 정답 수집)
y_true = []
y_scores = []

print(f"\n🚀 Precision-Recall Curve 계산 시작 (총 {len(test_loader)} 배치)\n")

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):
        print(f"  🔄 배치 {batch_idx + 1}/{len(test_loader)} 처리 중...")

        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()

        y_scores.extend(probs)
        y_true.extend(labels.numpy())

# ✅ 5. Precision-Recall Curve 계산
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
ap_score = average_precision_score(y_true, y_scores)

# ✅ 6. 시각화
plt.figure()
plt.plot(recall, precision, color='green', lw=2, label=f'PR Curve (AP = {ap_score:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Base Model)')
plt.legend(loc='lower left')
plt.grid()
plt.show()

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# ✅ 1. 장치 및 경로 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = '/content/drive/MyDrive/split_dataset'
test_dir = os.path.join(data_dir, 'test')
model_path = '/content/drive/MyDrive/resnet18_aug.pt'

# ✅ 2. 전처리 및 데이터 로딩
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ✅ 3. 모델 불러오기
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

# ✅ 4. 테스트셋 예측 (확률 및 정답 수집)
y_true = []
y_scores = []

print(f"\n🚀 Precision-Recall Curve 계산 시작 (총 {len(test_loader)} 배치)\n")

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):
        print(f"  🔄 배치 {batch_idx + 1}/{len(test_loader)} 처리 중...")

        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()

        y_scores.extend(probs)
        y_true.extend(labels.numpy())

# ✅ 5. Precision-Recall Curve 계산
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
ap_score = average_precision_score(y_true, y_scores)

# ✅ 6. 시각화
plt.figure()
plt.plot(recall, precision, color='green', lw=2, label=f'PR Curve (AP = {ap_score:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Augmented Model)')
plt.legend(loc='lower left')
plt.grid()
plt.show()
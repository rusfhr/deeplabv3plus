import os
import numpy as np
import argparse

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from Data_loader import CustomDataset   # 커스텀 데이터 셋

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from .model.DeepLabV3Plus import DeepLabV3Plus  # 딥 러닝 모델 코드를 가져옵니다.

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", default = 'D:\\samsung_segmentation\\training_result', type=str)
    parser.add_argument("--data_root", default = 'D:\\samsung_segmentation\\open', type=str)
    parser.add_argument("--device", default = 'cuda:0', type=str)

    return parser.parse_args()

# 하이퍼파라미터 설정
batch_size = 32
num_epochs = 50
lr = 0.001

# 데이터 전처리 및 로더 초기화
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = CustomDataset(root = 'path_to_training_data', transform = transform)
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

# 모델 초기화
model = DeepLabV3Plus(num_classes = 1000)  # num_classes는 클래스 수입니다.

# 옵티마이저 및 손실 함수 설정
optimizer = optim.Adam(model.parameters(), lr = lr)
criterion = nn.CrossEntropyLoss()

# 학습 루프
for epoch in range(num_epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss : {loss.item()}')

# 학습된 모델 저장
torch.save(model.state_dict(), 'path_to_save_model.pth')

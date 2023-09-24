import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from .model.DeepLabV3Plus import DeepLabV3Plus  # 딥 러닝 모델 코드를 가져옵니다.
from Data_loader import CustomDataset  # 데이터셋 및 전처리 코드를 가져옵니다.

# 데이터 전처리 및 로더 초기화
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_dataset = CustomDataset(root='path_to_test_data', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1)

# 저장된 학습된 모델 불러오기
model = DeepLabV3Plus(num_classes = 1000)
model.load_state_dict(torch.load('path_to_saved_model.pth'))
model.eval()  # 모델을 평가 모드로 설정

# 테스트 루프
with torch.no_grad():
    for images, _ in test_loader:
        outputs = model(images)
        # 여기에서 outputs를 사용하여 원하는 후처리 작업을 수행합니다.

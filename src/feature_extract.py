import numpy as np
import os

import torch
import torchvision.models as models
import torchvision.transforms as transforms

preprocessed_data_path='../src/preprocessed_data.npz'
preprocessed_data=np.load(preprocessed_data_path)
X_train=preprocessed_data['X_train']

device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu') #Train on Metal GPU if available, else CPU

model=models.resnet50(pretrained=True)
model=model.to(device)
model=torch.nn.Sequential(*list(model.children())[:-1]) #Remove the classification layer and learn features only

model.eval()

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def extract_features(images):
    features=[]
    with torch.no_grad():
        for img in images:
            img_tensor=transform(img).unsqueeze(0).to(device)
            feature=model(img_tensor)
            features.append(feature.cpu().numpy().flatten())
    return np.array(features)

X_train_features=extract_features(X_train)

extracted_features_path='../models/features'
np.save(extracted_features_path, X_train_features)
print('Features extracted and saved successfully!')
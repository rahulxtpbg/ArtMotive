import numpy as np
import os

import torch
import torchvision.transforms as transforms

from keras.api.models import load_model

from diffusers import StableDiffusionPipeline

preprocessed_data_path='../src/preprocessed_data.npz'
preprocessed_data=np.load(preprocessed_data_path)
y_train=preprocessed_data['y_train']

extracted_features_path='../models/features.npy'
extracted_features=np.load(extracted_features_path)

model_save_path='../models/biased_learning_model.h5'
model=load_model(model_save_path)

pipeline=StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5')
pipeline.to('mps' if torch.backends.mps.is_available() else 'cpu')

unique_labels, counts=np.unique(y_train, return_counts=True) #Unique labels (students) and their counts of submissions
submission_weights={label: count for label, count in zip(unique_labels, counts)} #Map each label to count

total_submissions=sum(submission_weights.values())
normalized_weights={label: submission_weight/total_submissions for label, submission_weight in submission_weights.items()}

feature_vector=np.zeros(extracted_features.shape[1])
for i, label in enumerate(y_train):
    student_weight=normalized_weights[label]
    feature_vector+=student_weight*feature_vector[i]
print('Feature vector computed!')

prompt=input('Input a prompt for artwork generation:')
generated_artwork=pipeline(prompt, guidance_scale=7.5).images[0] #Default guidance_scale. As can be inferred from Hugging Face documentation, fiddle to get image closer to input prompt

artwork_save_path='../outputs/generated_artwork.png'
generated_artwork.save(artwork_save_path)
print('Generated artwork has been saved to path successfully!')
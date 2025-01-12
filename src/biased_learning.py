import numpy as np

from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from keras import Sequential
from keras.src.layers import Dense, Dropout

preprocessed_data_path='../src/preprocessed_data.npz'
extracted_features_path='../models/features.npy'

preprocessed_data=np.load(preprocessed_data_path)
extracted_features=np.load(extracted_features_path)
y_train=preprocessed_data['y_train']

label_encoder=LabelEncoder()
y_train_encoded=label_encoder.fit_transform(y_train) #Encode to numerical

unique_labels, counts=np.unique(y_train, return_counts=True) #Unique labels (students) and their counts of submissions
submission_weights={label: count for label, count in zip(unique_labels, counts)} #Map each label to count

weights=np.array([submission_weights[label] for label in y_train])
extracted_features_weighted=extracted_features*weights[:, np.newaxis]

model=Sequential([
    Dense(128, activation='relu', input_shape=(extracted_features.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(len(unique_labels), activation='softmax')
])

model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(extracted_features_weighted, y_train_encoded, epochs=20, batch_size=32, validation_split=0.2)

model_save_path='../models/biased_learning_model.h5'
model.save(model_save_path)
print('Model trained and saved successfully!')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import os

# SageMaker directories per l'input e l'output
TRAINING_DATA_PATH = 's3://bucket-prova3/input/train.csv'
MODEL_OUTPUT_PATH = 's3://bucket-prova3/output/titanic_model.h5'


np.random.seed(42)
tf.random.set_seed(42)

# Load data
data = pd.read_csv(TRAINING_DATA_PATH)
print("Data loaded successfully.")

# data preprocessing 
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Fare'] = data['Fare'].fillna(data['Fare'].median())
data = data.drop(columns=['Ticket', "Cabin", "PassengerId", "Name"])
label_encoder = LabelEncoder()
data["Sex"] = label_encoder.fit_transform(data["Sex"])
data = pd.get_dummies(data, columns=["Embarked"], drop_first=True)

# Separate features and target
X = data.drop(columns=['Survived'])
y = data['Survived']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# normalizzazione
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Modello 

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])


# compilazione
optimizer = Adam(learning_rate=1e-4, clipnorm=1.0)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# training
history = model.fit(X_train, y_train, epochs=300, batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping],
                    verbose=1)

# Evaluation
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4f}")

# salvataggio del modello
model.save(MODEL_OUTPUT_PATH)
print(f"Model saved to {MODEL_OUTPUT_PATH}")

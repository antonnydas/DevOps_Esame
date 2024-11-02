import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.layers.core import dense

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load data
path = "/Users/eleonoragonzaga/.cache/kagglehub/datasets/harunshimanto/titanic-solution-for-beginners-guide/versions/1"
data = pd.read_csv(f"{path}/train.csv")
test = pd.read_csv(f"{path}/test.csv")

# Fill missing values
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
test['Embarked'] = test['Embarked'].fillna(test['Embarked'].mode()[0])
data['Age'] = data['Age'].fillna(data['Age'].median())
test['Age'] = test['Age'].fillna(test['Age'].median())
data['Fare'] = data['Fare'].fillna(data['Fare'].median())
test['Fare'] = test['Fare'].fillna(test['Fare'].median())

# Drop unnecessary columns
data = data.drop(columns=['Ticket', "Cabin", "PassengerId", "Name"])
test = test.drop(columns=['Ticket', "Cabin", "PassengerId", "Name"])

# Label encoding for 'Sex'
label_encoder = LabelEncoder()
data["Sex"] = label_encoder.fit_transform(data["Sex"])
test["Sex"] = label_encoder.transform(test["Sex"])

# One-Hot Encoding for 'Embarked'
data = pd.get_dummies(data, columns=["Embarked"], drop_first=True)
test = pd.get_dummies(test, columns=["Embarked"], drop_first=True)

# Ensure test set has the same columns as data
missing_cols = set(data.columns) - set(test.columns)
for col in missing_cols:
    test[col] = 0
test = test[data.columns.drop('Survived')]

# Separate features and target
X = data.drop(columns=['Survived'])
y = data['Survived']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define a more complex model with dropout layers for regularization
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),  # Add dropout for regularization
    Dense(32, activation='relu'),
    Dropout(0.3),  # Add another dropout
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model using binary_crossentropy and a smaller learning rate with gradient clipping
optimizer = Adam(learning_rate=1e-4, clipnorm=1.0)  # Adjusted learning rate
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with early stopping
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=300, batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping],
                    verbose=1)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4f}")
# Plot accuracy and loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

model.save('titanic_model.h5')
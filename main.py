

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt


path = "/Users/eleonoragonzaga/.cache/kagglehub/datasets/harunshimanto/titanic-solution-for-beginners-guide/versions/1"
data = pd.read_csv(f"{path}/train.csv")

test = pd.read_csv(f"{path}/test.csv")
print(test.columns)
# Check for missing columns and fill missing values
if 'Embarked' in data.columns:
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
if 'Embarked' in test.columns:
    test['Embarked'] = test['Embarked'].fillna(test['Embarked'].mode()[0])


test = test.drop(columns=['Ticket', "Cabin",  "PassengerId", "Name"])
data = data.drop(columns=['Ticket', "Cabin",  "PassengerId", "Name"])

label_encoder = LabelEncoder()
test["Sex"] = label_encoder.fit_transform(test["Sex"])

X = data.drop(columns=['Survived'])
y = data['Survived']

# 3. Encoding delle variabili categoriche
# Usare Label Encoding per la colonna 'Sex'

data['Sex'] = label_encoder.fit_transform(data['Sex'])

test['Sex'] = label_encoder.fit_transform(test['Sex'])
# Usare One-Hot Encoding per la colonna 'Embarked'

# Encode 'Embarked' only if it exists in both DataFrames
if 'Embarked' in data.columns:
    # Fill missing values if necessary
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
    data = pd.get_dummies(data, columns=["Embarked"], drop_first=True)
else:
    print("Warning: 'Embarked' column not found in the data DataFrame.")

if 'Embarked' in test.columns:
    # Fill missing values if necessary
    test['Embarked'] = test['Embarked'].fillna(test['Embarked'].mode()[0])
    test = pd.get_dummies(test, columns=["Embarked"], drop_first=True)
else:
    print("Warning: 'Embarked' column not found in the test DataFrame.")


# Ensure test set has the same columns as data (align columns)
missing_cols = set(data.columns) - set(test.columns)
for col in missing_cols:
    test[col] = 0  # Add missing columns with default values (e.g., 0)
test = test[data.columns.drop('Survived')]

# Separate features and target
X = data.drop(columns=['Survived'])
y = data['Survived']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Creazione del modello di rete neurale
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compilazione del modello
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Addestramento del modello
history = model.fit(X, y, epochs=50, batch_size=64, validation_split=0.2)



# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

#plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# 1. Crear el modelo secuencial
model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)), # Capa oculta con 64 neuronas
    Dense(32, activation='relu'),                   # Segunda capa oculta
    Dense(1, activation='sigmoid')                  # Capa de salida (ej. clasificación binaria)
])

# 2. Compilar el modelo configurando cómo aprenderá
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Modelo creado y compilado exitosamente.")
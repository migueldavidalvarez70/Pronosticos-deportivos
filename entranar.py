import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import soccerdata as sd
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib

directorio_actual = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# PASO 1: EXTRACCIÓN REAL CON SOCCERDATA
# ==========================================
def obtener_datos_soccerdata():
    print("Descargando historial de partidos y cuotas desde SoccerData (MatchHistory)...")
    # Filtramos grandes ligas europeas y temporadas recientes
    mh = sd.MatchHistory(leagues=['ESP-La Liga', 'ENG-Premier League'], seasons=['2223', '2324'])
    df = mh.read_games()
    
    # Calculamos el promedio móvil de goles anotados por el local y recibidos por el visitante
    df = df.sort_index()
    df['goles_local'] = df.groupby('home_team')['FTHG'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df['goles_concedidos_visita'] = df.groupby('away_team')['FTHG'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    
    # B365H es la cuota de Bet365 para victoria local; FTR == 'H' indica que ganó el Local
    df['cuota_local'] = df['B365H']
    df['resultado'] = (df['FTR'] == 'H').astype(float)
    
    # Limpiamos filas vacías (primeros partidos de la temporada sin historial previo)
    df_limpio = df[['goles_local', 'goles_concedidos_visita', 'cuota_local', 'resultado']].dropna()
    print(f"¡Se cargaron {len(df_limpio)} partidos reales con éxito!")
    return df_limpio

df_partidos = obtener_datos_soccerdata()

# ==========================================
# PASO 2: PREPROCESAMIENTO Y NORMALIZACIÓN
# ==========================================
df_partidos['prob_implicita'] = 1.0 / df_partidos['cuota_local']

X_raw = df_partidos[['goles_local', 'goles_concedidos_visita', 'prob_implicita']].values
y = df_partidos['resultado'].values

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# ==========================================
# PASO 3: ESTRUCTURA DEL MODELO (SIN OVERFITTING)
# ==========================================
# Agregamos Dropout(0.2) para que la IA no memorice y dé probabilidades realistas
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=32, activation='relu', input_shape=[3]),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid') 
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

print("Entrenando a QBETS con cientos de partidos reales...")
model.fit(X, y, epochs=80, batch_size=32, verbose=0)

# ==========================================
# PASO 4: GUARDAR MODELO Y ESCALADOR
# ==========================================
ruta_modelo = os.path.join(directorio_actual, 'qbets_deportivo.keras')
ruta_scaler = os.path.join(directorio_actual, 'scaler_qbets.pkl')

model.save(ruta_modelo)
joblib.dump(scaler, ruta_scaler)
print("✅ ¡Cerebro de QBETS actualizado con datos reales de SoccerData!")
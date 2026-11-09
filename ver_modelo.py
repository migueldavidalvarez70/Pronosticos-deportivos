import tensorflow as tf

# Cambia 'modelo.keras' por el nombre exacto de tu archivo si se llama diferente
nombre_archivo = 'modelo.keras' 

try:
    modelo = tf.keras.models.load_model(nombre_archivo)
    print("\n--- ¡MODELO CARGADO EXITOSAMENTE! ---\n")
    
    # Muestra el resumen de la arquitectura del modelo
    modelo.summary()
    
except Exception as e:
    print(f"No se pudo cargar el archivo '{nombre_archivo}': {e}")
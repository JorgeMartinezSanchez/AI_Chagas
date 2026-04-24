import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
from PIL import Image
import glob

# ============================================================
# 1. CARGAR EL MODELO .H5
# ============================================================
modelo_path = "keras_model.h5"  # Cambia por la ruta de tu archivo
modelo = keras.models.load_model(modelo_path)
print("✅ Modelo cargado exitosamente")
print(f"Arquitectura del modelo: {modelo.summary()}")

# ============================================================
# 2. CARGAR LAS ETIQUETAS (CLASES)
# ============================================================
# Las etiquetas están en el archivo labels.txt que viene con el modelo
with open("labels.txt", "r") as f:
    etiquetas = [linea.strip() for linea in f.readlines()]
print(f"📋 Clases del modelo: {etiquetas}")

# ============================================================
# 3. PREPARAR TUS IMÁGENES DE PRUEBA
# ============================================================
def cargar_y_preprocesar_imagen(ruta_imagen, tamaño=(224, 224)):
    """
    Carga una imagen y la prepara para el modelo.
    Teachable Machine usa imágenes de 224x224 píxeles por defecto.
    """
    imagen = Image.open(ruta_imagen)
    imagen = imagen.resize(tamaño)
    imagen = np.array(imagen, dtype=np.float32)
    
    # Normalizar: Teachable Machine usa valores entre 0 y 1
    # (pero verifica si tu modelo espera 0-1 o 0-255)
    imagen = imagen / 255.0
    
    # Agregar dimensión de batch
    imagen = np.expand_dims(imagen, axis=0)
    return imagen

# ============================================================
# 4. OBTENER PREDICCIONES PARA MÚLTIPLES IMÁGENES
# ============================================================
def predecir_lote(ruta_carpeta_prueba, etiquetas_verdaderas_dict):
    """
    Predice todas las imágenes en una carpeta.
    
    Args:
        ruta_carpeta_prueba: Carpeta con subcarpetas por cada clase
        etiquetas_verdaderas_dict: Diccionario {nombre_carpeta: índice_clase}
    
    Returns:
        y_true: Lista de etiquetas verdaderas
        y_pred: Lista de etiquetas predichas (índices)
        confianzas: Lista de confianzas de la predicción
    """
    y_true = []
    y_pred = []
    confianzas = []
    
    for clase_nombre, clase_idx in etiquetas_verdaderas_dict.items():
        carpeta_clase = os.path.join(ruta_carpeta_prueba, clase_nombre)
        if not os.path.exists(carpeta_clase):
            print(f"⚠️ Carpeta no encontrada: {carpeta_clase}")
            continue
            
        imagenes = glob.glob(os.path.join(carpeta_clase, "*.*"))
        print(f"📁 Procesando clase '{clase_nombre}': {len(imagenes)} imágenes")
        
        for img_path in imagenes:
            try:
                img = cargar_y_preprocesar_imagen(img_path)
                prediccion = modelo.predict(img, verbose=0)
                clase_predicha = np.argmax(prediccion[0])
                confianza = np.max(prediccion[0])
                
                y_true.append(clase_idx)
                y_pred.append(clase_predicha)
                confianzas.append(confianza)
            except Exception as e:
                print(f"❌ Error procesando {img_path}: {e}")
    
    return y_true, y_pred, confianzas

# ============================================================
# 5. EJEMPLO DE USO
# ============================================================
# Supongamos que tienes esta estructura de carpetas:
# datos_prueba/
#   ├── vector/
#   │   ├── imagen1.jpg
#   │   └── imagen2.jpg
#   └── no_vector/
#       ├── imagen3.jpg
#       └── imagen4.jpg

# Mapea los nombres de tus carpetas a los índices de las etiquetas
# Esto debe coincidir con el orden en labels.txt
mapeo_clases = {
    "vector": 0,      # Suponiendo que labels.txt[0] = "vector"
    "no_vector": 1,   # Suponiendo que labels.txt[1] = "no_vector"
}

# Ejecutar predicciones
y_true, y_pred, confianzas = predecir_lote("datos_prueba/", mapeo_clases)

# ============================================================
# 6. GENERAR MATRIZ DE CONFUSIÓN
# ============================================================
# Calcular matriz
cm = confusion_matrix(y_true, y_pred)

# Visualizar con seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=etiquetas, yticklabels=etiquetas)
plt.title('Matriz de Confusión - Modelo Teachable Machine')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.tight_layout()
plt.savefig('matriz_confusion.png', dpi=300)
plt.show()

# ============================================================
# 7. MÉTRICAS DETALLADAS
# ============================================================
print("\n" + "="*50)
print("📊 REPORTE DE CLASIFICACIÓN")
print("="*50)
print(classification_report(y_true, y_pred, target_names=etiquetas))

# Métricas adicionales
exactitud = np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)
print(f"\n🎯 Exactitud global: {exactitud:.4f} ({exactitud*100:.2f}%)")

# Confianza promedio por clase
for i, clase in enumerate(etiquetas):
    confianzas_clase = [confianzas[j] for j in range(len(y_true)) if y_pred[j] == i]
    if confianzas_clase:
        print(f"Confianza promedio para {clase}: {np.mean(confianzas_clase):.4f}")
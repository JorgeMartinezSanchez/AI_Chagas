import tf_keras as keras
from tf_keras.models import load_model
from tf_keras.layers import DepthwiseConv2D
from PIL import Image, ImageOps
import numpy as np

np.set_printoptions(suppress=True)

class FixedDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)

# Cargar el modelo y las etiquetas (solo una vez al iniciar)
model = load_model(
    "keras_model.h5",
    compile=False,
    custom_objects={"DepthwiseConv2D": FixedDepthwiseConv2D}
)

class_names = open("labels.txt", "r").readlines()

def predecir_imagen(image_path):
    """
    Función que recibe la ruta de una imagen y retorna la predicción
    
    Args:
        image_path (str): Ruta de la imagen a analizar
    
    Returns:
        tuple: (nombre_clase, confianza) o None si hay error
    """
    try:
        # Crear el array para la predicción
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        
        # Usar el parámetro image_path en lugar de la ruta fija
        image = Image.open(image_path).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        
        # Convertir a array y normalizar
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array
        
        # Predecir
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]
        
        # Limpiar el nombre (quitar el número inicial si existe)
        if class_name and len(class_name) > 2 and class_name[0].isdigit():
            class_name = class_name[2:].strip()
        
        return (class_name, confidence_score)
        
    except Exception as e:
        print(f"Error al predecir: {e}")
        return None

# Ya no necesitas el código que estaba al final
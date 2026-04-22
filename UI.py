import run_model_0
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import threading
import tempfile
import os

class UI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.configurar_ventana()
        self.crear_widgets()
        self.ruta_imagen_temp = None
    
    def configurar_ventana(self):
        """Configura las propiedades de la ventana"""
        self.title("Detector de Insecto Chagas")
        self.geometry("650x700")
        self.minsize(550, 600)
        self.configure(bg="#f0f0f0")
        
        # Centrar la ventana en la pantalla
        self.centrar_ventana()

    def centrar_ventana(self):
        """Centra la ventana en la pantalla"""
        self.update_idletasks()
        ancho = self.winfo_width()
        alto = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (ancho // 2)
        y = (self.winfo_screenheight() // 2) - (alto // 2)
        self.geometry(f'{ancho}x{alto}+{x}+{y}')

    def crear_widgets(self):
        """Crea y organiza los widgets"""
        # Título
        titulo = tk.Label(
            self, 
            text="Detector de Insecto Chagas", 
            font=("Arial", 20, "bold"),
            bg="#f0f0f0",
            fg="#333333"
        )
        titulo.pack(pady=20)

        # Frame para los botones superiores (importar y cámara)
        frame_botones_superior = tk.Frame(self, bg="#f0f0f0")
        frame_botones_superior.pack(pady=10)

        btn_take_a_photo = tk.Button(
            frame_botones_superior,
            text="📸 Tomar foto",
            command=self.tomar_foto_camara,
            bg="#4CAF50",
            fg="white",
            padx=20,
            pady=10,
            font=("Arial", 11, "bold"),
            cursor="hand2"
        )
        btn_take_a_photo.pack(side=tk.LEFT, padx=10)

        btn_import_a_photo = tk.Button(
            frame_botones_superior,
            text="🏞️ Importar foto",
            command=self.importar_imagen,
            bg="#2196F3",
            fg="white",
            padx=20,
            pady=10,
            font=("Arial", 11, "bold"),
            cursor="hand2"
        )
        btn_import_a_photo.pack(side=tk.LEFT, padx=10)

        # Label para mostrar la imagen
        self.label_imagen = tk.Label(
            self, 
            bg="#e0e0e0", 
            relief="sunken",
            width=500,
            height=300
        )
        self.label_imagen.pack(pady=20, padx=20)

        # Frame para mostrar resultados
        frame_resultado = tk.Frame(self, bg="#f0f0f0")
        frame_resultado.pack(pady=15)

        # Etiqueta para el resultado
        self.label_resultado = tk.Label(
            frame_resultado,
            text="📷 Esperando imagen...",
            font=("Arial", 16, "bold"),
            bg="#f0f0f0",
            fg="#666666"
        )
        self.label_resultado.pack()

        # Etiqueta para la confianza
        self.label_confianza = tk.Label(
            frame_resultado,
            text="",
            font=("Arial", 12),
            bg="#f0f0f0",
            fg="#888888"
        )
        self.label_confianza.pack(pady=5)

        # Barra de progreso (para mostrar cuando está procesando)
        self.progreso = tk.Label(
            frame_resultado,
            text="",
            font=("Arial", 10),
            bg="#f0f0f0",
            fg="#2196F3"
        )
        self.progreso.pack()

        # Botón de salida
        btn_salir = tk.Button(
            self,
            text="Uscita",
            command=self.limpiar_y_salir,
            bg="#f44336",
            fg="white",
            padx=30,
            pady=10,
            font=("Arial", 11, "bold"),
            cursor="hand2"
        )
        btn_salir.pack(pady=20)

    def procesar_imagen_con_modelo(self, ruta_imagen):
        """Procesa la imagen con el modelo de Teachable Machine"""
        try:
            # Actualizar UI para mostrar que está procesando
            self.progreso.config(text="🔄 Procesando imagen con el modelo...")
            self.label_resultado.config(text="Analizando...", fg="#FF9800")
            self.update()
            
            # Importar las funciones necesarias del modelo
            from run_model_0 import model, class_names
            
            # Crear el array para la predicción
            import numpy as np
            from PIL import Image, ImageOps
            
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            
            # Abrir y preparar la imagen
            image = Image.open(ruta_imagen).convert("RGB")
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
            
            # Limpiar el nombre de la clase (quitar el número inicial)
            if class_name and len(class_name) > 2 and class_name[0].isdigit():
                class_name = class_name[2:].strip()
            
            # Actualizar UI con los resultados
            self.progreso.config(text="✅ Análisis completado")
            
            if class_name.lower() == "chagas":
                self.label_resultado.config(
                    text=f"⚠️ ¡DETECTADO: {class_name.upper()}!",
                    fg="#f44336"
                )
                # Mostrar alerta si la confianza es alta
                if confidence_score > 0.7:
                    messagebox.showwarning(
                        "Alerta Sanitaria",
                        f"⚠️ ¡POSITIVO PARA CHAGAS!\n\n"
                        f"Resultado: {class_name}\n"
                        f"Confianza: {confidence_score:.2%}\n\n"
                        f"Por favor, consulta a un médico."
                    )
            else:
                self.label_resultado.config(
                    text=f"✅ Resultado: {class_name}",
                    fg="#4CAF50"
                )
            
            self.label_confianza.config(
                text=f"Confianza: {confidence_score:.2%}",
                fg="#2196F3"
            )
            
        except Exception as e:
            self.progreso.config(text="❌ Error en el procesamiento")
            self.label_resultado.config(text="Error al procesar", fg="#f44336")
            self.label_confianza.config(text="")
            messagebox.showerror("Error", f"Error al procesar la imagen:\n{str(e)}")

    def mostrar_imagen_y_procesar(self, ruta_imagen):
        """Muestra la imagen en la interfaz y la procesa con el modelo"""
        try:
            # Abrir y mostrar la imagen
            img = Image.open(ruta_imagen)
            
            # Redimensionar para mostrar (manteniendo proporción)
            img_copy = img.copy()
            img_copy.thumbnail((480, 280), Image.Resampling.LANCZOS)
            
            self.imagen_actual = ImageTk.PhotoImage(img_copy)
            self.label_imagen.config(image=self.imagen_actual)
            
            # Procesar la imagen con el modelo en un hilo separado
            # para no congelar la interfaz
            threading.Thread(
                target=self.procesar_imagen_con_modelo,
                args=(ruta_imagen,),
                daemon=True
            ).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar la imagen:\n{str(e)}")

    def importar_imagen(self):
        """Importa una imagen desde el disco duro"""
        ruta = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[
                ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("Todos los archivos", "*.*")
            ]
        )
        if ruta:
            # Resetear resultados anteriores
            self.label_resultado.config(text="📷 Procesando imagen...", fg="#FF9800")
            self.label_confianza.config(text="")
            self.progreso.config(text="")
            
            self.mostrar_imagen_y_procesar(ruta)

    def tomar_foto_camara(self):
        """Toma una foto con la cámara y la procesa"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "No se pudo abrir la cámara")
            return
        
        # Mostrar mensaje de espera
        self.label_resultado.config(text="📸 Tomando foto...", fg="#FF9800")
        self.label_confianza.config(text="")
        self.progreso.config(text="")
        self.update()
        
        # Pequeña pausa para que la cámara se estabilice
        import time
        time.sleep(0.5)
        
        # Capturar foto
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Guardar la foto en un archivo temporal
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            ruta_temp = temp_file.name
            temp_file.close()
            
            cv2.imwrite(ruta_temp, frame)
            self.ruta_imagen_temp = ruta_temp
            
            self.mostrar_imagen_y_procesar(ruta_temp)
        else:
            messagebox.showerror("Error", "No se pudo capturar la foto")
            self.label_resultado.config(text="Error al capturar foto", fg="#f44336")
    
    def limpiar_y_salir(self):
        """Limpia archivos temporales y cierra la aplicación"""
        if self.ruta_imagen_temp and os.path.exists(self.ruta_imagen_temp):
            try:
                os.unlink(self.ruta_imagen_temp)
            except:
                pass
        self.destroy()

    def ejecutar(self):
        """Inicia el bucle principal"""
        self.mainloop()
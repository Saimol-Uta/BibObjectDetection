# Manual de Instalación y Ejecución - Detección de Números de Dorsal

## Requisitos del Sistema

### Hardware
- **GPU**: NVIDIA GeForce RTX 3050 (✓ Confirmado)
- **RAM**: Mínimo 8 GB (Recomendado 16 GB)
- **Almacenamiento**: Mínimo 10 GB de espacio libre
- **Sistema Operativo**: Windows 10/11

### Software Base
- Windows 10 o superior
- NVIDIA GeForce RTX 3050 con drivers actualizados

---

## Paso 1: Instalación de Drivers NVIDIA

1. **Descargar NVIDIA Drivers más recientes**:
   - Visita: https://www.nvidia.com/Download/index.aspx
   - Selecciona:
     - Product Type: GeForce
     - Product Series: GeForce RTX 30 Series
     - Product: GeForce RTX 3050
     - Operating System: Windows 10/11 64-bit
   - Descarga e instala el driver más reciente

2. **Verificar instalación**:
   ```powershell
   nvidia-smi
   ```
   Deberías ver información sobre tu GPU RTX 3050.

---

## Paso 2: Instalación de CUDA Toolkit

1. **Descargar CUDA Toolkit 11.8** (compatible con RTX 3050):
   - Visita: https://developer.nvidia.com/cuda-11-8-0-download-archive
   - Selecciona: Windows > x86_64 > 10/11 > exe (local)
   - Descarga el instalador (aproximadamente 3 GB)

2. **Instalar CUDA**:
   - Ejecuta el instalador descargado
   - Selecciona instalación "Express" (recomendado)
   - Espera a que complete la instalación (puede tomar 10-15 minutos)

3. **Verificar instalación**:
   ```powershell
   nvcc --version
   ```
   Deberías ver la versión de CUDA instalada (11.8).

---

## Paso 3: Instalación de cuDNN

1. **Descargar cuDNN**:
   - Visita: https://developer.nvidia.com/cudnn
   - Necesitas crear una cuenta NVIDIA Developer (gratuita)
   - Descarga: cuDNN v8.6.0 para CUDA 11.x
   - Descarga el archivo ZIP para Windows

2. **Instalar cuDNN**:
   - Extrae el archivo ZIP descargado
   - Copia los archivos a la carpeta de CUDA:
     ```powershell
     # Asumiendo que CUDA está en: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
     # Copia los archivos de la carpeta 'bin' del ZIP a:
     # C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
     
     # Copia los archivos de la carpeta 'include' del ZIP a:
     # C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include
     
     # Copia los archivos de la carpeta 'lib' del ZIP a:
     # C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64
     ```

3. **Agregar a PATH** (si no está ya):
   ```powershell
   $env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"
   $env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp"
   ```

---

## Paso 4: Instalación de Python y Entorno Virtual

1. **Instalar Python 3.8 o 3.9** (recomendado para compatibilidad):
   - Descarga desde: https://www.python.org/downloads/
   - Durante la instalación, marca "Add Python to PATH"
   - Verifica la instalación:
     ```powershell
     python --version
     ```

2. **Crear entorno virtual en el proyecto**:
   ```powershell
   cd "d:\Univercidad\ModeloDetecion\BibObjectDetection"
   python -m venv venv
   ```

3. **Activar el entorno virtual**:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
   
   **Nota**: Si obtienes un error de ejecución de scripts, ejecuta primero:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

---

## Paso 5: Instalación de Dependencias de Python

1. **Con el entorno virtual activado**, instala las dependencias:

   ```powershell
   # Actualizar pip
   python -m pip install --upgrade pip
   
   # Instalar PyTorch con soporte CUDA para RTX 3050
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # Instalar OpenCV
   pip install opencv-python opencv-contrib-python
   
   # Instalar otras dependencias del proyecto
   pip install h5py numpy matplotlib scipy pandas imgaug jupyter notebook ipython
   ```

2. **Verificar que PyTorch detecta la GPU**:
   ```powershell
   python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
   ```
   Debería mostrar:
   ```
   CUDA disponible: True
   GPU: NVIDIA GeForce RTX 3050
   ```

---

## Paso 6: Compilar Darknet para Windows con GPU

### Opción A: Descargar Binario Precompilado (Más Fácil)

1. **Descargar Darknet precompilado**:
   - Visita: https://github.com/AlexeyAB/darknet/releases
   - Descarga: `darknet_yolo_v4_pre_release.zip` o similar
   - Extrae en una carpeta, por ejemplo: `C:\darknet`

2. **Configurar PATH**:
   ```powershell
   $env:PATH += ";C:\darknet"
   ```

### Opción B: Compilar desde Código Fuente (Avanzado)

1. **Instalar Visual Studio 2019 o 2022**:
   - Descarga Visual Studio Community (gratuito)
   - Durante instalación, selecciona "Desktop development with C++"

2. **Clonar y compilar Darknet**:
   ```powershell
   cd C:\
   git clone https://github.com/AlexeyAB/darknet.git
   cd darknet
   ```

3. **Editar archivo de configuración** `build\darknet\x64\darknet.vcxproj`:
   - Abre con Visual Studio
   - En las propiedades del proyecto:
     - GPU = 1
     - CUDNN = 1
     - OPENCV = 1 (opcional pero recomendado)

4. **Compilar**:
   - Abre el proyecto en Visual Studio
   - Selecciona "Release" y "x64"
   - Build > Build Solution

---

## Paso 7: Descargar Datasets (Opcional - Solo para Entrenamiento)

Si solo quieres ejecutar los modelos ya entrenados, **puedes omitir este paso**.

### Dataset SVHN (Street View House Numbers)
```powershell
# Crear carpeta para datasets
mkdir "d:\Univercidad\ModeloDetecion\BibObjectDetection\datasets"
cd "d:\Univercidad\ModeloDetecion\BibObjectDetection\datasets"

# Descargar SVHN
# Visitar: http://ufldl.stanford.edu/housenumbers/
# Descargar manualmente:
# - train_32x32.mat
# - test_32x32.mat
# - extra_32x32.mat
```

### Dataset RBNR (Racing Bib Number Recognition)
```powershell
# Visitar: https://people.csail.mit.edu/talidekel/RBNR.html
# Descargar el dataset manualmente y extraer en:
# d:\Univercidad\ModeloDetecion\BibObjectDetection\datasets\RBNR
```

---

## Paso 8: Configurar y Ejecutar Jupyter Notebooks

1. **Iniciar Jupyter Notebook** (con el entorno virtual activado):
   ```powershell
   cd "d:\Univercidad\ModeloDetecion\BibObjectDetection\notebooks+utils+data"
   jupyter notebook
   ```

2. **Se abrirá tu navegador** con la interfaz de Jupyter

3. **Ejecutar notebooks en orden**:

   ### Para entrenar modelos (requiere datasets):
   - `01 - Prepocessing & Training SVHN YOLOv4-tiny Darknet.ipynb`
   - `03 - Preprocessing Racing Bib Numbers (RBNR) Datasets.ipynb`
   - `04 - Run Yolov4 Tiny on RBNR Data.ipynb`

   ### Para validar y hacer demos (usa pesos pre-entrenados):
   - `02 - Digit Detection Validation Using RBNR Data.ipynb`
   - `05 - Bib Detection Validation & Demo.ipynb` ⭐ **Recomendado para empezar**

---

## Paso 9: Ejecutar Detección en Imágenes/Videos

### Usando el Notebook de Demo (Más Fácil)

1. Abre `05 - Bib Detection Validation & Demo.ipynb`
2. Ejecuta las celdas secuencialmente
3. El notebook cargará los pesos pre-entrenados de la carpeta `weights-classes/`
4. Procesará las imágenes y videos de ejemplo

### Usando Script Python Personalizado

Crea un archivo `detect_bib.py` en la carpeta del proyecto:

```python
import cv2
import numpy as np
from utils import *

# Configuración
weights_path = "../weights-classes/RBNR_custom-yolov4-tiny-detector_best.weights"
config_path = "../weights-classes/RBNR_custom-yolov4-tiny-detector.cfg"
names_path = "../weights-classes/RBRN_obj.names"

# Cargar modelo
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Cargar nombres de clases
with open(names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Cargar imagen
image_path = "tu_imagen.jpg"  # Cambia esto
image = cv2.imread(image_path)

# Realizar detección
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
outputs = net.forward(output_layers)

# Procesar detecciones
height, width = image.shape[:2]
boxes = []
confidences = []
class_ids = []

for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Non-maximum suppression
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Dibujar detecciones
for i in indices:
    i = i if isinstance(i, int) else i[0]
    box = boxes[i]
    x, y, w, h = box
    label = str(classes[class_ids[i]])
    confidence = confidences[i]
    
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Mostrar resultado
cv2.imshow("Deteccion de Dorsales", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Guardar resultado
cv2.imwrite("resultado_deteccion.jpg", image)
print("Detección completada. Resultado guardado en resultado_deteccion.jpg")
```

Ejecutar:
```powershell
python detect_bib.py
```

---

## Paso 10: Ejecutar Detección en Video

Para procesar el video de ejemplo:

```python
import cv2
import numpy as np

# Configuración
weights_path = "../weights-classes/RBNR_custom-yolov4-tiny-detector_best.weights"
config_path = "../weights-classes/RBNR_custom-yolov4-tiny-detector.cfg"
names_path = "../weights-classes/RBRN_obj.names"
video_path = "VIDEO0433.mp4"
output_path = "output_deteccion.mp4"

# Cargar modelo
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Cargar video
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Configurar video de salida
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    print(f"Procesando frame {frame_count}...")
    
    # Realizar detección (mismo código que para imagen)
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)
    
    # [Agregar código de procesamiento de detecciones aquí...]
    
    # Escribir frame procesado
    out.write(frame)

cap.release()
out.release()
print(f"Video procesado guardado en: {output_path}")
```

---

## Solución de Problemas Comunes

### Error: "CUDA out of memory"
- **Solución**: Reduce el tamaño del batch o la resolución de entrada en los archivos `.cfg`
- En el archivo `.cfg`, busca `batch=64` y cámbialo a `batch=32` o `batch=16`

### Error: "Could not find CUDA"
- Verifica que `nvidia-smi` funciona
- Reinstala los drivers NVIDIA
- Verifica que las rutas de CUDA estén en PATH

### Error: "DLL load failed" al importar módulos
- Asegúrate de que todas las DLLs de CUDA estén en PATH
- Reinstala Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe

### Rendimiento lento
- Verifica que OpenCV esté usando GPU:
  ```python
  print(cv2.cuda.getCudaEnabledDeviceCount())  # Debe ser > 0
  ```
- Asegúrate de que no hay otros procesos usando la GPU
- Usa `nvidia-smi` para monitorear uso de GPU

### El modelo no detecta bien
- Ajusta el umbral de confianza (threshold) en el código
- Prueba con diferentes valores entre 0.3 y 0.7
- Asegúrate de estar usando los pesos correctos (RBNR para dorsales, SVHN para dígitos)

---

## Optimizaciones para RTX 3050

Tu RTX 3050 tiene 8 GB de VRAM, lo cual es suficiente para este proyecto. Recomendaciones:

1. **Usar Mixed Precision** (FP16) para mejor rendimiento:
   - En los scripts de detección, agrega:
   ```python
   net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
   net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)  # Usar FP16
   ```

2. **Batch Size óptimo**:
   - Para entrenamiento: batch=16 o batch=32
   - Para inferencia: puedes procesar múltiples imágenes a la vez

3. **Monitorear uso de GPU**:
   ```powershell
   # En otra terminal
   nvidia-smi -l 1  # Actualiza cada segundo
   ```

---

## Estructura Final del Proyecto

```
BibObjectDetection/
├── venv/                          # Entorno virtual
├── notebooks+utils+data/          # Notebooks y utilidades
│   ├── *.ipynb                    # Notebooks Jupyter
│   ├── utils.py                   # Funciones auxiliares
│   ├── VIDEO0433.mp4              # Video de ejemplo
│   └── BibDetectorSample.jpeg     # Imagen de ejemplo
├── weights-classes/               # Modelos pre-entrenados
│   ├── RBNR_custom-*.weights      # Pesos para dorsales
│   ├── SVHN_custom-*.weights      # Pesos para dígitos
│   └── *.cfg                      # Configuraciones
├── datasets/                      # Datasets (opcional)
└── presentation/                  # Presentación del proyecto
```

---

## Comandos Rápidos de Referencia

```powershell
# Activar entorno virtual
cd "d:\Univercidad\ModeloDetecion\BibObjectDetection"
.\venv\Scripts\Activate.ps1

# Iniciar Jupyter
cd "notebooks+utils+data"
jupyter notebook

# Verificar GPU
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Monitorear GPU durante ejecución
nvidia-smi -l 1
```

---

## Recursos Adicionales

- **Darknet GitHub**: https://github.com/AlexeyAB/darknet
- **YOLO Documentation**: https://pjreddie.com/darknet/yolo/
- **OpenCV DNN Module**: https://docs.opencv.org/master/d2/d58/tutorial_table_of_content_dnn.html
- **CUDA Programming Guide**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/

---

## Soporte y Contacto

Para problemas específicos del proyecto:
- GitHub Issues: https://github.com/Lwhieldon/BibObjectDetection/issues
- Autor Original: Lee Whieldon - https://github.com/Lwhieldon

---

## Licencia

Ver archivo `LICENSE` en el repositorio.

---

**¡Listo!** Ahora deberías poder ejecutar el proyecto de detección de números de dorsal en tu PC con NVIDIA RTX 3050. Comienza con el notebook `05 - Bib Detection Validation & Demo.ipynb` para ver resultados rápidamente usando los pesos pre-entrenados.

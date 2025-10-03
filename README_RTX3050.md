# 🏃 Detección de Números de Dorsal - Guía para PC con RTX 3050

<p align="center">
<img src="https://github.com/Lwhieldon/BibObjectDetection/blob/main/notebooks+utils+data/BibDetectorSample.jpeg?raw=true" height=400 />
</p>

## 📋 Descripción del Proyecto

Este proyecto utiliza **YOLOv4-tiny** con **NVIDIA CUDA** para detectar números de dorsal en imágenes y videos de carreras. Optimizado para ejecutarse en **NVIDIA GeForce RTX 3050**.

### 🎯 Resultados
- ✨ **99% mAP** en detección de dorsales (Dataset RBNR)
- ✨ **96% mAP** en detección de dígitos (Dataset SVHN)
- ⚡ **Procesamiento en tiempo real** con GPU RTX 3050

---

## 🚀 Inicio Rápido (3 pasos)

### 1️⃣ Instalar Pre-requisitos

- **Drivers NVIDIA**: [Descargar aquí](https://www.nvidia.com/Download/index.aspx)
- **CUDA 11.8**: [Descargar aquí](https://developer.nvidia.com/cuda-11-8-0-download-archive)
- **cuDNN 8.6**: [Descargar aquí](https://developer.nvidia.com/cudnn) (requiere cuenta gratuita)
- **Python 3.8-3.10**: [Descargar aquí](https://www.python.org/downloads/)

### 2️⃣ Ejecutar Instalación Automática

```powershell
# En PowerShell, en la carpeta del proyecto
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\instalar.ps1
```

### 3️⃣ Ejecutar el Demo

```powershell
.\venv\Scripts\Activate.ps1
cd "notebooks+utils+data"
jupyter notebook
# Abre: 05 - Bib Detection Validation & Demo.ipynb
```

---

## 📚 Documentación

### Guías de Instalación
- 📖 **[INICIO_RAPIDO.md](INICIO_RAPIDO.md)** - Guía de inicio rápido (15 min)
- 📖 **[MANUAL_INSTALACION.md](MANUAL_INSTALACION.md)** - Manual completo paso a paso

### Scripts Útiles
- ⚙️ **instalar.ps1** - Instalación automática de dependencias
- 🔍 **verificar_instalacion.py** - Verifica que todo funcione correctamente
- 📋 **requirements.txt** - Lista completa de dependencias Python

---

## 📁 Estructura del Proyecto

```
BibObjectDetection/
│
├── 📖 README.md                     # Este archivo
├── 📖 INICIO_RAPIDO.md              # Guía rápida
├── 📖 MANUAL_INSTALACION.md         # Manual completo
├── ⚙️ instalar.ps1                  # Instalador automático
├── 🔍 verificar_instalacion.py     # Script de verificación
├── 📋 requirements.txt              # Dependencias
│
├── notebooks+utils+data/
│   ├── 🎯 05 - Bib Detection Validation & Demo.ipynb          ⭐ EMPEZAR AQUÍ
│   ├── 01 - Prepocessing & Training SVHN YOLOv4-tiny Darknet.ipynb
│   ├── 02 - Digit Detection Validation Using RBNR Data.ipynb
│   ├── 03 - Preprocessing Racing Bib Numbers (RBNR) Datasets.ipynb
│   ├── 04 - Run Yolov4 Tiny on RBNR Data.ipynb
│   ├── utils.py                     # Funciones auxiliares
│   ├── ejemplo_deteccion_imagen.py  # Script de ejemplo
│   ├── VIDEO0433.mp4                # Video de prueba
│   ├── BibDetectorSample.jpeg       # Imagen de prueba
│   ├── marathon_output.gif          # Demo animado
│   └── output_marathon.mp4          # Video procesado
│
├── weights-classes/
│   ├── RBNR_custom-yolov4-tiny-detector_best.weights  # Modelo dorsales
│   ├── RBNR_custom-yolov4-tiny-detector.cfg
│   ├── RBRN_obj.names
│   ├── SVHN_custom-yolov4-tiny-detector_best.weights  # Modelo dígitos
│   ├── SVHN_custom-yolov4-tiny-detector.cfg
│   └── SVHN_obj.names
│
└── presentation/
    └── RaceBibDetection_Presentation.pdf
```

---

## 🎮 Demo en Video

![gif](https://github.com/Lwhieldon/BibObjectDetection/blob/main/notebooks+utils+data/marathon_output.gif)

---

## 💻 Especificaciones de Hardware

### Configuración Recomendada
- **GPU**: NVIDIA GeForce RTX 3050 (8 GB VRAM) ✅
- **RAM**: 8 GB mínimo (16 GB recomendado)
- **Almacenamiento**: 10 GB libres
- **Sistema**: Windows 10/11 64-bit

### Tu RTX 3050 es Perfecta para:
- ✅ Procesamiento en tiempo real de video
- ✅ Entrenamiento de modelos (con batch sizes ajustados)
- ✅ Inferencia con precisión FP16 para mejor rendimiento
- ✅ Procesamiento de múltiples streams simultáneos

---

## 🔧 Uso del Proyecto

### Opción 1: Jupyter Notebooks (Recomendado para principiantes)

```powershell
# Activar entorno
.\venv\Scripts\Activate.ps1

# Iniciar Jupyter
cd "notebooks+utils+data"
jupyter notebook
```

**Notebooks disponibles:**
1. `01 - Prepocessing & Training SVHN YOLOv4-tiny Darknet.ipynb` - Entrenar con dígitos
2. `02 - Digit Detection Validation Using RBNR Data.ipynb` - Validar dígitos
3. `03 - Preprocessing Racing Bib Numbers (RBNR) Datasets.ipynb` - Preparar datos dorsales
4. `04 - Run Yolov4 Tiny on RBNR Data.ipynb` - Entrenar dorsales
5. `05 - Bib Detection Validation & Demo.ipynb` - **Demo completo** ⭐

### Opción 2: Script Python (Avanzado)

```powershell
# Activar entorno
.\venv\Scripts\Activate.ps1

cd "notebooks+utils+data"

# Detectar dorsales en imagen
python ejemplo_deteccion_imagen.py --imagen tu_imagen.jpg --modelo RBNR

# Detectar dígitos en imagen
python ejemplo_deteccion_imagen.py --imagen tu_imagen.jpg --modelo SVHN --umbral 0.6

# Ver opciones
python ejemplo_deteccion_imagen.py --help
```

**Opciones disponibles:**
- `--imagen`: Ruta a tu imagen
- `--salida`: Ruta para guardar resultado
- `--modelo`: RBNR (dorsales) o SVHN (dígitos)
- `--umbral`: Confianza mínima (0.0-1.0)
- `--cpu`: Forzar uso de CPU

---

## 📊 Datasets Utilizados

### SVHN (Street View House Numbers)
- **Fuente**: [Stanford University](http://ufldl.stanford.edu/housenumbers)
- **Uso**: Entrenamiento de detección de dígitos
- **Resultado**: 96% mAP

### RBNR (Racing Bib Number Recognition)
- **Fuente**: [MIT CSAIL](https://people.csail.mit.edu/talidekel/RBNR.html)
- **Uso**: Entrenamiento de detección de dorsales
- **Resultado**: 99% mAP

---

## 🛠️ Tecnologías Utilizadas

- **Deep Learning**: YOLOv4-tiny, Darknet
- **GPU Acceleration**: NVIDIA CUDA 11.8, cuDNN 8.6
- **Computer Vision**: OpenCV con soporte CUDA
- **Framework**: PyTorch con CUDA
- **Procesamiento**: NumPy, Pandas, SciPy
- **Visualización**: Matplotlib, imgaug
- **Ambiente**: Jupyter Notebook

---

## ⚡ Optimizaciones para RTX 3050

### 1. Usar Precisión FP16
```python
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
```

### 2. Ajustar Batch Size
En archivos `.cfg`:
```
batch=16  # Óptimo para 8GB VRAM
subdivisions=8
```

### 3. Monitorear GPU
```powershell
nvidia-smi -l 1  # Actualiza cada segundo
```

---

## ❓ Solución de Problemas

### Error: "CUDA out of memory"
```
Solución: Reduce batch size en .cfg
batch=8 o batch=4
```

### Error: "nvidia-smi no encontrado"
```
Solución: Reinstala drivers NVIDIA
https://www.nvidia.com/Download/index.aspx
```

### PyTorch no detecta GPU
```powershell
# Reinstalar PyTorch con CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Más problemas?
Consulta [MANUAL_INSTALACION.md](MANUAL_INSTALACION.md) sección "Solución de Problemas Comunes"

---

## 📈 Rendimiento Esperado en RTX 3050

| Tarea | FPS (aprox) | Resolución |
|-------|-------------|------------|
| Detección en imagen | ~40 FPS | 416x416 |
| Detección en video | ~30 FPS | 720p |
| Entrenamiento | ~15 img/s | batch=16 |

---

## 📺 Recursos Adicionales

- **Video Presentación**: https://youtu.be/xfVfr0KmhYY
- **GitHub Original**: https://github.com/Lwhieldon/BibObjectDetection
- **Darknet**: https://github.com/AlexeyAB/darknet
- **YOLO Documentation**: https://pjreddie.com/darknet/yolo/

---

## 📝 Referencias

- A. Apap and D. Seychell, "Marathon bib number recognition using deep learning," ISPA 2019
- E. Ivarsson and R. M. Mueller, "Racing bib number recognition using deep learning," 2019
- P. Hernández-Carrascosa et al., "TGCRBNW: A Dataset for Runner Bib Number Detection," ICPR 2020
- RoboFlow: https://blog.roboflow.com/train-yolov4-tiny-on-custom-data-lighting-fast-detection/

---

## 👨‍💻 Autor Original

**Lee Whieldon**
- GitHub: [@Lwhieldon](https://github.com/Lwhieldon)
- Proyecto: UMBC Data Science Program - DATA690: Applied AI
- Fecha: Mayo 2022

---

## 📄 Licencia

Ver archivo [LICENSE](LICENSE)

---

## ✅ Checklist de Inicio

- [ ] Drivers NVIDIA instalados
- [ ] CUDA 11.8 instalado
- [ ] cuDNN extraído
- [ ] Python 3.8+ instalado
- [ ] Ejecutado `instalar.ps1`
- [ ] Ejecutado `verificar_instalacion.py`
- [ ] Jupyter Notebook funciona
- [ ] Demo notebook ejecutado exitosamente

---

## 🎉 ¡Todo Listo!

Si completaste el checklist, ¡estás listo para detectar dorsales!

```powershell
.\venv\Scripts\Activate.ps1
cd "notebooks+utils+data"
jupyter notebook
```

Abre: `05 - Bib Detection Validation & Demo.ipynb` y ¡disfruta! 🚀

---

**Optimizado para NVIDIA GeForce RTX 3050**  
**Última actualización: Octubre 2025**

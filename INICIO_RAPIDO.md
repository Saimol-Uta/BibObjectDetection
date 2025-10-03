# 🏃 Guía Rápida de Inicio - PC con NVIDIA RTX 3050

## ⚡ Instalación Rápida (15 minutos)

### Pre-requisitos
1. **Instalar Drivers NVIDIA**: https://www.nvidia.com/Download/index.aspx
   - Selecciona: RTX 3050 > Windows 10/11
   
2. **Instalar CUDA 11.8**: https://developer.nvidia.com/cuda-11-8-0-download-archive
   - Descarga e instala (3 GB, ~10 min)
   
3. **Instalar cuDNN 8.6**: https://developer.nvidia.com/cudnn
   - Requiere cuenta NVIDIA (gratis)
   - Extrae y copia archivos a carpeta CUDA
   
4. **Instalar Python 3.8-3.10**: https://www.python.org/downloads/
   - ✅ Marca "Add Python to PATH"

### Instalación Automática

Abre PowerShell en la carpeta del proyecto y ejecuta:

```powershell
# Permitir ejecución de scripts (solo primera vez)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Ejecutar instalador automático
.\instalar.ps1
```

El script automáticamente:
- ✅ Crea entorno virtual
- ✅ Instala PyTorch con CUDA
- ✅ Instala OpenCV y dependencias
- ✅ Verifica que todo funcione

---

## 🚀 Ejecución

### Opción 1: Jupyter Notebook (Recomendado)

```powershell
# Activar entorno virtual
.\venv\Scripts\Activate.ps1

# Iniciar Jupyter
cd "notebooks+utils+data"
jupyter notebook
```

Abre en tu navegador: `05 - Bib Detection Validation & Demo.ipynb`

### Opción 2: Script Python

```powershell
# Activar entorno virtual
.\venv\Scripts\Activate.ps1

# Ejecutar verificación
python verificar_instalacion.py

# Ejecutar detección personalizada (crea tu propio script)
python mi_detector.py
```

---

## 📁 Archivos Importantes

```
BibObjectDetection/
├── 📖 MANUAL_INSTALACION.md        ← Manual completo paso a paso
├── 📖 INICIO_RAPIDO.md              ← Este archivo
├── ⚙️ instalar.ps1                  ← Script de instalación automática
├── 🔍 verificar_instalacion.py     ← Verifica que todo funcione
├── 📋 requirements.txt              ← Lista de dependencias
│
├── notebooks+utils+data/
│   ├── 🎯 05 - Bib Detection Validation & Demo.ipynb  ← EMPEZAR AQUÍ
│   ├── 01 - Prepocessing & Training SVHN YOLOv4-tiny Darknet.ipynb
│   ├── 02 - Digit Detection Validation Using RBNR Data.ipynb
│   ├── 03 - Preprocessing Racing Bib Numbers (RBNR) Datasets.ipynb
│   ├── 04 - Run Yolov4 Tiny on RBNR Data.ipynb
│   ├── utils.py                     ← Funciones auxiliares
│   ├── VIDEO0433.mp4                ← Video de ejemplo
│   └── BibDetectorSample.jpeg       ← Imagen de ejemplo
│
└── weights-classes/
    ├── RBNR_custom-yolov4-tiny-detector_best.weights  ← Modelo dorsales
    ├── RBNR_custom-yolov4-tiny-detector.cfg
    ├── SVHN_custom-yolov4-tiny-detector_best.weights  ← Modelo dígitos
    └── SVHN_custom-yolov4-tiny-detector.cfg
```

---

## 🎯 ¿Qué hace este proyecto?

Detecta automáticamente **números de dorsal** en imágenes y videos de carreras usando:
- 🧠 **YOLOv4-tiny** (red neuronal rápida)
- 🎮 **NVIDIA CUDA** (aceleración GPU)
- 📷 **OpenCV** (procesamiento de video)

**Resultados**:
- ✨ 99% precisión en detección de dorsales
- ✨ 96% precisión en detección de dígitos
- ⚡ Procesamiento en tiempo real con RTX 3050

---

## 🔧 Comandos Útiles

### Verificar GPU
```powershell
nvidia-smi
```

### Verificar CUDA en Python
```powershell
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Activar entorno virtual
```powershell
.\venv\Scripts\Activate.ps1
```

### Desactivar entorno virtual
```powershell
deactivate
```

### Monitorear GPU durante ejecución
```powershell
nvidia-smi -l 1  # Actualiza cada segundo
```

---

## ❓ Solución de Problemas Rápida

### "No se puede ejecutar scripts"
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### "CUDA out of memory"
En el archivo `.cfg`, reduce el batch size:
```
batch=16  # Prueba con 16, 8 o incluso 4
```

### "nvidia-smi no encontrado"
Reinstala drivers NVIDIA desde: https://www.nvidia.com/Download/index.aspx

### "PyTorch no detecta GPU"
Reinstala PyTorch con CUDA:
```powershell
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Detección lenta
Verifica que estés usando GPU:
```python
import cv2
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)  # Usa FP16 para RTX 3050
```

---

## 📚 Más Información

- 📖 **Manual Completo**: Lee `MANUAL_INSTALACION.md` para instrucciones detalladas
- 🎥 **Presentación**: Ver `presentation/RaceBibDetection_Presentation.pdf`
- 🌐 **GitHub Original**: https://github.com/Lwhieldon/BibObjectDetection
- 📺 **Video Demo**: https://youtu.be/xfVfr0KmhYY

---

## 🎓 Datasets Utilizados

- **SVHN** (Street View House Numbers): http://ufldl.stanford.edu/housenumbers
- **RBNR** (Racing Bib Number Recognition): https://people.csail.mit.edu/talidekel/RBNR.html

---

## 📊 Especificaciones de tu Sistema

- **GPU**: NVIDIA GeForce RTX 3050
- **VRAM**: 8 GB
- **Compute Capability**: 8.6
- **CUDA Cores**: 2560
- **Tensor Cores**: 80

Tu RTX 3050 es **perfectamente capaz** de ejecutar este proyecto en tiempo real! 🚀

---

## ✅ Checklist de Instalación

- [ ] Drivers NVIDIA instalados (`nvidia-smi` funciona)
- [ ] CUDA 11.8 instalado (`nvcc --version` funciona)
- [ ] cuDNN extraído en carpeta CUDA
- [ ] Python 3.8+ instalado (`python --version`)
- [ ] Entorno virtual creado y activado
- [ ] PyTorch con CUDA instalado
- [ ] OpenCV y dependencias instaladas
- [ ] Script de verificación ejecutado exitosamente
- [ ] Jupyter Notebook funciona
- [ ] Notebook demo ejecutado correctamente

---

## 🎉 ¡Listo!

Una vez completada la instalación, ejecuta:

```powershell
.\venv\Scripts\Activate.ps1
cd "notebooks+utils+data"
jupyter notebook
```

Abre: `05 - Bib Detection Validation & Demo.ipynb` y ¡empieza a detectar dorsales!

---

**Creado para PC con NVIDIA RTX 3050**
**Octubre 2025**

# 🎯 Guía de Uso del Detector Propio

## 📋 Archivo Creado

He creado `mi_detector.py` - un script completo y profesional para detectar dorsales de manera fácil.

---

## 🚀 Instalación Rápida

### 1. Asegúrate de tener Python instalado

```powershell
python --version
```

### 2. Crea y activa el entorno virtual (si no lo has hecho)

```powershell
# Crear entorno
python -m venv venv

# Activar
.\venv\Scripts\Activate.ps1
```

### 3. Instala las dependencias

```powershell
# PyTorch con CUDA 12.1 (compatible con CUDA 13.0)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# OpenCV y otras librerías
pip install opencv-python numpy
```

---

## 🎮 Modos de Uso

### 📹 Modo 1: Detección con Cámara en Tiempo Real

```powershell
python mi_detector.py --modo camara
```

**Características:**
- ✅ Detección en tiempo real
- ✅ Muestra FPS y número de detecciones
- ✅ Capturar frames con tecla 'C'
- ✅ Pausar/reanudar con 'ESPACIO'
- ✅ Salir con 'Q' o 'ESC'

**Controles durante ejecución:**
- `Q` o `ESC` - Salir
- `C` - Capturar frame actual
- `ESPACIO` - Pausar/Reanudar

---

### 🖼️ Modo 2: Detección en Imagen

```powershell
python mi_detector.py --modo imagen --archivo ruta/a/tu/imagen.jpg
```

**Ejemplos:**

```powershell
# Detectar en la imagen de ejemplo incluida
python mi_detector.py --modo imagen --archivo "notebooks+utils+data/BibDetectorSample.jpeg"

# Detectar en tu propia imagen
python mi_detector.py --modo imagen --archivo "C:\Users\tuusuario\fotos\maraton.jpg"

# Usar modelo de dígitos en lugar de dorsales
python mi_detector.py --modo imagen --archivo foto.jpg --modelo SVHN
```

**Salida:**
- Muestra la imagen con detecciones
- Guarda resultado en `output/images/`
- Lista todas las detecciones en consola

---

### 🎥 Modo 3: Detección en Video

```powershell
python mi_detector.py --modo video --archivo ruta/a/tu/video.mp4
```

**Ejemplos:**

```powershell
# Detectar en el video de ejemplo incluido
python mi_detector.py --modo video --archivo "notebooks+utils+data/VIDEO0433.mp4"

# Detectar en tu propio video
python mi_detector.py --modo video --archivo "C:\Users\tuusuario\videos\carrera.mp4"

# Procesar sin guardar el resultado
python mi_detector.py --modo video --archivo video.mp4 --no-guardar
```

**Características:**
- ✅ Procesa todo el video automáticamente
- ✅ Muestra barra de progreso
- ✅ Guarda video procesado en `output/videos/`
- ✅ Muestra estadísticas al finalizar

---

## ⚙️ Opciones Avanzadas

### Cambiar Modelo

```powershell
# Usar modelo para DORSALES (por defecto)
python mi_detector.py --modo camara --modelo RBNR

# Usar modelo para DÍGITOS
python mi_detector.py --modo camara --modelo SVHN
```

### Forzar Uso de CPU

```powershell
# Si tienes problemas con la GPU, usa CPU
python mi_detector.py --modo imagen --archivo foto.jpg --cpu
```

### Ajustar Umbral de Confianza

```powershell
# Más estricto (menos detecciones, más precisas)
python mi_detector.py --modo camara --confianza 0.7

# Más permisivo (más detecciones, algunas falsas)
python mi_detector.py --modo camara --confianza 0.3

# Default: 0.5 (equilibrado)
```

### No Guardar Resultados

```powershell
# Solo visualizar, no guardar
python mi_detector.py --modo video --archivo video.mp4 --no-guardar
```

---

## 📂 Estructura de Salida

El script crea automáticamente estas carpetas:

```
BibObjectDetection/
└── output/
    ├── images/          ← Imágenes procesadas y capturas
    │   ├── deteccion_20251003_120530.jpg
    │   ├── captura_20251003_120615.jpg
    │   └── ...
    └── videos/          ← Videos procesados
        ├── deteccion_20251003_121045.mp4
        └── ...
```

**Nombres de archivos:**
- `deteccion_YYYYMMDD_HHMMSS.jpg/mp4` - Resultados automáticos
- `captura_YYYYMMDD_HHMMSS.jpg` - Capturas manuales de cámara

---

## 🎨 Ejemplos Completos

### Ejemplo 1: Test Rápido con Imagen de Ejemplo

```powershell
# Activar entorno
.\venv\Scripts\Activate.ps1

# Detectar en imagen de ejemplo
python mi_detector.py --modo imagen --archivo "notebooks+utils+data/BibDetectorSample.jpeg"
```

### Ejemplo 2: Video en Tiempo Real con Cámara

```powershell
# Activar entorno
.\venv\Scripts\Activate.ps1

# Iniciar cámara con umbral más estricto
python mi_detector.py --modo camara --confianza 0.6
```

### Ejemplo 3: Procesar Video Completo

```powershell
# Activar entorno
.\venv\Scripts\Activate.ps1

# Procesar video de ejemplo
python mi_detector.py --modo video --archivo "notebooks+utils+data/VIDEO0433.mp4"

# El resultado se guardará en: output/videos/deteccion_[timestamp].mp4
```

### Ejemplo 4: Detectar Dígitos en Lugar de Dorsales

```powershell
# Usar modelo SVHN para detectar dígitos individuales
python mi_detector.py --modo imagen --archivo foto.jpg --modelo SVHN
```

### Ejemplo 5: Uso con CPU (sin GPU)

```powershell
# Si tienes problemas con CUDA
python mi_detector.py --modo camara --cpu
```

---

## 🔍 Información Mostrada en Pantalla

Durante la ejecución, verás:

### En Modo Cámara/Video:
- **Frame actual**: Número de frame procesado
- **Detecciones**: Cantidad de dorsales detectados en el frame actual
- **Barra de progreso**: Solo en video, muestra el porcentaje completado
- **Controles**: Recordatorio de teclas disponibles

### En Consola:
```
======================================================================
  DETECTOR DE DORSALES
======================================================================

[1/3] Cargando modelo...
   ✓ Modelo cargado (88.7 MB)

[2/3] Cargando clases...
   ✓ 1 clases cargadas: ['bib']

[3/3] Configurando backend...
   ✓ Backend configurado: GPU (CUDA FP16)

✅ Detector inicializado correctamente
   Modelo: RBNR
   Backend: GPU (CUDA)
   Clases: 1
======================================================================
```

---

## ❓ Solución de Problemas

### ❌ Error: "No se encontró ninguna cámara"

**Solución:**
1. Cierra otras aplicaciones que usen la cámara (Zoom, Teams, etc.)
2. Verifica permisos de cámara en Windows
3. Usa el modo video con un archivo en su lugar

### ❌ Error: "No se encuentra el archivo .weights"

**Solución:**
Verifica que la carpeta `weights-classes/` contenga los archivos:
```
weights-classes/
├── RBNR_custom-yolov4-tiny-detector_best.weights
├── RBNR_custom-yolov4-tiny-detector.cfg
└── RBRN_obj.names
```

### ❌ Error: "DLL load failed" o problemas con CUDA

**Solución:**
```powershell
# Usar CPU en lugar de GPU
python mi_detector.py --modo camara --cpu
```

### ⚠️ Advertencia: "No se pudo usar GPU, usando CPU"

Esto es normal si:
- No tienes PyTorch con CUDA instalado
- Los drivers NVIDIA no están actualizados
- OpenCV no tiene soporte CUDA compilado

**El detector funcionará igual, solo más lento.**

---

## 📊 Rendimiento Esperado en RTX 3050

| Modo | FPS (aprox) | Resolución | Uso VRAM |
|------|-------------|------------|----------|
| Cámara | 30-40 FPS | 1280x720 | ~1 GB |
| Video | 30-40 FPS | Variable | ~1 GB |
| Imagen | Inmediato | Variable | ~1 GB |

---

## 🎯 Características del Script

### ✅ Ventajas
- **Fácil de usar**: Solo 1 comando para cualquier tarea
- **Completo**: Maneja cámara, imágenes y videos
- **Robusto**: Manejo de errores y validaciones
- **Visual**: Información clara en pantalla
- **Flexible**: Muchas opciones configurables
- **Profesional**: Código limpio y bien documentado

### 🎨 Personalización
Puedes editar el archivo `mi_detector.py` y cambiar:
- `Config.CONFIANZA_MIN`: Umbral de confianza por defecto
- `Config.COLOR_DETECCION`: Color de los rectángulos
- `Config.INPUT_SIZE`: Tamaño de entrada para YOLO
- `Config.CAMERA_WIDTH/HEIGHT`: Resolución de cámara

---

## 📝 Comandos de Referencia Rápida

```powershell
# Ver ayuda completa
python mi_detector.py --help

# Modo cámara
python mi_detector.py --modo camara

# Modo imagen
python mi_detector.py --modo imagen --archivo foto.jpg

# Modo video
python mi_detector.py --modo video --archivo video.mp4

# Cambiar modelo a dígitos
python mi_detector.py --modo camara --modelo SVHN

# Usar CPU
python mi_detector.py --modo camara --cpu

# Umbral personalizado
python mi_detector.py --modo camara --confianza 0.7

# No guardar resultado
python mi_detector.py --modo video --archivo video.mp4 --no-guardar

# Combinación de opciones
python mi_detector.py --modo imagen --archivo foto.jpg --modelo SVHN --confianza 0.6 --cpu
```

---

## 🚀 ¡Empezar Ahora!

### Paso 1: Activa el entorno
```powershell
.\venv\Scripts\Activate.ps1
```

### Paso 2: Prueba con la imagen de ejemplo
```powershell
python mi_detector.py --modo imagen --archivo "notebooks+utils+data/BibDetectorSample.jpeg"
```

### Paso 3: ¡Disfruta detectando dorsales!

---

## 💡 Consejos

1. **Para mejores resultados**: Usa `--confianza 0.6` o superior
2. **Para detectar más objetos**: Usa `--confianza 0.3` o inferior
3. **Si es lento**: Usa `--cpu` o reduce la resolución de entrada
4. **Para debugging**: El script imprime información detallada en consola
5. **Capturas de cámara**: Se guardan automáticamente con timestamp

---

## 📞 Soporte

Si tienes problemas:
1. Verifica que el entorno virtual esté activado
2. Confirma que los archivos de pesos existen
3. Prueba con `--cpu` si hay errores de GPU
4. Revisa los mensajes de error en consola

---

**¡Listo para detectar dorsales como un profesional! 🏃‍♂️🏃‍♀️**

*Creado: Octubre 2025 | Compatible con: Windows 10/11 + Python 3.8-3.10*

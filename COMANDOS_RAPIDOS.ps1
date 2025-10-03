# ═══════════════════════════════════════════════════════════════
# COMANDOS RÁPIDOS - Copia y Pega
# ═══════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────
# PASO 1: ACTIVAR ENTORNO VIRTUAL
# ───────────────────────────────────────────────────────────────

.\venv\Scripts\Activate.ps1


# ───────────────────────────────────────────────────────────────
# OPCIÓN FÁCIL: USAR MENÚ INTERACTIVO
# ───────────────────────────────────────────────────────────────

.\iniciar_detector.ps1


# ───────────────────────────────────────────────────────────────
# MODO CÁMARA - DETECCIÓN EN TIEMPO REAL
# ───────────────────────────────────────────────────────────────

# Básico
python mi_detector.py --modo camara

# Con modelo de dígitos
python mi_detector.py --modo camara --modelo SVHN

# Con mayor precisión (menos detecciones, más confiables)
python mi_detector.py --modo camara --confianza 0.7

# Forzar CPU si hay problemas con GPU
python mi_detector.py --modo camara --cpu


# ───────────────────────────────────────────────────────────────
# MODO IMAGEN - PROCESAR FOTOS
# ───────────────────────────────────────────────────────────────

# Imagen de ejemplo incluida
python mi_detector.py --modo imagen --archivo "notebooks+utils+data\BibDetectorSample.jpeg"

# Tu propia imagen (cambia la ruta)
python mi_detector.py --modo imagen --archivo "C:\ruta\a\tu\imagen.jpg"

# Con modelo de dígitos
python mi_detector.py --modo imagen --archivo imagen.jpg --modelo SVHN

# Con umbral personalizado
python mi_detector.py --modo imagen --archivo imagen.jpg --confianza 0.6

# Sin guardar resultado (solo visualizar)
python mi_detector.py --modo imagen --archivo imagen.jpg --no-guardar


# ───────────────────────────────────────────────────────────────
# MODO VIDEO - PROCESAR VIDEOS
# ───────────────────────────────────────────────────────────────

# Video de ejemplo incluido
python mi_detector.py --modo video --archivo "notebooks+utils+data\VIDEO0433.mp4"

# Tu propio video (cambia la ruta)
python mi_detector.py --modo video --archivo "C:\ruta\a\tu\video.mp4"

# Con mayor precisión
python mi_detector.py --modo video --archivo video.mp4 --confianza 0.7

# Sin guardar resultado
python mi_detector.py --modo video --archivo video.mp4 --no-guardar

# Usando CPU
python mi_detector.py --modo video --archivo video.mp4 --cpu


# ───────────────────────────────────────────────────────────────
# COMBINACIONES ÚTILES
# ───────────────────────────────────────────────────────────────

# Cámara con dígitos y alta precisión
python mi_detector.py --modo camara --modelo SVHN --confianza 0.7

# Imagen con CPU y umbral bajo
python mi_detector.py --modo imagen --archivo foto.jpg --cpu --confianza 0.3

# Video sin guardar, con dígitos
python mi_detector.py --modo video --archivo video.mp4 --modelo SVHN --no-guardar


# ───────────────────────────────────────────────────────────────
# AYUDA Y VERIFICACIÓN
# ───────────────────────────────────────────────────────────────

# Ver todas las opciones disponibles
python mi_detector.py --help

# Verificar instalación
python verificar_instalacion.py

# Verificar GPU
nvidia-smi

# Verificar PyTorch con CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"


# ───────────────────────────────────────────────────────────────
# EJEMPLOS POR CASO DE USO
# ───────────────────────────────────────────────────────────────

# CASO 1: Probar rápidamente el detector
python mi_detector.py --modo imagen --archivo "notebooks+utils+data\BibDetectorSample.jpeg"

# CASO 2: Detectar en fotos de una carrera
python mi_detector.py --modo imagen --archivo "C:\fotos\maraton\foto1.jpg"

# CASO 3: Procesar video de evento deportivo
python mi_detector.py --modo video --archivo "C:\videos\carrera.mp4"

# CASO 4: Detección en vivo durante un evento
python mi_detector.py --modo camara --confianza 0.6

# CASO 5: Análisis de dígitos individuales
python mi_detector.py --modo imagen --archivo foto.jpg --modelo SVHN

# CASO 6: Procesamiento rápido sin guardar
python mi_detector.py --modo video --archivo video.mp4 --no-guardar

# CASO 7: Máxima precisión (solo dorsales muy claros)
python mi_detector.py --modo camara --confianza 0.8

# CASO 8: Detectar todo lo posible (más detecciones, algunas falsas)
python mi_detector.py --modo camara --confianza 0.3


# ═══════════════════════════════════════════════════════════════
# NOTA: Los resultados se guardan automáticamente en:
#   - Imágenes: output\images\
#   - Videos:   output\videos\
# ═══════════════════════════════════════════════════════════════

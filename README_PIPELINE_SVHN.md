# Pipeline de Detección de Dorsales con SVHN

Sistema de detección y registro automático de números de dorsal en tiempo real usando YOLOv4-tiny (RBNR + SVHN).

## 🎯 Características

- **Detección de dorsales (bib)**: Usa modelo RBNR para localizar la región del dorsal (caja verde).
- **Reconocimiento de dígitos**: Usa modelo SVHN para detectar dígitos dentro del dorsal (cajas naranjas).
- **Filtrado inteligente**: 
  - Solo registra clusters de dígitos con alta confianza.
  - Valida solapamiento vertical y cobertura del dorsal.
  - Rechaza dígitos sueltos o parciales.
  - Evita duplicados con debounce temporal.
- **Registro automático en Excel**: Guarda posición, dorsal y hora en `registros_dorsales.xlsx`.

## 📋 Requisitos

```bash
pip install -r requirements.txt
```

Dependencias principales:
- opencv-python >= 4.6.0
- numpy >= 1.21.0
- pandas >= 1.3.0
- openpyxl >= 3.0.0

## 🚀 Uso Rápido

### Modo Cámara (Tiempo Real)
```powershell
py -3 .\pipeline_bib_svhn.py --modo camara
```

**Controles:**
- `q` o `ESC` - Salir
- `c` - Capturar frame actual
- `ESPACIO` - Pausar/Reanudar

### Modo Imagen
```powershell
py -3 .\pipeline_bib_svhn.py --modo imagen --archivo path\to\imagen.jpg
```

### Ajustar Umbrales
```powershell
py -3 .\pipeline_bib_svhn.py --modo camara --conf 0.35 --conf_svhn 0.2
```

## ⚙️ Parámetros de Configuración

Los siguientes parámetros se pueden ajustar en la clase `Config` dentro de `pipeline_bib_svhn.py`:

### Umbrales de Detección
- `CONF_RBNR = 0.3` - Confianza mínima para detección de bib
- `CONF_SVHN = 0.25` - Confianza mínima para detección inicial de dígitos

### Filtrado de Dígitos (Crítico para Precisión)
- `CONF_SVHN_MIN_DIGIT = 0.75` ⭐
  - Confianza mínima por dígito individual
  - **Aumentar** (0.8-0.85) para mayor precisión
  - **Disminuir** (0.65-0.7) si no detecta números válidos

- `CONF_SVHN_AVG_MIN = 0.85` ⭐
  - Confianza promedio del cluster completo
  - **Aumentar** (0.9) para ser más estricto
  - **Disminuir** (0.75-0.8) si rechaza dorsales válidos

### Validación Espacial
- `MIN_DIGITS_WIDTH_RATIO = 0.25` ⭐
  - Proporción mínima del ancho del bib que debe ocupar el número
  - Evita dígitos sueltos o parciales
  - **Aumentar** (0.3-0.4) para números más grandes
  - **Disminuir** (0.15-0.2) para dorsales pequeños

- `MIN_VERTICAL_OVERLAP_RATIO = 0.6`
  - Solapamiento vertical entre dígitos y bib
  - Asegura que los dígitos estén dentro del dorsal

### Validación de Longitud
- `MIN_DIGITS_COUNT = 2` ⭐
  - Número mínimo de dígitos (evita "0", "5" sueltos)
  
- `MAX_DIGITS_COUNT = 4`
  - Número máximo de dígitos (evita ruido)

### Debounce
- `DEBOUNCE_SECONDS = 15` ⭐
  - Tiempo mínimo entre registros del mismo dorsal
  - **Aumentar** (20-30) en carreras lentas
  - **Disminuir** (8-10) en sprints o alta velocidad

## 🎨 Visualización

- **Caja verde** 🟢: Región del dorsal detectada (modelo RBNR)
- **Texto verde grande**: Número de dorsal reconocido y aceptado
- **Cajas naranjas** 🟠: Dígitos individuales detectados (modelo SVHN)
- **Texto naranja**: Dígito y confianza de cada detección

## 📊 Archivo de Salida

`registros_dorsales.xlsx` contiene:
- **Posición**: Orden de llegada (auto-incremental)
- **Dorsal**: Número del corredor
- **HoraLlegada**: Timestamp en formato `YYYY-MM-DD HH:MM:SS`

### Reglas del Registro
✅ **Se registra si:**
- El cluster tiene ≥2 dígitos y ≤4 dígitos
- Confianza promedio ≥ 0.85
- Cada dígito tiene confianza ≥ 0.75
- Ocupa ≥25% del ancho del bib
- Solapamiento vertical ≥60%
- No fue registrado en los últimos 15 segundos

❌ **No se registra si:**
- Dígitos sueltos (ej. "0", "5")
- Sub-cadenas parciales (ej. "10" del "210")
- Baja confianza o ruido
- Dorsal ya registrado recientemente

## 🔧 Solución de Problemas

### Problema: No detecta dorsales válidos
**Solución:**
- Bajar `CONF_SVHN_MIN_DIGIT` a 0.65-0.70
- Bajar `CONF_SVHN_AVG_MIN` a 0.75-0.80
- Bajar `MIN_DIGITS_WIDTH_RATIO` a 0.15-0.20
- Verificar que los archivos `.weights` estén completos

### Problema: Registra dígitos sueltos (ej. "0", "5", "10")
**Solución:**
- ✅ Ya ajustado: `CONF_SVHN_MIN_DIGIT = 0.75`
- ✅ Ya ajustado: `CONF_SVHN_AVG_MIN = 0.85`
- ✅ Ya ajustado: `MIN_DIGITS_WIDTH_RATIO = 0.25`
- Si persiste, aumentar `MIN_DIGITS_WIDTH_RATIO` a 0.3

### Problema: Registra el mismo dorsal múltiples veces
**Solución:**
- Aumentar `DEBOUNCE_SECONDS` a 20-30
- Verificar que el dorsal no cambie ligeramente (ej. "100" vs "10")

### Problema: Detecta pero no registra nada
**Solución:**
- Revisar consola: debe mostrar `[REGISTRO] Añadida fila: ...`
- Si no aparece mensaje, los clusters no pasan los filtros
- Revisar si `should_register()` está bloqueando por debounce
- Eliminar `registros_dorsales.xlsx` para limpiar cache

## 📁 Estructura de Archivos

```
BibObjectDetection/
├── pipeline_bib_svhn.py          # Script principal
├── requirements.txt              # Dependencias
├── registros_dorsales.xlsx       # Salida (generado automáticamente)
├── output/                       # Capturas y resultados
│   ├── captura_*.jpg
│   └── pipeline_result_*.jpg
└── weights-classes/              # Modelos entrenados
    ├── RBNR_custom-yolov4-tiny-detector_best.weights
    ├── RBNR_custom-yolov4-tiny-detector.cfg
    ├── RBRN_obj.names
    ├── SVHN_custom-yolov4-tiny-detector_best.weights
    ├── SVHN_custom-yolov4-tiny-detector.cfg
    └── SVHN_obj.names
```

## 🎯 Calibración Recomendada

Para tu caso (dorsales 100, 154, 210, etc.):

**Configuración Estricta (Actual - Recomendada):**
```python
CONF_SVHN_MIN_DIGIT = 0.75
CONF_SVHN_AVG_MIN = 0.85
MIN_DIGITS_WIDTH_RATIO = 0.25
MIN_DIGITS_COUNT = 2
MAX_DIGITS_COUNT = 4
DEBOUNCE_SECONDS = 15
```

**Si no detecta suficientes dorsales válidos:**
```python
CONF_SVHN_MIN_DIGIT = 0.70
CONF_SVHN_AVG_MIN = 0.80
MIN_DIGITS_WIDTH_RATIO = 0.20
```

**Si sigue registrando parciales (ej. "10" del "210"):**
```python
CONF_SVHN_MIN_DIGIT = 0.80
CONF_SVHN_AVG_MIN = 0.90
MIN_DIGITS_WIDTH_RATIO = 0.30
DEBOUNCE_SECONDS = 20
```

## 📝 Notas Importantes

1. **El sistema solo registra el número final** (texto verde grande), no los dígitos individuales (naranjas).
2. **Los clusters se filtran por:**
   - Confianza individual y promedio
   - Cantidad de dígitos (2-4)
   - Cobertura espacial del bib
   - Solapamiento vertical
3. **El debounce previene registros repetidos** del mismo dorsal en frames consecutivos.
4. **Los dígitos sueltos se rechazan** porque no cumplen `MIN_DIGITS_COUNT = 2`.

## 🆘 Soporte

Si tienes problemas:
1. Verifica que los modelos `.weights` estén en `weights-classes/`
2. Revisa la consola para mensajes de error
3. Ajusta los umbrales según las recomendaciones arriba
4. Prueba con `--conf_svhn 0.2` para ver más detecciones y luego ajusta hacia arriba

---

**Autor**: Sistema de Detección de Dorsales  
**Fecha**: Octubre 2025  
**Versión**: 2.0 (con filtrado avanzado y validación de clusters)

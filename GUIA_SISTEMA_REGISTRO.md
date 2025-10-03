# 📊 GUÍA: SISTEMA DE REGISTRO DE LLEGADAS

## 🎯 DESCRIPCIÓN

Sistema automático para registrar dorsales detectados en un archivo Excel con:
- ✅ **Posición auto-incremental** (1, 2, 3, ...)
- ✅ **Número de dorsal** detectado
- ✅ **Hora de llegada** formato `YYYY-MM-DD HH:MM:SS`
- ✅ **Observaciones** (campo editable)
- ✅ **Prevención de duplicados**

---

## 📦 INSTALACIÓN

### 1️⃣ Instalar dependencias de Excel

```powershell
# Opción A: Script automático (RECOMENDADO)
.\instalar_registro.ps1

# Opción B: Manual
.\venv\Scripts\Activate.ps1
pip install pandas openpyxl
```

### 2️⃣ Verificar instalación

```powershell
python registro_llegadas.py
```

**Resultado esperado:**
- Crea archivo `test_registro_llegadas.xlsx`
- Muestra 3 registros de prueba
- Estadísticas del sistema

---

## 🚀 USO BÁSICO

### **Modo 1: Detector con Registro Automático (Cámara)**

```powershell
python mi_detector_registro.py --modo camara
```

**Funcionamiento:**
1. Abre la cámara en tiempo real
2. Detecta dorsales automáticamente
3. Registra cada dorsal nuevo en `registro_llegadas.xlsx`
4. Muestra dorsales registrados en **naranja**
5. Muestra dorsales nuevos en **verde**
6. Muestra posición en la etiqueta: `123: 0.95 [POS: 1]`

**Controles:**
- **`s`** - Mostrar estadísticas de la carrera
- **`r`** - Registrar dorsal manualmente (pausar y seleccionar)
- **`c`** - Capturar imagen
- **`ESPACIO`** - Pausar/Reanudar
- **`ESC` / `q`** - Salir

---

### **Modo 2: Detector con Registro Automático (Imagen)**

```powershell
python mi_detector_registro.py --modo imagen --archivo ruta\imagen.jpg
```

**Funcionamiento:**
- Procesa una imagen estática
- Detecta todos los dorsales
- Registra cada dorsal en Excel
- Muestra imagen con detecciones

---

### **Modo 3: Detector SIN Registro (Solo Detección)**

```powershell
python mi_detector_registro.py --modo camara --sin-registro
```

**Uso:** Solo quieres ver detecciones sin guardar nada.

---

## ⚙️ CONFIGURACIONES AVANZADAS

### **Archivo Excel Personalizado**

```powershell
# Cambiar nombre del archivo Excel
python mi_detector_registro.py --modo camara --excel maraton_2025.xlsx
```

### **Permitir Dorsales Duplicados**

Edita `mi_detector_registro.py` línea 28:

```python
# Cambiar de:
self.registro = RegistroLlegadas(archivo_excel, permitir_duplicados=False)

# A:
self.registro = RegistroLlegadas(archivo_excel, permitir_duplicados=True)
```

### **Ajustar Tiempo Entre Registros**

Evita registrar el mismo dorsal múltiples veces. Por defecto: **2 segundos**.

Edita línea 19:

```python
INTERVALO_REGISTRO = 5.0  # Cambiar a 5 segundos
```

---

## 📋 FORMATO DEL ARCHIVO EXCEL

El archivo `registro_llegadas.xlsx` contiene:

| Posicion | Dorsal | HoraLlegada          | Observaciones |
|----------|--------|----------------------|---------------|
| 1        | 123    | 2025-06-15 08:30:15 |               |
| 2        | 456    | 2025-06-15 08:30:47 |               |
| 3        | 789    | 2025-06-15 08:31:12 | Lesionado     |

### **Columnas:**
1. **Posicion** - Auto-incrementa (1, 2, 3, ...)
2. **Dorsal** - Número detectado
3. **HoraLlegada** - Formato `YYYY-MM-DD HH:MM:SS`
4. **Observaciones** - Campo editable (puedes agregar notas manualmente en Excel)

---

## 🔍 PREVENCIÓN DE DUPLICADOS

### **Sistema de Cooldown (2 segundos)**

Cuando detecta un dorsal:
1. **Primera detección** → Registra en Excel ✅
2. **Siguiente 2 segundos** → Ignora ese dorsal ⏱️
3. **Después de 2 segundos** → Si detecta de nuevo, NO registra (ya está en Excel) ❌

### **Visual:**
- **Verde** 🟢 = Dorsal nuevo, se registró
- **Naranja** 🟠 = Dorsal ya registrado anteriormente

---

## 📊 ESTADÍSTICAS EN TIEMPO REAL

Presiona **`s`** durante detección para ver:

```
==============================
 ESTADÍSTICAS DE LA CARRERA
==============================
Total llegadas: 15
Primer llegada: 2025-06-15 08:30:15
Última llegada: 2025-06-15 08:45:32
Tiempo transcurrido: 15.28 minutos

ÚLTIMAS 5 LLEGADAS:
  Pos 15: Dorsal 823 - 08:45:32
  Pos 14: Dorsal 456 - 08:44:18
  ...
==============================
```

---

## 🛠️ MÓDULO INDEPENDIENTE: `registro_llegadas.py`

Si quieres usar el sistema de registro en TU PROPIO script:

```python
from registro_llegadas import RegistroLlegadas

# Crear sistema de registro
registro = RegistroLlegadas('mi_carrera.xlsx', permitir_duplicados=False)

# Registrar llegada
resultado = registro.registrar_llegada(dorsal=123)
print(f"Posición: {resultado['posicion']}")
print(f"Dorsal: {resultado['dorsal']}")
print(f"Hora: {resultado['hora']}")
print(f"¿Duplicado? {resultado['duplicado']}")

# Actualizar observaciones
registro.actualizar_observaciones(dorsal=123, observaciones="Lesionado")

# Obtener estadísticas
stats = registro.obtener_estadisticas()
print(f"Total llegadas: {stats['total_llegadas']}")

# Listar llegadas
llegadas = registro.listar_llegadas()
for llegada in llegadas:
    print(f"Pos {llegada['Posicion']}: Dorsal {llegada['Dorsal']}")
```

---

## ⚡ CASOS DE USO

### **🏃 Carrera Pedestre (5-15 km/h)**

```powershell
python mi_detector_registro.py --modo camara --excel maraton_2025.xlsx
```

✅ 30 FPS suficiente  
✅ Registro automático  
✅ Cámara fija en meta  

---

### **🚴 Carrera Ciclista (20-40 km/h)**

```powershell
python mi_detector_registro.py --modo camara --excel ciclismo_2025.xlsx
```

⚠️ Ajustar cámara:
- Distancia: 2-3 metros
- Ángulo perpendicular
- Buena iluminación

---

### **📸 Procesar Fotos de Llegada**

```powershell
# Procesar una imagen
python mi_detector_registro.py --modo imagen --archivo meta_001.jpg

# Procesar múltiples imágenes (loop)
for %%f in (*.jpg) do python mi_detector_registro.py --modo imagen --archivo %%f
```

---

## 🐛 SOLUCIÓN DE PROBLEMAS

### ❌ **PROBLEMA: Se guarda "bib" en lugar del número del dorsal**

**Causa:** El modelo RBNR detecta la región del dorsal pero NO lee el número específico.

**Solución:** Usar el detector con OCR (lectura de caracteres):

```powershell
# 1. Instalar OCR
.\instalar_ocr.ps1

# 2. Usar el detector con OCR
python mi_detector_ocr.py --modo camara
```

**¿Qué es OCR?**
- Optical Character Recognition (Reconocimiento Óptico de Caracteres)
- Lee los números dentro del dorsal detectado
- Dos opciones:
  - **EasyOCR** (recomendado): Mayor precisión, descarga ~500MB
  - **Tesseract**: Más ligero, requiere instalación adicional

**Archivos:**
- `mi_detector_registro.py` - Detector SIN OCR (guarda "bib")
- `mi_detector_ocr.py` - Detector CON OCR (lee el número real) ✅

---

### ❌ Error: `ModuleNotFoundError: No module named 'pandas'`

**Solución:**
```powershell
.\instalar_registro.ps1
```

---

### ❌ No detecta ningún dorsal

**Causas:**
1. Modelo no cargado correctamente
2. Cámara sin permisos
3. Dorsales muy pequeños/lejanos

**Solución:**
```powershell
# Verificar modelo
dir weights-classes\

# Debe existir:
# RBNR_custom-yolov4-tiny-detector_best.weights
# RBNR_custom-yolov4-tiny-detector.cfg
# RBRN_obj.names
```

---

### ❌ Registra múltiples veces el mismo dorsal

**Causa:** Cooldown muy corto

**Solución:** Aumentar `INTERVALO_REGISTRO` en `mi_detector_registro.py`:

```python
INTERVALO_REGISTRO = 5.0  # De 2 a 5 segundos
```

---

### ❌ Excel no se actualiza

**Causa:** Archivo abierto en Excel

**Solución:** Cerrar Excel, el script actualiza automáticamente.

---

## 📈 RENDIMIENTO

| Escenario              | FPS  | Registros/seg | Adecuado para          |
|------------------------|------|---------------|------------------------|
| Cámara CPU (actual)    | 30   | 0.5           | ✅ Carreras pedestres  |
| Cámara GPU (compilado) | 60+  | 1.0+          | ✅ Ciclismo avanzado   |
| Imagen estática        | N/A  | Instantáneo   | ✅ Post-procesamiento  |

---

## 📝 TIPS Y RECOMENDACIONES

### ✅ **Mejores Prácticas**

1. **Cámara fija** en línea de meta
2. **Distancia óptima:** 2-5 metros
3. **Ángulo:** Perpendicular (90°) a los corredores
4. **Iluminación:** Natural o LED constante
5. **Fondo:** Contraste alto con dorsales

### ⚠️ **Evitar**

1. Cámara en movimiento
2. Contraluz directo
3. Dorsales mojados/arrugados
4. Múltiples corredores superpuestos
5. Distancia >10 metros

---

## 🎓 EJEMPLOS COMPLETOS

### **Ejemplo 1: Maratón Completo**

```powershell
# 1. Crear archivo específico
python mi_detector_registro.py --modo camara --excel maraton_lima_2025.xlsx

# 2. Durante la carrera:
#    - Presiona 's' cada 10 minutos para ver estadísticas
#    - Sistema registra automáticamente cada llegada
#    - Dorsales registrados se ven en naranja

# 3. Al finalizar:
#    - Abre maraton_lima_2025.xlsx
#    - Verifica registros
#    - Agrega observaciones manualmente si necesitas
```

---

### **Ejemplo 2: Procesamiento Post-Carrera**

Si tienes fotos de la meta:

```powershell
# Procesar todas las fotos de la carpeta
python mi_detector_registro.py --modo imagen --archivo foto_meta_01.jpg
python mi_detector_registro.py --modo imagen --archivo foto_meta_02.jpg
python mi_detector_registro.py --modo imagen --archivo foto_meta_03.jpg

# El sistema acumula todos los dorsales en el Excel
```

---

### **Ejemplo 3: Verificación Rápida**

```powershell
# 1. Probar sistema de registro
python registro_llegadas.py
# Crea test_registro_llegadas.xlsx con datos de prueba

# 2. Verificar detector sin guardar
python mi_detector_registro.py --modo camara --sin-registro
# Solo muestra detecciones, no registra nada
```

---

## 🆘 SOPORTE

### **Archivos Importantes:**

- `mi_detector_registro.py` - Detector con registro
- `registro_llegadas.py` - Módulo de Excel
- `mi_detector_rapido.py` - Detector original (sin registro)
- `instalar_registro.ps1` - Instalador de dependencias

### **Comandos Útiles:**

```powershell
# Verificar instalación
python -c "import pandas, openpyxl; print('OK')"

# Ver versiones
python -c "import pandas, openpyxl; print(pandas.__version__, openpyxl.__version__)"

# Listar archivos Excel
dir *.xlsx
```

---

## 🎯 RESUMEN RÁPIDO

```powershell
# INSTALAR
.\instalar_registro.ps1

# USAR (MODO MÁS COMÚN)
python mi_detector_registro.py --modo camara

# CONTROLES
# s = estadísticas
# r = registro manual
# c = capturar
# ESPACIO = pausar
# ESC = salir

# RESULTADO
# Archivo: registro_llegadas.xlsx
# Formato: Posicion | Dorsal | HoraLlegada | Observaciones
```

---

## ✅ CHECKLIST PRE-CARRERA

- [ ] Dependencias instaladas (`.\instalar_registro.ps1`)
- [ ] Modelo cargado (`weights-classes/` existe)
- [ ] Cámara funciona (`python mi_detector_registro.py --modo camara`)
- [ ] Excel se crea correctamente (ejecuta y verifica archivo)
- [ ] Configurar nombre archivo: `--excel nombre_carrera.xlsx`
- [ ] Ajustar cooldown si necesario (`INTERVALO_REGISTRO`)
- [ ] Probar detección con dorsal de prueba

---

**🏁 ¡Listo para registrar tu carrera!**

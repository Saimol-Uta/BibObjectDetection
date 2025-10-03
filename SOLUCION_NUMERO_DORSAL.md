# 🔧 SOLUCIÓN: Guardar NÚMERO del dorsal (no "bib")

## ❌ **PROBLEMA**
El sistema guarda esto en Excel:
```
Posicion | Dorsal | HoraLlegada
1        | bib    | 2025-10-03 16:42:20
```

**Debería guardar:**
```
Posicion | Dorsal | HoraLlegada
1        | 123    | 2025-10-03 16:42:20
```

---

## 🎯 **CAUSA**

El modelo YOLOv4 detecta la **región del dorsal** pero NO lee el **número específico**.

- `mi_detector_registro.py` → Solo detecta "bib" (la clase)
- `mi_detector_ocr.py` → Detecta + Lee el número con OCR ✅

---

## ✅ **SOLUCIÓN (2 PASOS)**

### **1️⃣ Instalar OCR**

```powershell
.\instalar_ocr.ps1
```

**Opciones:**
- **A) EasyOCR** (RECOMENDADO)
  - ✅ Mayor precisión con dorsales
  - ⚠️ Descarga ~500MB (solo primera vez)
  - ⚠️ Primera ejecución más lenta

- **B) Tesseract**
  - ✅ Más rápido de instalar
  - ⚠️ Menor precisión
  - ⚠️ Requiere instalación extra en Windows

**Recomendación:** Elige **A) EasyOCR**

---

### **2️⃣ Usar Detector con OCR**

```powershell
python mi_detector_ocr.py --modo camara
```

---

## 🎨 **COLORES (con OCR)**

- 🟢 **Verde** = Dorsal nuevo detectado con número
- 🟠 **Naranja** = Dorsal ya registrado anteriormente
- 🔴 **Rojo** = Dorsal detectado pero sin número legible

---

## 🔍 **DIFERENCIAS**

| Característica | mi_detector_registro.py | mi_detector_ocr.py |
|----------------|--------------------------|---------------------|
| Detecta dorsales | ✅ | ✅ |
| Lee números | ❌ (guarda "bib") | ✅ (guarda "123") |
| Requiere OCR | ❌ | ✅ |
| Velocidad | Más rápido | Más lento (OCR) |
| Precisión | N/A | Depende de calidad |

---

## ⚡ **EJEMPLO COMPLETO**

### **Paso 1: Instalar**
```powershell
# Instalar OCR (elige opción A)
.\instalar_ocr.ps1
```

### **Paso 2: Ejecutar**
```powershell
# Detector con lectura de números
python mi_detector_ocr.py --modo camara
```

### **Resultado en Excel:**
```
Posicion | Dorsal | HoraLlegada          | Observaciones
1        | 123    | 2025-10-03 16:45:12 |
2        | 456    | 2025-10-03 16:45:35 |
3        | 789    | 2025-10-03 16:46:01 |
```

---

## 🛠️ **OPCIONES ADICIONALES**

### **Archivo Excel personalizado**
```powershell
python mi_detector_ocr.py --modo camara --excel maraton_2025.xlsx
```

### **Procesar imagen**
```powershell
python mi_detector_ocr.py --modo imagen --archivo foto.jpg
```

### **Sin registro (solo detección)**
```powershell
python mi_detector_ocr.py --modo camara --sin-registro
```

---

## 📊 **MEJORAR PRECISIÓN OCR**

Si el OCR no lee bien los números:

### ✅ **Mejores Prácticas:**
1. **Distancia cámara:** 2-4 metros del dorsal
2. **Iluminación:** Evitar sombras y reflejos
3. **Ángulo:** Perpendicular (90°) al dorsal
4. **Velocidad:** Dorsales más lentos = mejor lectura
5. **Resolución:** Usar cámara HD (1280x720 mínimo)
6. **Contraste:** Fondo diferente al color del dorsal

### ⚠️ **Evitar:**
1. Dorsales arrugados o doblados
2. Números muy pequeños (<5cm altura)
3. Contraluz directo
4. Movimiento muy rápido
5. Números con fuentes extrañas

---

## 🐛 **SOLUCIÓN DE PROBLEMAS**

### ❌ **Error: "No module named 'easyocr'"**

**Solución:**
```powershell
.\instalar_ocr.ps1
```

---

### ❌ **OCR no detecta números (muestra "SIN NUMERO")**

**Causas:**
1. Dorsal muy lejos o pequeño
2. Mala iluminación
3. Dorsal arrugado
4. Movimiento muy rápido

**Soluciones:**
1. Acercar cámara (2-3 metros)
2. Mejorar iluminación
3. Pausar con ESPACIO y verificar
4. Ajustar resolución de cámara

---

### ❌ **Primera ejecución muy lenta (EasyOCR)**

**Es normal:**
- EasyOCR descarga modelos (~500MB) la primera vez
- Puede tardar 5-10 minutos dependiendo de tu internet
- Solo ocurre UNA VEZ
- Las siguientes ejecuciones son normales

**Solución:** Ten paciencia la primera vez.

---

### ❌ **Lee números incorrectos**

**Ajustar confianza:**

Edita `mi_detector_ocr.py` línea 48:

```python
# Cambiar de:
OCR_CONFIANZA_MIN = 0.3

# A (más estricto):
OCR_CONFIANZA_MIN = 0.5
```

---

## 📈 **RENDIMIENTO ESPERADO**

| Escenario | FPS | Tiempo OCR | Adecuado para |
|-----------|-----|------------|---------------|
| CPU + EasyOCR | 10-15 | ~0.3s | ✅ Carreras pedestres |
| CPU + Tesseract | 15-20 | ~0.2s | ✅ Carreras rápidas |
| GPU + EasyOCR | 20-30 | ~0.1s | ✅ Ciclismo |

**Nota:** FPS más bajo que detector sin OCR, pero lee el número real.

---

## ✅ **CHECKLIST**

- [ ] Instalé OCR con `.\instalar_ocr.ps1`
- [ ] Elegí EasyOCR (opción A)
- [ ] Primera ejecución completada (descargó modelos)
- [ ] Ejecuto `python mi_detector_ocr.py --modo camara`
- [ ] Veo números reales (no "bib") en pantalla
- [ ] Excel se actualiza con números correctos
- [ ] Ajusté distancia cámara (2-4 metros)
- [ ] Buena iluminación sin reflejos

---

## 🎯 **RESUMEN**

```powershell
# PROBLEMA: Se guarda "bib" en lugar del número

# SOLUCIÓN:
.\instalar_ocr.ps1          # Instalar OCR (opción A)
python mi_detector_ocr.py --modo camara  # Usar detector con OCR

# RESULTADO: Excel con números reales
# Posicion | Dorsal | HoraLlegada
# 1        | 123    | 2025-10-03 16:45:12
# 2        | 456    | 2025-10-03 16:45:35
```

---

**🏁 ¡Ahora sí registrarás los números reales de los dorsales!**

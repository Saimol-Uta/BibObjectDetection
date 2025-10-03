# 🔧 SOLUCIÓN: PyTorch no detecta GPU (CUDA disponible: False)

## 📋 Diagnóstico de tu Sistema

✅ **GPU NVIDIA RTX 3050**: Detectada correctamente  
✅ **Driver NVIDIA 580.88**: Instalado y funcionando  
✅ **CUDA 13.0**: Instalado en el sistema  
❌ **Python**: NO está instalado o NO está en PATH  
❌ **PyTorch**: No puede verificarse sin Python  

---

## 🎯 Problema Identificado

Tu sistema tiene **CUDA 13.0** pero:
1. Python no está instalado o no está en el PATH del sistema
2. El script de instalación falló porque no pudo ejecutar Python
3. PyTorch necesita ser instalado con soporte para CUDA 12.x (compatible con 13.0)

---

## 🚀 SOLUCIÓN PASO A PASO

### PASO 1: Instalar Python Correctamente

#### Opción A: Descargar desde python.org (RECOMENDADO)

1. **Descarga Python**:
   - Visita: https://www.python.org/downloads/
   - Descarga Python 3.10.x o 3.11.x (NO uses 3.12 aún, tiene problemas de compatibilidad)
   - Archivo recomendado: `python-3.10.11-amd64.exe`

2. **Instalar Python**:
   - ✅ **MUY IMPORTANTE**: Marca la casilla "Add Python to PATH"
   - Selecciona "Install Now"
   - Espera a que complete la instalación

3. **Verificar instalación**:
   - Cierra y reabre PowerShell
   - Ejecuta:
     ```powershell
     python --version
     ```
   - Deberías ver: `Python 3.10.11` (o la versión que instalaste)

#### Opción B: Usar Microsoft Store

1. Abre Microsoft Store
2. Busca "Python 3.10"
3. Instala Python 3.10
4. Verifica con `python --version`

---

### PASO 2: Crear Entorno Virtual

Una vez que Python esté instalado:

```powershell
# Navegar al proyecto
cd "D:\Univercidad\ModeloDetecion\BibObjectDetection"

# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
.\venv\Scripts\Activate.ps1
```

Si obtienes error de "ejecución de scripts deshabilitada":
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

### PASO 3: Instalar PyTorch con CUDA 12.x (Compatible con tu CUDA 13.0)

Con el entorno virtual activado:

```powershell
# Actualizar pip
python -m pip install --upgrade pip

# Instalar PyTorch con CUDA 12.1 (compatible con CUDA 13.0)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**NOTA**: Usamos `cu121` (CUDA 12.1) en lugar de `cu118` porque:
- Tu sistema tiene CUDA 13.0
- PyTorch con CUDA 12.1 es compatible con CUDA 13.0
- PyTorch aún no tiene binarios específicos para CUDA 13.0

---

### PASO 4: Verificar que PyTorch Detecta la GPU

```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

**Resultado esperado**:
```
PyTorch: 2.x.x+cu121
CUDA disponible: True
CUDA version: 12.1
GPU: NVIDIA GeForce RTX 3050 Laptop GPU
```

---

### PASO 5: Instalar el Resto de Dependencias

```powershell
# Con el entorno virtual activado

# OpenCV
pip install opencv-python opencv-contrib-python

# Librerías científicas
pip install numpy pandas scipy h5py matplotlib

# Aumento de datos
pip install imgaug

# Jupyter
pip install jupyter notebook ipython ipykernel

# Utilidades
pip install tqdm Pillow
```

---

### PASO 6: Ejecutar Script de Verificación

```powershell
python verificar_instalacion.py
```

Debería mostrar todas las verificaciones en verde ✅

---

## 🔄 Script de Instalación Corregido

He aquí un script PowerShell actualizado que puedes usar:

```powershell
# instalacion_corregida.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  INSTALACIÓN CORREGIDA" -ForegroundColor Cyan
Write-Host "  Para CUDA 13.0 + RTX 3050" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Verificar Python
Write-Host "[1/5] Verificando Python..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ $pythonVersion encontrado" -ForegroundColor Green
} else {
    Write-Host "✗ Python no encontrado." -ForegroundColor Red
    Write-Host ""
    Write-Host "SOLUCIÓN:" -ForegroundColor Yellow
    Write-Host "1. Descarga Python desde: https://www.python.org/downloads/" -ForegroundColor White
    Write-Host "2. Durante instalación, MARCA 'Add Python to PATH'" -ForegroundColor White
    Write-Host "3. Cierra y reabre PowerShell" -ForegroundColor White
    Write-Host "4. Ejecuta este script nuevamente" -ForegroundColor White
    exit 1
}

# Crear entorno virtual
Write-Host ""
Write-Host "[2/5] Creando entorno virtual..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "⚠ Eliminando entorno virtual anterior..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "venv"
}
python -m venv venv
Write-Host "✓ Entorno virtual creado" -ForegroundColor Green

# Activar entorno virtual
Write-Host ""
Write-Host "[3/5] Activando entorno virtual..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Actualizar pip
Write-Host ""
Write-Host "[4/5] Actualizando pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Instalar dependencias
Write-Host ""
Write-Host "[5/5] Instalando dependencias (puede tomar 5-10 minutos)..." -ForegroundColor Yellow

Write-Host "  -> Instalando PyTorch con CUDA 12.1..." -ForegroundColor Cyan
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Write-Host "  -> Instalando OpenCV..." -ForegroundColor Cyan
pip install opencv-python opencv-contrib-python

Write-Host "  -> Instalando paquetes científicos..." -ForegroundColor Cyan
pip install numpy pandas scipy h5py matplotlib imgaug

Write-Host "  -> Instalando Jupyter..." -ForegroundColor Cyan
pip install jupyter notebook ipython ipykernel

Write-Host "  -> Instalando utilidades..." -ForegroundColor Cyan
pip install tqdm Pillow

# Verificar PyTorch + CUDA
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  VERIFICACIÓN" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$cudaCheck = python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')" 2>&1

Write-Host $cudaCheck

if ($cudaCheck -like "*True*") {
    Write-Host ""
    Write-Host "✅ ¡INSTALACIÓN EXITOSA!" -ForegroundColor Green
    Write-Host "Tu GPU RTX 3050 está lista para usar" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "⚠️  PyTorch instalado pero GPU no detectada" -ForegroundColor Yellow
    Write-Host "Consulta: SOLUCION_GPU_NO_DETECTADA.md" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Próximos pasos:" -ForegroundColor Cyan
Write-Host "  cd notebooks+utils+data" -ForegroundColor White
Write-Host "  jupyter notebook" -ForegroundColor White
```

---

## 🆘 Si PyTorch Sigue Sin Detectar la GPU

### Verificación 1: Compatibilidad de Versiones

```powershell
# Verificar versión de CUDA del sistema
nvidia-smi

# Verificar versión de CUDA en PyTorch
python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')"
```

### Verificación 2: Reinstalar PyTorch

Si instalaste con CUDA 11.8 antes:

```powershell
# Desinstalar PyTorch antiguo
pip uninstall torch torchvision torchaudio -y

# Instalar con CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Verificación 3: Probar con CUDA 11.8

Si CUDA 12.1 no funciona, intenta con 11.8:

```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**NOTA**: CUDA 11.8 generalmente funciona incluso con drivers más nuevos por retrocompatibilidad.

---

## 📝 Resumen de Versiones Recomendadas

```
Sistema Operativo: Windows 10/11 64-bit
Python: 3.10.x o 3.11.x
CUDA del sistema: 13.0 (ya instalado ✅)
PyTorch: Última versión con CUDA 12.1 o 11.8
Driver NVIDIA: 580.88 (ya instalado ✅)
```

---

## ✅ Checklist de Solución

- [ ] Python instalado correctamente (marca "Add to PATH")
- [ ] `python --version` funciona en PowerShell
- [ ] Entorno virtual creado (`python -m venv venv`)
- [ ] Entorno virtual activado (`.\venv\Scripts\Activate.ps1`)
- [ ] PyTorch instalado con CUDA 12.1 o 11.8
- [ ] Verificación exitosa: `torch.cuda.is_available()` retorna `True`
- [ ] GPU detectada correctamente

---

## 🎯 Comando de Verificación Final

Ejecuta este comando después de todo:

```powershell
python -c "import torch; import sys; print(f'Python: {sys.version}'); print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'Nombre GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}'); print(f'Compute Capability: {torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None}')"
```

**Salida esperada**:
```
Python: 3.10.11 (o similar)
PyTorch: 2.x.x+cu121
CUDA disponible: True
Nombre GPU: NVIDIA GeForce RTX 3050 Laptop GPU
Compute Capability: (8, 6)
```

---

## 📞 ¿Necesitas Ayuda Adicional?

Si después de seguir estos pasos sigue sin funcionar:

1. Ejecuta: `nvidia-smi` y copia la salida
2. Ejecuta: `python -c "import torch; print(torch.__version__)"` y copia la salida
3. Revisa si hay algún error específico en PowerShell

---

**Creado: Octubre 2025**  
**Para: NVIDIA GeForce RTX 3050 con CUDA 13.0**

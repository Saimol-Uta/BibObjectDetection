"""
Script de Verificación de Instalación
Detecta automáticamente si todos los requisitos están instalados correctamente
para ejecutar el proyecto de Detección de Números de Dorsal
"""

import sys
import subprocess
import platform

def print_header(text):
    """Imprime un encabezado formateado"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_success(text):
    """Imprime mensaje de éxito"""
    print(f"✓ {text}")

def print_error(text):
    """Imprime mensaje de error"""
    print(f"✗ {text}")

def print_warning(text):
    """Imprime mensaje de advertencia"""
    print(f"⚠ {text}")

def check_python_version():
    """Verifica la versión de Python"""
    print_header("Verificando Python")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 7:
        print_success(f"Python {version.major}.{version.minor} es compatible")
        return True
    else:
        print_error(f"Se requiere Python 3.7 o superior")
        return False

def check_nvidia_driver():
    """Verifica que el driver NVIDIA esté instalado"""
    print_header("Verificando Driver NVIDIA")
    try:
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            # Extraer información de la GPU
            lines = result.stdout.split('\n')
            for line in lines:
                if 'RTX' in line or 'GeForce' in line:
                    print_success(f"GPU detectada: {line.strip()}")
                    break
            return True
        else:
            print_error("nvidia-smi no se ejecutó correctamente")
            return False
    except FileNotFoundError:
        print_error("nvidia-smi no encontrado. Instala los drivers NVIDIA")
        return False
    except subprocess.TimeoutExpired:
        print_error("nvidia-smi timeout")
        return False
    except Exception as e:
        print_error(f"Error al ejecutar nvidia-smi: {e}")
        return False

def check_cuda():
    """Verifica que CUDA esté instalado"""
    print_header("Verificando CUDA")
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            # Buscar versión de CUDA
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    print_success(f"CUDA instalado: {line.strip()}")
                    return True
            print_success("CUDA instalado (versión no detectada)")
            return True
        else:
            print_error("nvcc no se ejecutó correctamente")
            return False
    except FileNotFoundError:
        print_error("nvcc no encontrado. Instala CUDA Toolkit")
        print("  Descarga desde: https://developer.nvidia.com/cuda-downloads")
        return False
    except Exception as e:
        print_error(f"Error al verificar CUDA: {e}")
        return False

def check_python_package(package_name, import_name=None):
    """Verifica si un paquete de Python está instalado"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'versión desconocida')
        print_success(f"{package_name}: {version}")
        return True
    except ImportError:
        print_error(f"{package_name} no instalado")
        return False

def check_pytorch_cuda():
    """Verifica que PyTorch pueda usar CUDA"""
    print_header("Verificando PyTorch + CUDA")
    try:
        import torch
        print_success(f"PyTorch versión: {torch.__version__}")
        
        if torch.cuda.is_available():
            print_success(f"CUDA disponible en PyTorch: {torch.version.cuda}")
            print_success(f"GPU detectada: {torch.cuda.get_device_name(0)}")
            print_success(f"Número de GPUs: {torch.cuda.device_count()}")
            
            # Probar operación en GPU
            try:
                x = torch.rand(3, 3).cuda()
                print_success("Operación de prueba en GPU exitosa")
                return True
            except Exception as e:
                print_error(f"Error al ejecutar operación en GPU: {e}")
                return False
        else:
            print_error("CUDA no está disponible en PyTorch")
            print_warning("Verifica que instalaste PyTorch con soporte CUDA:")
            print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            return False
    except ImportError:
        print_error("PyTorch no está instalado")
        return False

def check_opencv_cuda():
    """Verifica si OpenCV puede usar CUDA"""
    print_header("Verificando OpenCV")
    try:
        import cv2
        print_success(f"OpenCV versión: {cv2.__version__}")
        
        # Verificar si OpenCV tiene soporte CUDA
        try:
            cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
            if cuda_count > 0:
                print_success(f"OpenCV detecta {cuda_count} dispositivo(s) CUDA")
                return True
            else:
                print_warning("OpenCV instalado pero sin detectar dispositivos CUDA")
                print_warning("Esto es normal si usaste opencv-python estándar")
                print_warning("La detección funcionará pero puede ser más lenta")
                return True
        except:
            print_warning("OpenCV sin soporte CUDA compilado")
            print_warning("La detección funcionará pero puede ser más lenta")
            return True
    except ImportError:
        print_error("OpenCV no está instalado")
        return False

def check_python_packages():
    """Verifica todos los paquetes de Python necesarios"""
    print_header("Verificando Paquetes de Python")
    
    packages = [
        ('numpy', 'numpy'),
        ('h5py', 'h5py'),
        ('matplotlib', 'matplotlib'),
        ('scipy', 'scipy'),
        ('pandas', 'pandas'),
        ('imgaug', 'imgaug'),
        ('jupyter', 'jupyter'),
        ('ipython', 'IPython'),
    ]
    
    results = []
    for package_name, import_name in packages:
        result = check_python_package(package_name, import_name)
        results.append(result)
    
    return all(results)

def check_project_files():
    """Verifica que los archivos del proyecto existan"""
    print_header("Verificando Archivos del Proyecto")
    
    import os
    
    required_files = [
        ('notebooks+utils+data/utils.py', 'Script de utilidades'),
        ('weights-classes/RBNR_custom-yolov4-tiny-detector_best.weights', 'Pesos RBNR'),
        ('weights-classes/RBNR_custom-yolov4-tiny-detector.cfg', 'Config RBNR'),
        ('weights-classes/RBRN_obj.names', 'Clases RBNR'),
        ('weights-classes/SVHN_custom-yolov4-tiny-detector_best.weights', 'Pesos SVHN'),
        ('weights-classes/SVHN_custom-yolov4-tiny-detector.cfg', 'Config SVHN'),
        ('weights-classes/SVHN_obj.names', 'Clases SVHN'),
    ]
    
    results = []
    for file_path, description in required_files:
        if os.path.exists(file_path):
            print_success(f"{description}: {file_path}")
            results.append(True)
        else:
            print_error(f"{description} no encontrado: {file_path}")
            results.append(False)
    
    return all(results)

def check_notebooks():
    """Lista los notebooks disponibles"""
    print_header("Notebooks Disponibles")
    
    import os
    import glob
    
    notebooks = glob.glob('notebooks+utils+data/*.ipynb')
    
    if notebooks:
        for nb in sorted(notebooks):
            print_success(os.path.basename(nb))
        return True
    else:
        print_error("No se encontraron notebooks")
        return False

def print_summary(results):
    """Imprime un resumen de los resultados"""
    print_header("RESUMEN DE VERIFICACIÓN")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nPruebas pasadas: {passed}/{total}")
    
    if passed == total:
        print("\n" + "🎉 " * 10)
        print_success("¡TODAS LAS VERIFICACIONES PASARON!")
        print_success("Tu sistema está listo para ejecutar el proyecto")
        print("🎉 " * 10)
        print("\nPróximos pasos:")
        print("  1. Activa el entorno virtual: .\\venv\\Scripts\\Activate.ps1")
        print("  2. Navega a notebooks: cd notebooks+utils+data")
        print("  3. Inicia Jupyter: jupyter notebook")
        print("  4. Abre: 05 - Bib Detection Validation & Demo.ipynb")
    else:
        print("\n" + "⚠️  " * 10)
        print_warning(f"Algunas verificaciones fallaron ({total - passed} de {total})")
        print_warning("Revisa los errores anteriores y consulta el MANUAL_INSTALACION.md")
        print("⚠️  " * 10)
        
        # Recomendaciones basadas en fallos
        print("\nRecomendaciones:")
        if not results.get('nvidia_driver'):
            print("  • Instala/actualiza drivers NVIDIA desde:")
            print("    https://www.nvidia.com/Download/index.aspx")
        if not results.get('cuda'):
            print("  • Instala CUDA Toolkit desde:")
            print("    https://developer.nvidia.com/cuda-downloads")
        if not results.get('pytorch_cuda'):
            print("  • Reinstala PyTorch con CUDA:")
            print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        if not results.get('packages'):
            print("  • Instala paquetes faltantes:")
            print("    pip install numpy h5py matplotlib scipy pandas imgaug jupyter ipython")
        if not results.get('opencv'):
            print("  • Instala OpenCV:")
            print("    pip install opencv-python opencv-contrib-python")
        if not results.get('project_files'):
            print("  • Verifica que descargaste todos los archivos del proyecto")

def main():
    """Función principal"""
    print("\n" + "🔍 " * 20)
    print("VERIFICADOR DE INSTALACIÓN - Detección de Números de Dorsal")
    print("🔍 " * 20)
    
    print(f"\nSistema Operativo: {platform.system()} {platform.release()}")
    print(f"Arquitectura: {platform.machine()}")
    
    results = {}
    
    # Verificaciones
    results['python'] = check_python_version()
    results['nvidia_driver'] = check_nvidia_driver()
    results['cuda'] = check_cuda()
    results['pytorch_cuda'] = check_pytorch_cuda()
    results['opencv'] = check_opencv_cuda()
    results['packages'] = check_python_packages()
    results['project_files'] = check_project_files()
    results['notebooks'] = check_notebooks()
    
    # Resumen
    print_summary(results)
    
    return all(results.values())

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

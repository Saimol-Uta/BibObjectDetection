Solución: OpenCV sin soporte GUI (cv2.imshow) en Windows

Problema

Al ejecutar el detector aparece este error de OpenCV:

  error: The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support.

Significa que la versión instalada de OpenCV fue la "headless" o no incluye soporte GUI, por ejemplo `opencv-python-headless` o una build sin Qt/Win32.

Opciones para arreglarlo (Windows)

1) Reinstalar usando pip (recomendado para la mayoría de usuarios)

  # Desinstalar variantes que puedan causar conflicto
  pip uninstall -y opencv-python-headless opencv-python opencv-contrib-python

  # Instalar la versión completa con contrib (incluye soporte GUI)
  pip install opencv-contrib-python

Explicación: `opencv-contrib-python` incluye el paquete principal con módulos contrib y soporte GUI en Windows. Evita instalar `opencv-python-headless` si quieres usar `cv2.imshow`.

2) Usar conda (si usas Anaconda/Miniconda)

  conda install -c conda-forge opencv

Conda habitualmente provee builds con soporte GUI integrados.

Solución temporal: modo headless del detector (ya incluido)

El script `mi_detector_registro.py` detecta si OpenCV tiene soporte GUI. Si no lo tiene:

- No intentará abrir ventanas con `cv2.imshow`.
- Guardará una imagen de preview en `output/headless_preview.jpg` cada 5 segundos.
- Guardará detecciones en `output/deteccion_YYYYMMDD_HHMMSS.jpg` como normalmente.

Esto permite ejecutar el detector sin interfaz gráfica y revisar archivos en la carpeta `output/`.

Siguientes pasos

- Si quieres que yo aplique un cambio adicional (por ejemplo forzar import try/except para evitar errores en entornos sin numpy/cv2), dime y lo aplico.
- Si quieres que pruebe reinstalar paquetes en tu entorno, puedo darte el comando exacto para ejecutar en PowerShell.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SISTEMA DE REGISTRO DE LLEGADAS - CARRERAS
Registra dorsales detectados en archivo Excel con timestamp
Evita duplicados y mantiene orden de llegada
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import threading


class RegistroLlegadas:
    """Sistema de registro de llegadas para carreras"""
    
    def __init__(self, archivo_excel="registro_llegadas.xlsx", permitir_duplicados=False):
        """
        Inicializa el sistema de registro
        
        Args:
            archivo_excel: Ruta del archivo Excel
            permitir_duplicados: Si False, ignora dorsales ya registrados
        """
        self.archivo_excel = Path(archivo_excel)
        self.permitir_duplicados = permitir_duplicados
        self.lock = threading.Lock()  # Para operaciones thread-safe
        
        # Inicializar o cargar archivo
        self._inicializar_archivo()
        
        print(f"✓ Sistema de registro inicializado")
        print(f"  Archivo: {self.archivo_excel}")
        print(f"  Duplicados: {'Permitidos' if permitir_duplicados else 'Bloqueados'}")
    
    def _inicializar_archivo(self):
        """Crea el archivo Excel si no existe"""
        if not self.archivo_excel.exists():
            # Crear DataFrame vacio con columnas
            df = pd.DataFrame(columns=['Posicion', 'Dorsal', 'HoraLlegada', 'Observaciones'])
            df.to_excel(self.archivo_excel, index=False, engine='openpyxl')
            print(f"  ✓ Archivo Excel creado: {self.archivo_excel}")
    
    def registrar_llegada(self, dorsal, observaciones=""):
        """
        Registra una llegada en el Excel
        
        Args:
            dorsal: Número de dorsal detectado (string o int)
            observaciones: Campo opcional para notas
            
        Returns:
            dict: Información de la llegada registrada o None si es duplicado
        """
        with self.lock:
            try:
                # Convertir dorsal a string para consistencia
                dorsal = str(dorsal)
                
                # Leer archivo actual
                df = pd.read_excel(self.archivo_excel, engine='openpyxl')
                
                # Verificar duplicados
                if not self.permitir_duplicados:
                    if dorsal in df['Dorsal'].astype(str).values:
                        print(f"  [!] Dorsal {dorsal} ya registrado - Ignorado")
                        # Retornar info del registro existente
                        fila_existente = df[df['Dorsal'].astype(str) == dorsal].iloc[0]
                        return {
                            'posicion': int(fila_existente['Posicion']),
                            'dorsal': dorsal,
                            'hora': fila_existente['HoraLlegada'],
                            'observaciones': fila_existente['Observaciones'],
                            'duplicado': True
                        }
                
                # Calcular nueva posición
                nueva_posicion = len(df) + 1
                
                # Obtener hora actual
                hora_llegada = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Crear nueva fila
                nueva_fila = {
                    'Posicion': nueva_posicion,
                    'Dorsal': dorsal,
                    'HoraLlegada': hora_llegada,
                    'Observaciones': observaciones
                }
                
                # Añadir al DataFrame
                df = pd.concat([df, pd.DataFrame([nueva_fila])], ignore_index=True)
                
                # Guardar archivo
                df.to_excel(self.archivo_excel, index=False, engine='openpyxl')
                
                # Mensaje de confirmación
                print(f"  ✓ LLEGADA REGISTRADA:")
                print(f"    Posición: {nueva_posicion}")
                print(f"    Dorsal: {dorsal}")
                print(f"    Hora: {hora_llegada}")
                if observaciones:
                    print(f"    Obs: {observaciones}")
                
                return {
                    'posicion': nueva_posicion,
                    'dorsal': dorsal,
                    'hora': hora_llegada,
                    'observaciones': observaciones,
                    'duplicado': False
                }
                
            except Exception as e:
                print(f"  [X] Error al registrar llegada: {e}")
                return None
    
    def actualizar_observaciones(self, dorsal, observaciones):
        """
        Actualiza las observaciones de un dorsal ya registrado
        
        Args:
            dorsal: Número de dorsal
            observaciones: Nuevo texto de observaciones
            
        Returns:
            bool: True si se actualizó correctamente
        """
        with self.lock:
            try:
                dorsal = str(dorsal)
                
                # Leer archivo
                df = pd.read_excel(self.archivo_excel, engine='openpyxl')
                
                # Buscar dorsal
                if dorsal not in df['Dorsal'].astype(str).values:
                    print(f"  [!] Dorsal {dorsal} no encontrado")
                    return False
                
                # Actualizar observaciones
                df.loc[df['Dorsal'].astype(str) == dorsal, 'Observaciones'] = observaciones
                
                # Guardar
                df.to_excel(self.archivo_excel, index=False, engine='openpyxl')
                
                print(f"  ✓ Observaciones actualizadas para dorsal {dorsal}")
                return True
                
            except Exception as e:
                print(f"  [X] Error al actualizar: {e}")
                return False
    
    def obtener_estadisticas(self):
        """
        Obtiene estadísticas del registro actual
        
        Returns:
            dict: Estadísticas
        """
        try:
            df = pd.read_excel(self.archivo_excel, engine='openpyxl')
            
            return {
                'total_llegadas': len(df),
                'primer_dorsal': df.iloc[0]['Dorsal'] if len(df) > 0 else None,
                'ultimo_dorsal': df.iloc[-1]['Dorsal'] if len(df) > 0 else None,
                'primera_hora': df.iloc[0]['HoraLlegada'] if len(df) > 0 else None,
                'ultima_hora': df.iloc[-1]['HoraLlegada'] if len(df) > 0 else None
            }
        except Exception as e:
            print(f"  [X] Error al obtener estadísticas: {e}")
            return None
    
    def obtener_posicion(self, dorsal):
        """
        Obtiene la posición de un dorsal
        
        Args:
            dorsal: Número de dorsal
            
        Returns:
            int: Posición o None si no está registrado
        """
        try:
            dorsal = str(dorsal)
            df = pd.read_excel(self.archivo_excel, engine='openpyxl')
            
            if dorsal in df['Dorsal'].astype(str).values:
                fila = df[df['Dorsal'].astype(str) == dorsal].iloc[0]
                return int(fila['Posicion'])
            return None
            
        except Exception as e:
            print(f"  [X] Error: {e}")
            return None
    
    def listar_llegadas(self, ultimas=10):
        """
        Lista las últimas llegadas registradas
        
        Args:
            ultimas: Número de llegadas a mostrar (0 = todas)
            
        Returns:
            DataFrame: Últimas llegadas
        """
        try:
            df = pd.read_excel(self.archivo_excel, engine='openpyxl')
            
            if ultimas > 0:
                return df.tail(ultimas)
            return df
            
        except Exception as e:
            print(f"  [X] Error: {e}")
            return None
    
    def resetear_registro(self):
        """
        Resetea el registro (CUIDADO: borra todos los datos)
        """
        try:
            df = pd.DataFrame(columns=['Posicion', 'Dorsal', 'HoraLlegada', 'Observaciones'])
            df.to_excel(self.archivo_excel, index=False, engine='openpyxl')
            print(f"  ✓ Registro reseteado")
            return True
        except Exception as e:
            print(f"  [X] Error al resetear: {e}")
            return False


# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def mostrar_estadisticas(registro):
    """Muestra estadísticas del registro"""
    print("\n" + "="*70)
    print("  ESTADÍSTICAS DEL REGISTRO")
    print("="*70)
    
    stats = registro.obtener_estadisticas()
    if stats:
        print(f"Total llegadas: {stats['total_llegadas']}")
        if stats['total_llegadas'] > 0:
            print(f"Primer lugar: Dorsal {stats['primer_dorsal']} - {stats['primera_hora']}")
            print(f"Último registro: Dorsal {stats['ultimo_dorsal']} - {stats['ultima_hora']}")
    
    print("="*70)


def mostrar_ultimas_llegadas(registro, n=10):
    """Muestra las últimas n llegadas"""
    print("\n" + "="*70)
    print(f"  ÚLTIMAS {n} LLEGADAS")
    print("="*70)
    
    df = registro.listar_llegadas(ultimas=n)
    if df is not None and len(df) > 0:
        print(df.to_string(index=False))
    else:
        print("  No hay llegadas registradas")
    
    print("="*70)


# ============================================================================
# PRUEBA DEL SISTEMA
# ============================================================================

def prueba_sistema():
    """Función de prueba del sistema de registro"""
    print("\n" + "="*70)
    print("  PRUEBA DEL SISTEMA DE REGISTRO")
    print("="*70 + "\n")
    
    # Crear instancia
    registro = RegistroLlegadas(
        archivo_excel="test_registro_llegadas.xlsx",
        permitir_duplicados=False
    )
    
    print("\n--- Registrando llegadas de prueba ---\n")
    
    # Simular detecciones
    llegadas_prueba = [
        ("123", ""),
        ("456", "Primer lugar categoria master"),
        ("789", ""),
        ("123", "Intento duplicado"),  # Duplicado - será rechazado
        ("321", "Llegada correcta"),
    ]
    
    for dorsal, obs in llegadas_prueba:
        resultado = registro.registrar_llegada(dorsal, obs)
        print()
    
    # Mostrar estadísticas
    mostrar_estadisticas(registro)
    
    # Mostrar últimas llegadas
    mostrar_ultimas_llegadas(registro, n=5)
    
    # Actualizar observaciones
    print("\n--- Actualizando observaciones ---\n")
    registro.actualizar_observaciones("456", "Actualizado: Ganador Master 40+")
    
    # Consultar posición
    print("\n--- Consultas ---\n")
    pos = registro.obtener_posicion("456")
    print(f"Dorsal 456 llegó en posición: {pos}")
    
    print("\n" + "="*70)
    print(f"  Archivo generado: test_registro_llegadas.xlsx")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Verificar que openpyxl esté instalado
    try:
        import openpyxl
    except ImportError:
        print("[X] Error: openpyxl no está instalado")
        print("    Instala con: pip install openpyxl pandas")
        exit(1)
    
    # Ejecutar prueba
    prueba_sistema()

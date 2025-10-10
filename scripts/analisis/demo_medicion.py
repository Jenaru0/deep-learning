"""
Demo del Sistema de Medición de Parámetros Estructurales
=========================================================

Script de demostración que procesa imágenes de prueba del dataset CRACK500
y genera análisis completo con visualizaciones.

Uso:
    python demo_medicion.py [--num-imagenes N] [--pixeles-por-mm X]
"""

import os
import sys
from pathlib import Path
import random

# Añadir ruta raíz
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from scripts.analisis.medir_parametros import (
    configurar_gpu,
    ModeloSegmentacion,
    analizar_fisura_completo
)
from config import RUTA_CRACK500


def obtener_imagenes_muestra(num_imagenes: int = 5):
    """Obtiene imágenes aleatorias del dataset CRACK500 para demo."""
    test_dir = os.path.join(RUTA_CRACK500, 'images')
    
    if not os.path.exists(test_dir):
        print(f"❌ No se encontró el directorio: {test_dir}")
        return []
    
    # Obtener todas las imágenes .jpg
    imagenes = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
    
    if len(imagenes) == 0:
        print(f"❌ No se encontraron imágenes en: {test_dir}")
        return []
    
    # Seleccionar muestra aleatoria
    muestra = random.sample(imagenes, min(num_imagenes, len(imagenes)))
    
    return [os.path.join(test_dir, img) for img in muestra]


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Demo de análisis de parámetros de fisuras')
    parser.add_argument('--num-imagenes', type=int, default=5,
                        help='Número de imágenes a procesar (default: 5)')
    parser.add_argument('--pixeles-por-mm', type=float, default=1.0,
                        help='Calibración píxeles/mm (default: 1.0)')
    parser.add_argument('--umbral', type=float, default=0.5,
                        help='Umbral de segmentación (default: 0.5)')
    
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║       DEMO: SISTEMA DE MEDICIÓN DE PARÁMETROS ESTRUCTURALES         ║
║                                                                      ║
║  📏 Medición de ancho (skeletonization + distance transform)        ║
║  🧭 Detección de orientación (Hough line transform)                 ║
║  🕳️  Estimación de profundidad visual (análisis de intensidad)      ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    # Configurar GPU
    configurar_gpu()
    
    # Obtener imágenes de muestra
    print(f"\n📂 Buscando {args.num_imagenes} imágenes de muestra...")
    imagenes = obtener_imagenes_muestra(args.num_imagenes)
    
    if len(imagenes) == 0:
        print("❌ No se encontraron imágenes para procesar")
        return
    
    print(f"   ✅ {len(imagenes)} imágenes seleccionadas")
    
    # Cargar modelo una sola vez
    print("\n🔄 Cargando modelo U-Net...")
    modelo = ModeloSegmentacion()
    modelo.cargar()
    
    # Procesar cada imagen
    resultados_totales = []
    
    for i, imagen_path in enumerate(imagenes, 1):
        print(f"\n{'='*70}")
        print(f"Procesando imagen {i}/{len(imagenes)}")
        print(f"{'='*70}")
        
        try:
            resultado = analizar_fisura_completo(
                imagen_path=imagen_path,
                modelo=modelo,
                pixeles_por_mm=args.pixeles_por_mm,
                umbral_segmentacion=args.umbral,
                guardar_visualizacion=True
            )
            
            resultados_totales.append(resultado)
            
        except Exception as e:
            print(f"❌ Error procesando {os.path.basename(imagen_path)}: {e}")
            continue
    
    # Resumen final
    print(f"\n{'='*70}")
    print(f"📊 RESUMEN FINAL")
    print(f"{'='*70}")
    print(f"Imágenes procesadas: {len(resultados_totales)}/{len(imagenes)}")
    
    if len(resultados_totales) > 0:
        # Estadísticas de ancho
        anchos_promedio = [r['ancho']['ancho_promedio_mm'] for r in resultados_totales 
                          if r['ancho']['ancho_promedio_mm'] > 0]
        
        if anchos_promedio:
            print(f"\n📏 Estadísticas de Ancho:")
            print(f"   • Promedio general: {sum(anchos_promedio)/len(anchos_promedio):.2f} mm")
            print(f"   • Máximo encontrado: {max(anchos_promedio):.2f} mm")
            print(f"   • Mínimo encontrado: {min(anchos_promedio):.2f} mm")
        
        # Distribución de orientaciones
        orientaciones = [r['orientacion']['orientacion'] for r in resultados_totales]
        from collections import Counter
        dist_orientacion = Counter(orientaciones)
        
        print(f"\n🧭 Distribución de Orientaciones:")
        for orient, count in dist_orientacion.items():
            porcentaje = (count / len(orientaciones)) * 100
            print(f"   • {orient}: {count} ({porcentaje:.1f}%)")
        
        # Distribución de profundidades
        profundidades = [r['profundidad']['profundidad_categoria'] for r in resultados_totales]
        dist_profundidad = Counter(profundidades)
        
        print(f"\n🕳️  Distribución de Profundidades (Visual):")
        for prof, count in dist_profundidad.items():
            porcentaje = (count / len(profundidades)) * 100
            print(f"   • {prof}: {count} ({porcentaje:.1f}%)")
        
        # Ubicación de resultados
        dir_salida = os.path.join(ROOT_DIR, 'resultados', 'analisis_parametros')
        print(f"\n💾 Visualizaciones guardadas en:")
        print(f"   {dir_salida}")
    
    print(f"\n{'='*70}")
    print(f"✅ DEMO COMPLETADA")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

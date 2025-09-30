"""
Prueba Rápida del Clasificador Mejorado
=======================================
"""

import sys
sys.path.append('.')

import importlib.util
import numpy as np
from pathlib import Path

# Importar el clasificador
spec = importlib.util.spec_from_file_location("severity_classifier", "src/models/03_severity_classifier.py")
severity_classifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(severity_classifier)
CrackSeverityClassifier = severity_classifier.CrackSeverityClassifier

def quick_test():
    """Prueba rápida con una imagen"""
    print("🧪 PRUEBA RÁPIDA DEL CLASIFICADOR MEJORADO")
    print("=" * 50)
    
    # Crear clasificador
    classifier = CrackSeverityClassifier()
    
    # Buscar imágenes de prueba
    test_crack_dir = Path("data/processed/sdnet2018_prepared/test/crack")
    test_no_crack_dir = Path("data/processed/sdnet2018_prepared/test/no_crack")
    
    crack_images = list(test_crack_dir.glob("*.jpg"))[:2]  # Solo 2 para prueba rápida
    no_crack_images = list(test_no_crack_dir.glob("*.jpg"))[:1]  # Solo 1
    
    all_test_images = crack_images + no_crack_images
    
    if not all_test_images:
        print("❌ No se encontraron imágenes de prueba")
        return
    
    print(f"📸 Probando con {len(all_test_images)} imágenes...")
    print()
    
    for i, img_path in enumerate(all_test_images, 1):
        print(f"{'='*40}")
        print(f"🔍 IMAGEN {i}: {img_path.name}")
        print(f"Categoría esperada: {'CRACK' if 'crack' in str(img_path) else 'NO_CRACK'}")
        print(f"{'='*40}")
        
        try:
            # Analizar imagen sin visualización
            result = classifier.analyze_single_image(str(img_path), show_analysis=False)
            
            # Mostrar resultados
            print(f"✅ Detección: {'SÍ' if result['has_crack'] else 'NO'}")
            print(f"📊 Probabilidad CNN: {result['detection_probability']:.3f}")
            print(f"⚠️  Severidad: {result['severity'].upper()}")
            print(f"🔧 Método: {result.get('method', 'N/A')}")
            
            if result.get('note'):
                print(f"📝 Nota: {result['note']}")
            
            if result['has_crack'] and result['crack_analysis']:
                print(f"📏 Densidad: {result['crack_analysis']['crack_density']:.4f}")
                print(f"📈 Contornos: {result['crack_analysis']['total_contours']}")
                print(f"🎯 Confianza: {result['severity_info'].get('confidence', 'N/A')}")
            
            print()
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print()
    
    print("🎉 PRUEBA COMPLETADA")
    print("✅ Clasificador mejorado funcionando")

if __name__ == "__main__":
    quick_test()
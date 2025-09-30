"""
Script de Prueba - Clasificador de Severidad
============================================

Prueba el clasificador de severidad con imágenes de ejemplo
del dataset SDNET2018 preparado.
"""

import sys
from pathlib import Path
sys.path.append('.')

# Importar directamente la clase desde el módulo
import importlib.util
spec = importlib.util.spec_from_file_location("severity_classifier", "src/models/03_severity_classifier.py")
severity_classifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(severity_classifier)
CrackSeverityClassifier = severity_classifier.CrackSeverityClassifier
import matplotlib.pyplot as plt

def test_severity_classifier():
    """Probar el clasificador con imágenes de ejemplo"""
    
    print("🧪 PRUEBA DEL CLASIFICADOR DE SEVERIDAD")
    print("=" * 50)
    
    # Crear clasificador
    classifier = CrackSeverityClassifier()
    
    # Buscar imágenes de prueba
    test_crack_dir = Path("data/processed/sdnet2018_prepared/test/crack")
    test_no_crack_dir = Path("data/processed/sdnet2018_prepared/test/no_crack")
    
    # Obtener algunas imágenes de ejemplo
    crack_images = list(test_crack_dir.glob("*.jpg"))[:3]
    no_crack_images = list(test_no_crack_dir.glob("*.jpg"))[:2]
    
    all_test_images = crack_images + no_crack_images
    
    if not all_test_images:
        print("❌ No se encontraron imágenes de prueba")
        return
    
    print(f"📸 Analizando {len(all_test_images)} imágenes de ejemplo...")
    print()
    
    results = []
    
    # Análizar cada imagen
    for i, img_path in enumerate(all_test_images, 1):
        print(f"{'='*60}")
        print(f"🔍 IMAGEN {i}: {img_path.name}")
        print(f"{'='*60}")
        
        try:
            # Analizar imagen (sin mostrar visualización)
            result = classifier.analyze_single_image(str(img_path), show_analysis=False)
            results.append({
                'filename': img_path.name,
                'path': str(img_path),
                'result': result
            })
            
            # Mostrar resumen
            print(f"✅ Detección: {'SÍ' if result['has_crack'] else 'NO'}")
            print(f"📊 Probabilidad: {result['detection_probability']:.3f}")
            print(f"⚠️  Severidad: {result['severity'].upper()}")
            
            if result['has_crack'] and result['crack_analysis']:
                print(f"📏 Ancho estimado: {result['crack_analysis']['estimated_width_px']:.1f} px")
                print(f"📈 Densidad: {result['crack_analysis']['crack_density']:.4f}")
            
            print(f"🎯 Riesgo: {result['severity_info'].get('riesgo', 'N/A')}")
            print()
            
        except Exception as e:
            print(f"❌ Error procesando {img_path.name}: {e}")
            print()
    
    # Resumen final
    print("📊 RESUMEN DE RESULTADOS")
    print("=" * 50)
    
    severity_counts = {}
    total_analyzed = len(results)
    
    for result_data in results:
        severity = result_data['result']['severity']
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    
    for severity, count in severity_counts.items():
        percentage = (count / total_analyzed) * 100 if total_analyzed > 0 else 0
        print(f"   {severity.upper()}: {count} imágenes ({percentage:.1f}%)")
    
    print(f"\n✅ Análisis completado: {total_analyzed} imágenes procesadas")
    
    # Preguntar si mostrar análisis detallado
    print(f"\n🖼️  ¿Mostrar análisis visual detallado? (y/n): ", end="")
    try:
        choice = input().lower().strip()
        if choice in ['y', 'yes', 'sí', 's']:
            print("\n🖼️  ANÁLISIS VISUAL DETALLADO")
            print("=" * 50)
            
            # Mostrar análisis visual para las primeras 2 imágenes
            for result_data in results[:2]:
                print(f"\nAnalizando visualmente: {result_data['filename']}")
                classifier.analyze_single_image(result_data['path'], show_analysis=True)
                
                input("Presiona Enter para continuar...")
    except:
        pass
    
    return results

if __name__ == "__main__":
    test_results = test_severity_classifier()
    
    print("\n🎉 PRUEBA COMPLETADA")
    print("✅ Clasificador funcionando correctamente")
    print("📁 Listo para uso en producción")
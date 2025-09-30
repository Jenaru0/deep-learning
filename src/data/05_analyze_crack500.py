"""
Análisis de Estructura CRACK500
===============================

Script para analizar la estructura del dataset CRACK500 y determinar
cómo integrarlo con SDNET2018 de manera óptima.
"""

import json
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

class CRACK500Analyzer:
    """Analizar estructura y contenido de CRACK500"""
    
    def __init__(self):
        self.crack500_path = Path("data/external/CRACK500")
        self.sdnet_path = Path("data/processed/sdnet2018_prepared")
        
    def analyze_crack500_structure(self):
        """Analizar estructura completa de CRACK500"""
        print("🔍 ANÁLISIS DE ESTRUCTURA CRACK500")
        print("=" * 50)
        
        if not self.crack500_path.exists():
            print("❌ CRACK500 no encontrado")
            return
        
        # Analizar directorios
        subdirs = []
        total_files = 0
        
        for item in self.crack500_path.rglob("*"):
            if item.is_file():
                total_files += 1
                subdir = str(item.relative_to(self.crack500_path).parent)
                if subdir not in [d['path'] for d in subdirs]:
                    subdirs.append({
                        'path': subdir,
                        'files': [],
                        'count': 0
                    })
        
        # Contar archivos por directorio
        for subdir in subdirs:
            dir_path = self.crack500_path / subdir['path']
            files = list(dir_path.glob("*"))
            subdir['files'] = [f.name for f in files[:5]]  # Solo primeros 5
            subdir['count'] = len(list(dir_path.glob("*")))
        
        print(f"📊 Total archivos: {total_files}")
        print(f"📁 Directorios encontrados: {len(subdirs)}")
        print()
        
        for subdir in sorted(subdirs, key=lambda x: x['count'], reverse=True):
            print(f"📂 {subdir['path']}: {subdir['count']} archivos")
            if subdir['files']:
                print(f"   Ejemplos: {', '.join(subdir['files'][:3])}")
            print()
        
        return subdirs
    
    def analyze_image_characteristics(self):
        """Analizar características de las imágenes"""
        print("🖼️ ANÁLISIS DE CARACTERÍSTICAS DE IMÁGENES")
        print("=" * 50)
        
        # Buscar imágenes en diferentes carpetas
        image_dirs = [
            "images",
            "traindata", 
            "testdata",
            "valdata"
        ]
        
        all_stats = []
        
        for img_dir in image_dirs:
            dir_path = self.crack500_path / img_dir
            if not dir_path.exists():
                continue
                
            print(f"📁 Analizando: {img_dir}")
            
            image_files = list(dir_path.glob("*.jpg")) + list(dir_path.glob("*.png"))
            if not image_files:
                print(f"   ❌ No se encontraron imágenes")
                continue
            
            # Analizar muestra de imágenes
            sample_size = min(20, len(image_files))
            sample_files = image_files[:sample_size]
            
            sizes = []
            for img_path in sample_files:
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        h, w = img.shape[:2]
                        sizes.append((w, h))
                except:
                    continue
            
            if sizes:
                widths = [s[0] for s in sizes]
                heights = [s[1] for s in sizes]
                
                stats = {
                    'directory': img_dir,
                    'total_images': len(image_files),
                    'analyzed': len(sizes),
                    'avg_width': np.mean(widths),
                    'avg_height': np.mean(heights),
                    'min_width': min(widths),
                    'max_width': max(widths),
                    'min_height': min(heights),
                    'max_height': max(heights)
                }
                
                all_stats.append(stats)
                
                print(f"   📊 Imágenes totales: {stats['total_images']}")
                print(f"   📏 Tamaño promedio: {stats['avg_width']:.0f}x{stats['avg_height']:.0f}")
                print(f"   📐 Rango width: {stats['min_width']}-{stats['max_width']}")
                print(f"   📐 Rango height: {stats['min_height']}-{stats['max_height']}")
                print()
        
        return all_stats
    
    def compare_with_sdnet2018(self):
        """Comparar características con SDNET2018"""
        print("⚖️ COMPARACIÓN CON SDNET2018")
        print("=" * 50)
        
        # Analizar SDNET2018
        sdnet_train_crack = self.sdnet_path / "train" / "crack"
        sdnet_train_no_crack = self.sdnet_path / "train" / "no_crack"
        
        if not sdnet_train_crack.exists():
            print("❌ SDNET2018 no encontrado")
            return
        
        # Contar imágenes SDNET2018
        sdnet_crack_count = len(list(sdnet_train_crack.glob("*.jpg")))
        sdnet_no_crack_count = len(list(sdnet_train_no_crack.glob("*.jpg")))
        sdnet_total = sdnet_crack_count + sdnet_no_crack_count
        
        print(f"📊 SDNET2018 (actual):")
        print(f"   🔴 Con fisuras: {sdnet_crack_count}")
        print(f"   🟢 Sin fisuras: {sdnet_no_crack_count}")
        print(f"   📈 Total: {sdnet_total}")
        print()
        
        # Analizar muestra de SDNET2018
        sample_sdnet = list(sdnet_train_crack.glob("*.jpg"))[:10]
        sdnet_sizes = []
        
        for img_path in sample_sdnet:
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    h, w = img.shape[:2]
                    sdnet_sizes.append((w, h))
            except:
                continue
        
        if sdnet_sizes:
            sdnet_widths = [s[0] for s in sdnet_sizes]
            sdnet_heights = [s[1] for s in sdnet_sizes]
            
            print(f"📏 SDNET2018 tamaños:")
            print(f"   Promedio: {np.mean(sdnet_widths):.0f}x{np.mean(sdnet_heights):.0f}")
            print(f"   Rango: {min(sdnet_widths)}-{max(sdnet_widths)} x {min(sdnet_heights)}-{max(sdnet_heights)}")
        
        print()
        return {
            'sdnet_crack': sdnet_crack_count,
            'sdnet_no_crack': sdnet_no_crack_count,
            'sdnet_total': sdnet_total
        }
    
    def create_integration_plan(self):
        """Crear plan de integración"""
        print("📋 PLAN DE INTEGRACIÓN")
        print("=" * 50)
        
        plan = {
            'strategy': 'unified_dataset',
            'steps': [
                '1. Identificar imágenes con fisuras en CRACK500',
                '2. Identificar imágenes sin fisuras (o crear artificialmente)',
                '3. Redimensionar todas las imágenes a 128x128',
                '4. Crear nuevo dataset unificado balanceado',
                '5. Re-entrenar modelo con dataset expandido'
            ],
            'expected_benefits': [
                'Mayor cantidad de datos de entrenamiento',
                'Mejor generalización del modelo',
                'Posible aumento en precisión',
                'Mayor robustez ante diferentes tipos de fisuras'
            ],
            'challenges': [
                'CRACK500 puede tener solo imágenes CON fisuras',
                'Necesidad de balancear el dataset',
                'Posible sobre-ajuste si no se maneja bien'
            ]
        }
        
        for step in plan['steps']:
            print(f"   {step}")
        
        print(f"\n✅ Beneficios esperados:")
        for benefit in plan['expected_benefits']:
            print(f"   • {benefit}")
        
        print(f"\n⚠️ Desafíos a considerar:")
        for challenge in plan['challenges']:
            print(f"   • {challenge}")
        
        return plan
    
    def generate_summary_report(self):
        """Generar reporte resumen"""
        print("\n" + "="*60)
        print("📄 REPORTE RESUMEN - INTEGRACIÓN CRACK500")
        print("="*60)
        
        # Ejecutar todos los análisis
        structure = self.analyze_crack500_structure()
        image_stats = self.analyze_image_characteristics()
        comparison = self.compare_with_sdnet2018()
        plan = self.create_integration_plan()
        
        # Guardar reporte
        report = {
            'analysis_date': '2025-09-26',
            'crack500_structure': structure,
            'image_characteristics': image_stats,
            'sdnet_comparison': comparison,
            'integration_plan': plan
        }
        
        report_path = Path("results/crack500_analysis_report.json")
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Reporte guardado: {report_path}")
        print("✅ Análisis completado!")
        
        return report

if __name__ == "__main__":
    analyzer = CRACK500Analyzer()
    analyzer.generate_summary_report()
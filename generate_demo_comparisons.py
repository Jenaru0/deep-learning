"""
Generador de Comparaciones Visuales para Presentación
=====================================================

Script para generar imágenes comparativas mostrando:
1. Imagen original
2. Detección del modelo CNN
3. Análisis de severidad
4. Resultados antes/después de mejoras
"""

import sys
sys.path.append('.')

import importlib.util
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2

# Importar el clasificador
spec = importlib.util.spec_from_file_location("severity_classifier", "src/models/03_severity_classifier.py")
severity_classifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(severity_classifier)
CrackSeverityClassifier = severity_classifier.CrackSeverityClassifier

class DemoGenerator:
    """Generar demos visuales para presentación"""
    
    def __init__(self):
        self.classifier = CrackSeverityClassifier()
        self.results_dir = Path("results/demo_comparisons")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_single_analysis_demo(self, image_path, save_name):
        """Generar demo de análisis de una imagen"""
        print(f"📸 Generando demo: {save_name}")
        
        # Cargar imagen original
        image_bgr = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Análisis completo
        result = self.classifier.analyze_single_image(str(image_path), show_analysis=False)
        
        # Crear figura con subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'ANÁLISIS DE FISURAS - {Path(image_path).name}', fontsize=16, fontweight='bold')
        
        # 1. Imagen original
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title('1. Imagen Original', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # 2. Preprocesado para el modelo
        image_resized, image_normalized = self.classifier.preprocess_image(image_path)
        axes[0, 1].imshow(image_resized)
        axes[0, 1].set_title('2. Preprocesado (128x128)', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        # 3. Detección morfológica
        contours, processed = self.classifier.detect_crack_contours(image_resized)
        contour_img = image_resized.copy()
        if contours:
            cv2.drawContours(contour_img, contours, -1, (255, 0, 0), 2)
        axes[1, 0].imshow(contour_img)
        axes[1, 0].set_title('3. Análisis Morfológico', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        # 4. Resultado final
        result_img = image_resized.copy()
        severity = result['severity']
        has_crack = result['has_crack']
        
        # Color según severidad (RGB 0-255 para OpenCV)
        color_map_cv = {
            'sin_fisura': (0, 255, 0),      # Verde
            'superficial': (255, 255, 0),   # Amarillo
            'moderada': (255, 165, 0),      # Naranja
            'estructural': (255, 0, 0)      # Rojo
        }
        
        # Color para matplotlib (RGB 0-1)
        color_map_plt = {
            'sin_fisura': 'green',
            'superficial': 'gold',
            'moderada': 'orange',
            'estructural': 'red'
        }
        
        color_cv = color_map_cv.get(severity, (128, 128, 128))
        color_plt = color_map_plt.get(severity, 'black')
        
        # Dibujar bounding box si hay fisura
        if has_crack and result['crack_analysis']:
            x, y, w, h = result['crack_analysis']['bounding_box']
            cv2.rectangle(result_img, (x, y), (x+w, y+h), color_cv, 2)
        
        axes[1, 1].imshow(result_img)
        
        # Título con resultados
        title = f"4. RESULTADO: {severity.upper()}"
        axes[1, 1].set_title(title, fontsize=12, fontweight='bold', color=color_plt)
        axes[1, 1].axis('off')
        
        # Agregar información detallada
        info_text = f"""DETALLES DEL ANÁLISIS:
        
📊 Probabilidad CNN: {result['detection_probability']:.3f}
✅ Fisura detectada: {'SÍ' if has_crack else 'NO'}
⚠️  Severidad: {severity.upper()}
🔧 Método: {result.get('method', 'N/A')}
        
"""
        
        if has_crack and result['crack_analysis']:
            info_text += f"""📏 Densidad: {result['crack_analysis']['crack_density']:.4f}
📈 Contornos: {result['crack_analysis']['total_contours']}
🎯 Confianza: {result['severity_info'].get('confidence', 'N/A')}"""
        
        # Agregar texto informativo
        fig.text(0.02, 0.02, info_text, fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        
        # Guardar
        save_path = self.results_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Demo guardado: {save_path}")
        return str(save_path)
    
    def generate_comparison_grid(self, max_images=6):
        """Generar grid de comparación con múltiples imágenes"""
        print("🎯 GENERANDO GRID DE COMPARACIÓN")
        print("=" * 50)
        
        # Obtener imágenes de ejemplo
        crack_dir = Path("data/processed/sdnet2018_prepared/test/crack")
        no_crack_dir = Path("data/processed/sdnet2018_prepared/test/no_crack")
        
        crack_images = list(crack_dir.glob("*.jpg"))[:max_images//2]
        no_crack_images = list(no_crack_dir.glob("*.jpg"))[:max_images//2]
        
        all_images = crack_images + no_crack_images
        
        if not all_images:
            print("❌ No se encontraron imágenes")
            return
        
        # Crear grid
        rows = 2
        cols = len(all_images)
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 8))
        fig.suptitle('COMPARACIÓN DE RESULTADOS - DETECCIÓN DE FISURAS', 
                    fontsize=16, fontweight='bold')
        
        if cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, img_path in enumerate(all_images):
            # Cargar y analizar imagen
            image_bgr = cv2.imread(str(img_path))
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image_rgb, (128, 128))
            
            result = self.classifier.analyze_single_image(str(img_path), show_analysis=False)
            
            # Imagen original (fila superior)
            axes[0, i].imshow(image_resized)
            expected = "CRACK" if "crack" in str(img_path) else "NO CRACK"
            axes[0, i].set_title(f'Original\n{img_path.name}\nEsperado: {expected}', 
                               fontsize=10)
            axes[0, i].axis('off')
            
            # Resultado del análisis (fila inferior)
            result_img = image_resized.copy()
            severity = result['severity']
            has_crack = result['has_crack']
            
            # Color del marco según resultado
            color_map = {
                'sin_fisura': (0, 255, 0),
                'superficial': (255, 255, 0),
                'moderada': (255, 165, 0),
                'estructural': (255, 0, 0)
            }
            
            color = np.array(color_map.get(severity, (128, 128, 128))) / 255.0
            
            # Dibujar marco de color
            for thickness in range(5):
                cv2.rectangle(result_img, (thickness, thickness), 
                            (128-thickness, 128-thickness), color*255, 1)
            
            axes[1, i].imshow(result_img)
            
            result_text = f"Resultado: {'SÍ' if has_crack else 'NO'}\nSeveridad: {severity.upper()}\nProb: {result['detection_probability']:.2f}"
            axes[1, i].set_title(result_text, fontsize=9)
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        # Guardar
        save_path = self.results_dir / "comparison_grid.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Grid guardado: {save_path}")
        return str(save_path)
    
    def generate_methodology_diagram(self):
        """Generar diagrama de metodología"""
        print("📊 GENERANDO DIAGRAMA DE METODOLOGÍA")
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        
        # Título
        ax.text(5, 7.5, 'METODOLOGÍA DE DETECCIÓN Y CLASIFICACIÓN DE FISURAS', 
               ha='center', fontsize=16, fontweight='bold')
        
        # Paso 1
        ax.add_patch(plt.Rectangle((0.5, 6), 2, 1, facecolor='lightblue', edgecolor='black'))
        ax.text(1.5, 6.5, '1. CARGA\nIMAGEN', ha='center', va='center', fontweight='bold')
        
        # Flecha
        ax.arrow(2.7, 6.5, 0.6, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        # Paso 2
        ax.add_patch(plt.Rectangle((3.5, 6), 2, 1, facecolor='lightgreen', edgecolor='black'))
        ax.text(4.5, 6.5, '2. PREPROCESO\n128x128', ha='center', va='center', fontweight='bold')
        
        # Flecha
        ax.arrow(5.7, 6.5, 0.6, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        # Paso 3
        ax.add_patch(plt.Rectangle((6.5, 6), 2, 1, facecolor='lightyellow', edgecolor='black'))
        ax.text(7.5, 6.5, '3. CNN\nDETECCIÓN', ha='center', va='center', fontweight='bold')
        
        # Decisión
        ax.arrow(7.5, 5.8, 0, -0.6, head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        # Diamante de decisión
        diamond = plt.Polygon([(7.5, 4.8), (8.3, 4.3), (7.5, 3.8), (6.7, 4.3)], 
                            facecolor='orange', edgecolor='black')
        ax.add_patch(diamond)
        ax.text(7.5, 4.3, 'Prob > 0.7?', ha='center', va='center', fontweight='bold', fontsize=9)
        
        # NO - Sin fisura
        ax.arrow(6.5, 4.3, -1.3, 0, head_width=0.1, head_length=0.1, fc='red', ec='red')
        ax.add_patch(plt.Rectangle((3, 3.8), 2, 1, facecolor='lightcoral', edgecolor='black'))
        ax.text(4, 4.3, 'SIN FISURA', ha='center', va='center', fontweight='bold')
        ax.text(5.5, 4.5, 'NO', ha='center', va='center', color='red', fontweight='bold')
        
        # SÍ - Análisis morfológico
        ax.arrow(7.5, 3.6, 0, -0.8, head_width=0.1, head_length=0.1, fc='green', ec='green')
        ax.text(7.8, 3.2, 'SÍ', ha='center', va='center', color='green', fontweight='bold')
        
        # Paso 4
        ax.add_patch(plt.Rectangle((6.5, 2.2), 2, 1, facecolor='lightpink', edgecolor='black'))
        ax.text(7.5, 2.7, '4. ANÁLISIS\nMORFOLÓGICO', ha='center', va='center', fontweight='bold')
        
        # Flecha
        ax.arrow(7.5, 2, 0, -0.6, head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        # Clasificación final
        ax.add_patch(plt.Rectangle((6, 0.5), 3, 1, facecolor='gold', edgecolor='black'))
        ax.text(7.5, 1, '5. CLASIFICACIÓN\nSEVERIDAD', ha='center', va='center', fontweight='bold')
        
        # Resultados
        results = [
            ('SUPERFICIAL', 'lightgreen', 2),
            ('MODERADA', 'orange', 5),
            ('ESTRUCTURAL', 'red', 8)
        ]
        
        for i, (label, color, x) in enumerate(results):
            ax.add_patch(plt.Rectangle((x, 0.2), 1.5, 0.6, facecolor=color, edgecolor='black'))
            ax.text(x+0.75, 0.5, label, ha='center', va='center', fontweight='bold', fontsize=9)
        
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Guardar
        save_path = self.results_dir / "methodology_diagram.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Diagrama guardado: {save_path}")
        return str(save_path)

def main():
    """Generar todas las comparaciones"""
    print("🎬 GENERADOR DE DEMOS PARA PRESENTACIÓN")
    print("=" * 60)
    
    demo = DemoGenerator()
    
    # 1. Grid de comparación
    demo.generate_comparison_grid()
    
    # 2. Diagrama de metodología
    demo.generate_methodology_diagram()
    
    # 3. Análisis detallado de imágenes específicas
    crack_dir = Path("data/processed/sdnet2018_prepared/test/crack")
    no_crack_dir = Path("data/processed/sdnet2018_prepared/test/no_crack")
    
    crack_images = list(crack_dir.glob("*.jpg"))[:2]
    no_crack_images = list(no_crack_dir.glob("*.jpg"))[:1]
    
    for img_path in crack_images:
        demo.generate_single_analysis_demo(img_path, f"analisis_crack_{img_path.stem}")
    
    for img_path in no_crack_images:
        demo.generate_single_analysis_demo(img_path, f"analisis_no_crack_{img_path.stem}")
    
    print("\n🎉 TODOS LOS DEMOS GENERADOS")
    print(f"📁 Ubicación: {demo.results_dir}")
    print("✅ Listo para presentar al docente!")

if __name__ == "__main__":
    main()
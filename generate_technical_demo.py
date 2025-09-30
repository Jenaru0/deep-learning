"""
Demo Completo para Presentación
===============================

Genera análisis técnico detallado con métricas, orientaciones,
y validaciones para responder preguntas del docente.
"""

import sys
sys.path.append('.')

import importlib.util
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2
import json
from datetime import datetime

# Importar el clasificador
spec = importlib.util.spec_from_file_location("severity_classifier", "src/models/03_severity_classifier.py")
severity_classifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(severity_classifier)
CrackSeverityClassifier = severity_classifier.CrackSeverityClassifier

class TechnicalDemo:
    """Demo técnico completo para presentación"""
    
    def __init__(self):
        self.classifier = CrackSeverityClassifier()
        self.results_dir = Path("results/technical_demo")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_crack_orientation(self, contours, image_shape):
        """Análisis de orientación de fisuras"""
        if not contours:
            return None
            
        orientations = []
        for contour in contours:
            if len(contour) >= 5:
                # Ajustar elipse para obtener orientación
                ellipse = cv2.fitEllipse(contour)
                angle = ellipse[2]
                
                # Normalizar ángulo
                if angle > 90:
                    angle = angle - 180
                    
                orientations.append(angle)
        
        if orientations:
            avg_orientation = np.mean(orientations)
            std_orientation = np.std(orientations)
            
            # Clasificar tipo de fisura por orientación
            if abs(avg_orientation) < 15:
                crack_type = "Horizontal"
                risk_level = "Medio"
            elif abs(avg_orientation) > 75:
                crack_type = "Vertical" 
                risk_level = "Bajo"
            else:
                crack_type = "Diagonal"
                risk_level = "Alto"
                
            return {
                "average_angle": avg_orientation,
                "std_angle": std_orientation,
                "crack_type": crack_type,
                "risk_level": risk_level,
                "individual_angles": orientations
            }
        return None
    
    def calculate_technical_metrics(self, contours, image_shape):
        """Calcular métricas técnicas avanzadas"""
        if not contours:
            return None
            
        h, w = image_shape[:2]
        
        metrics = {
            "total_contours": len(contours),
            "image_dimensions": (w, h),
            "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Métricas geométricas
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Bounding box
            x, y, bbox_w, bbox_h = cv2.boundingRect(largest_contour)
            
            # Convex hull para solidez
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Aspect ratio
            aspect_ratio = bbox_w / bbox_h if bbox_h > 0 else 0
            
            # Extent (relación área/área_bounding_box)
            bbox_area = bbox_w * bbox_h
            extent = area / bbox_area if bbox_area > 0 else 0
            
            # Compacidad
            compactness = (perimeter ** 2) / area if area > 0 else 0
            
            metrics.update({
                "largest_crack": {
                    "area_px": area,
                    "perimeter_px": perimeter,
                    "solidity": solidity,
                    "aspect_ratio": aspect_ratio,
                    "extent": extent,
                    "compactness": compactness,
                    "bounding_box": {
                        "x": x, "y": y, "width": bbox_w, "height": bbox_h
                    }
                },
                "density_analysis": {
                    "crack_density": area / (w * h),
                    "coverage_percentage": (area / (w * h)) * 100
                }
            })
        
        return metrics
    
    def generate_technical_analysis(self, image_path, save_name):
        """Generar análisis técnico completo"""
        print(f"🔬 Análisis técnico: {save_name}")
        
        # Cargar imagen
        image_bgr = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Análisis con el clasificador
        result = self.classifier.analyze_single_image(str(image_path), show_analysis=False)
        
        # Análisis morfológico detallado
        image_resized, _ = self.classifier.preprocess_image(image_path)
        contours, processed = self.classifier.detect_crack_contours(image_resized)
        
        # Análisis de orientación
        orientation_analysis = self.analyze_crack_orientation(contours, image_resized.shape)
        
        # Métricas técnicas
        technical_metrics = self.calculate_technical_metrics(contours, image_resized.shape)
        
        # Crear visualización completa
        fig = plt.figure(figsize=(16, 12))
        
        # Layout: 3x3 grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Imagen original
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image_resized)
        ax1.set_title('1. Imagen Original\n128x128px', fontweight='bold')
        ax1.axis('off')
        
        # 2. Preprocesamiento
        gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(bilateral, cmap='gray')
        ax2.set_title('2. Filtro Bilateral\nPreservación de bordes', fontweight='bold')
        ax2.axis('off')
        
        # 3. Detección de bordes
        edges = cv2.Canny(bilateral, 50, 150)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(edges, cmap='gray')
        ax3.set_title('3. Detección Canny\nUmbrales: 50-150', fontweight='bold')
        ax3.axis('off')
        
        # 4. Contornos detectados
        contour_img = image_resized.copy()
        if contours:
            cv2.drawContours(contour_img, contours, -1, (255, 0, 0), 2)
            
            # Dibujar orientación principal
            if orientation_analysis:
                for i, contour in enumerate(contours[:3]):  # Solo primeros 3
                    if len(contour) >= 5:
                        ellipse = cv2.fitEllipse(contour)
                        cv2.ellipse(contour_img, ellipse, (0, 255, 255), 1)
        
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.imshow(contour_img)
        ax4.set_title(f'4. Contornos Detectados\nTotal: {len(contours)}', fontweight='bold')
        ax4.axis('off')
        
        # 5. Análisis de severidad
        severity_img = image_resized.copy()
        severity = result['severity']
        
        color_map = {
            'sin_fisura': (0, 255, 0),
            'superficial': (255, 255, 0),
            'moderada': (255, 165, 0),
            'estructural': (255, 0, 0)
        }
        
        if result['has_crack'] and result['crack_analysis']:
            x, y, w, h = result['crack_analysis']['bounding_box']
            color = color_map.get(severity, (128, 128, 128))
            cv2.rectangle(severity_img, (x, y), (x+w, y+h), color, 3)
        
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.imshow(severity_img)
        ax5.set_title(f'5. Clasificación\n{severity.upper()}', fontweight='bold', 
                     color='red' if severity == 'estructural' else 'orange' if severity == 'moderada' else 'green')
        ax5.axis('off')
        
        # 6. Orientación de fisuras
        ax6 = fig.add_subplot(gs[1, 2])
        if orientation_analysis and contours:
            angles = orientation_analysis['individual_angles']
            ax6.hist(angles, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            ax6.axvline(orientation_analysis['average_angle'], color='red', 
                       linestyle='--', label=f'Promedio: {orientation_analysis["average_angle"]:.1f}°')
            ax6.set_xlabel('Ángulo (grados)')
            ax6.set_ylabel('Frecuencia')
            ax6.set_title('6. Orientación de Fisuras')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'Sin fisuras\ndetectadas', ha='center', va='center', 
                    transform=ax6.transAxes, fontsize=12)
            ax6.set_title('6. Orientación de Fisuras')
        
        # 7. Métricas CNN
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.axis('off')
        
        cnn_text = f"""MÉTRICAS CNN:
        
Probabilidad: {result['detection_probability']:.3f}
Umbral: 0.700
Resultado: {'POSITIVO' if result['has_crack'] else 'NEGATIVO'}
Método: {result.get('method', 'N/A')}

Confianza: {result.get('severity_info', {}).get('confidence', 'N/A')}"""
        
        ax7.text(0.05, 0.95, cnn_text, transform=ax7.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 8. Métricas morfológicas
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.axis('off')
        
        if result['crack_analysis']:
            morph_text = f"""MÉTRICAS MORFOLÓGICAS:
            
Densidad: {result['crack_analysis']['crack_density']:.4f}
Contornos: {result['crack_analysis']['total_contours']}
Área mayor: {result['crack_analysis']['largest_contour_area']:.0f}px²
Ancho est.: {result['crack_analysis']['estimated_width_px']:.1f}px

Perímetro: {result['crack_analysis']['arc_length']:.1f}px"""
        else:
            morph_text = "MÉTRICAS MORFOLÓGICAS:\n\nNo hay fisuras detectadas"
            
        ax8.text(0.05, 0.95, morph_text, transform=ax8.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # 9. Análisis de orientación
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        if orientation_analysis:
            orient_text = f"""ANÁLISIS DE ORIENTACIÓN:
            
Tipo: {orientation_analysis['crack_type']}
Ángulo prom.: {orientation_analysis['average_angle']:.1f}°
Desv. est.: {orientation_analysis['std_angle']:.1f}°
Nivel riesgo: {orientation_analysis['risk_level']}

Total ángulos: {len(orientation_analysis['individual_angles'])}"""
        else:
            orient_text = "ANÁLISIS DE ORIENTACIÓN:\n\nNo hay orientación\ndeterminada"
            
        ax9.text(0.05, 0.95, orient_text, transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # Título general
        main_title = f'ANÁLISIS TÉCNICO COMPLETO - {Path(image_path).name}'
        fig.suptitle(main_title, fontsize=16, fontweight='bold', y=0.98)
        
        # Guardar
        save_path = self.results_dir / f"{save_name}_technical.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Guardar datos técnicos en JSON
        technical_data = {
            "image_info": {
                "filename": Path(image_path).name,
                "analysis_timestamp": datetime.now().isoformat()
            },
            "cnn_results": result,
            "orientation_analysis": orientation_analysis,
            "technical_metrics": technical_metrics
        }
        
        json_path = self.results_dir / f"{save_name}_data.json"
        with open(json_path, 'w') as f:
            json.dump(technical_data, f, indent=2, default=str)
        
        print(f"✅ Análisis guardado: {save_path}")
        print(f"📊 Datos técnicos: {json_path}")
        
        return str(save_path), str(json_path)

def main():
    """Generar demos técnicos para presentación"""
    print("🔬 GENERANDO ANÁLISIS TÉCNICO COMPLETO")
    print("=" * 60)
    
    demo = TechnicalDemo()
    
    # Buscar imágenes de ejemplo
    crack_dir = Path("data/processed/sdnet2018_prepared/test/crack")
    no_crack_dir = Path("data/processed/sdnet2018_prepared/test/no_crack")
    
    # Seleccionar imágenes representativas
    crack_images = list(crack_dir.glob("*.jpg"))[:3]  # 3 con fisuras
    no_crack_images = list(no_crack_dir.glob("*.jpg"))[:1]  # 1 sin fisuras
    
    all_images = crack_images + no_crack_images
    
    if not all_images:
        print("❌ No se encontraron imágenes para análisis")
        return
    
    print(f"📸 Analizando {len(all_images)} imágenes técnicamente...")
    
    results_summary = []
    
    for i, img_path in enumerate(all_images, 1):
        category = "CRACK" if "crack" in str(img_path) else "NO_CRACK"
        save_name = f"demo_{i}_{category}_{img_path.stem}"
        
        print(f"\n{'='*50}")
        print(f"🔍 ANÁLISIS {i}/{len(all_images)}: {img_path.name}")
        print(f"Categoría: {category}")
        print(f"{'='*50}")
        
        try:
            img_result, json_result = demo.generate_technical_analysis(img_path, save_name)
            results_summary.append({
                "image": img_path.name,
                "category": category,
                "analysis_image": img_result,
                "technical_data": json_result
            })
            
        except Exception as e:
            print(f"❌ Error procesando {img_path.name}: {e}")
    
    # Crear resumen final
    print(f"\n🎉 ANÁLISIS TÉCNICO COMPLETADO")
    print("=" * 60)
    print(f"📁 Ubicación: {demo.results_dir}")
    print(f"📊 Imágenes analizadas: {len(results_summary)}")
    
    for result in results_summary:
        print(f"✅ {result['image']} ({result['category']})")
    
    print(f"\n🎯 ARCHIVOS LISTOS PARA PRESENTACIÓN:")
    for file in demo.results_dir.glob("*"):
        print(f"   📄 {file.name}")
    
    print(f"\n🚀 ¡LISTO PARA RESPONDER PREGUNTAS TÉCNICAS!")

if __name__ == "__main__":
    main()
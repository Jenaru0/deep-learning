"""
Análisis Especializado de Fisuras Reales
========================================

Script optimizado para detectar y analizar fisuras reales con orientaciones,
métricas detalladas y clasificaciones precisas.
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

class RealCrackAnalyzer:
    """Analizador especializado para fisuras reales"""
    
    def __init__(self):
        self.classifier = CrackSeverityClassifier()
        self.results_dir = Path("results/real_crack_analysis")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def enhanced_crack_detection(self, image):
        """Detección mejorada de fisuras reales"""
        # Convertir a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Múltiples técnicas de preprocesamiento
        
        # 1. Filtro bilateral más agresivo
        bilateral = cv2.bilateralFilter(gray, 15, 80, 80)
        
        # 2. CLAHE más intensivo
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(bilateral)
        
        # 3. Filtro Gaussiano
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # 4. Múltiples umbrales de Canny para capturar más fisuras
        edges1 = cv2.Canny(blurred, 30, 90, apertureSize=3)
        edges2 = cv2.Canny(blurred, 50, 120, apertureSize=3)
        edges3 = cv2.Canny(blurred, 20, 80, apertureSize=3)
        
        # Combinar diferentes detecciones
        combined_edges = cv2.bitwise_or(edges1, cv2.bitwise_or(edges2, edges3))
        
        # 5. Operaciones morfológicas más agresivas
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel_close)
        
        # Dilatación para conectar fragmentos
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(closed, kernel_dilate, iterations=1)
        
        # 6. Encontrar contornos
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 7. Filtrado menos restrictivo para fisuras reales
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Filtros ajustados para fisuras reales
            if area < 15:  # Área mínima más pequeña
                continue
                
            # Relación perímetro/área para fisuras lineales
            if perimeter > 0:
                elongation = perimeter / np.sqrt(area) if area > 0 else 0
                if elongation < 3:  # Menos restrictivo
                    continue
                    
            filtered_contours.append(contour)
        
        return filtered_contours, dilated, {
            'edges1': edges1, 'edges2': edges2, 'edges3': edges3,
            'combined': combined_edges, 'enhanced': enhanced
        }
    
    def analyze_crack_orientation_detailed(self, contours, image_shape):
        """Análisis detallado de orientación de fisuras"""
        if not contours:
            return None
            
        orientations = []
        crack_lines = []
        
        for i, contour in enumerate(contours):
            if len(contour) >= 5:
                # Método 1: Elipse fitting
                ellipse = cv2.fitEllipse(contour)
                angle_ellipse = ellipse[2]
                
                # Método 2: Línea fitting
                [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
                angle_line = np.arctan2(vy, vx) * 180 / np.pi
                
                # Normalizar ángulos
                if angle_ellipse > 90:
                    angle_ellipse = angle_ellipse - 180
                if angle_line > 90:
                    angle_line = angle_line - 90
                elif angle_line < -90:
                    angle_line = angle_line + 90
                    
                # Usar promedio de ambos métodos
                avg_angle = (angle_ellipse + angle_line) / 2
                
                orientations.append(avg_angle)
                crack_lines.append({
                    'contour_id': i,
                    'ellipse_angle': angle_ellipse,
                    'line_angle': angle_line,
                    'average_angle': avg_angle,
                    'contour_area': cv2.contourArea(contour),
                    'contour_length': cv2.arcLength(contour, False)
                })
        
        if orientations:
            avg_orientation = np.mean(orientations)
            std_orientation = np.std(orientations)
            
            # Clasificación más detallada por orientación
            abs_angle = abs(avg_orientation)
            if abs_angle < 10:
                crack_type = "Horizontal Pura"
                structural_risk = "Alto - Posible asentamiento"
            elif abs_angle < 25:
                crack_type = "Casi Horizontal"
                structural_risk = "Alto-Medio - Revisar carga"
            elif abs_angle > 80:
                crack_type = "Vertical Pura"
                structural_risk = "Medio - Contracción térmica"
            elif abs_angle > 65:
                crack_type = "Casi Vertical"
                structural_risk = "Medio-Bajo - Secado"
            elif 40 <= abs_angle <= 65:
                crack_type = "Diagonal Pronunciada"
                structural_risk = "Muy Alto - Falla estructural"
            elif 25 <= abs_angle < 40:
                crack_type = "Diagonal Moderada"
                structural_risk = "Alto - Esfuerzo cortante"
            else:
                crack_type = "Inclinada"
                structural_risk = "Medio-Alto"
                
            return {
                "average_angle": avg_orientation,
                "std_angle": std_orientation,
                "crack_type": crack_type,
                "structural_risk": structural_risk,
                "individual_cracks": crack_lines,
                "total_cracks_detected": len(orientations),
                "angle_distribution": {
                    "min": min(orientations),
                    "max": max(orientations),
                    "median": np.median(orientations)
                }
            }
        return None
    
    def calculate_advanced_metrics(self, contours, image_shape, orientation_data):
        """Calcular métricas avanzadas de fisuras reales"""
        if not contours:
            return None
            
        h, w = image_shape[:2]
        total_area = h * w
        
        # Métricas por contorno
        contour_metrics = []
        total_crack_area = 0
        total_crack_length = 0
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            total_crack_area += area
            total_crack_length += perimeter
            
            # Bounding box
            x, y, bbox_w, bbox_h = cv2.boundingRect(contour)
            
            # Métricas geométricas
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                (center_x, center_y), (width, height), angle = ellipse
                
                # Aspect ratio de la elipse
                ellipse_aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0
                
                contour_metrics.append({
                    'id': i,
                    'area': area,
                    'perimeter': perimeter,
                    'bbox': (x, y, bbox_w, bbox_h),
                    'ellipse_center': (center_x, center_y),
                    'ellipse_dimensions': (width, height),
                    'ellipse_aspect_ratio': ellipse_aspect_ratio,
                    'estimated_width': min(width, height),
                    'estimated_length': max(width, height)
                })
        
        # Métricas globales
        crack_density = total_crack_area / total_area
        coverage_percentage = crack_density * 100
        
        # Severidad basada en múltiples factores
        severity_score = 0
        severity_factors = []
        
        # Factor 1: Densidad
        if crack_density > 0.01:
            severity_score += 3
            severity_factors.append("Alta densidad")
        elif crack_density > 0.005:
            severity_score += 2
            severity_factors.append("Densidad moderada")
        elif crack_density > 0.002:
            severity_score += 1
            severity_factors.append("Densidad baja")
        
        # Factor 2: Número de fisuras
        if len(contours) > 10:
            severity_score += 2
            severity_factors.append("Múltiples fisuras")
        elif len(contours) > 5:
            severity_score += 1
            severity_factors.append("Varias fisuras")
        
        # Factor 3: Orientación (si disponible)
        if orientation_data:
            if "Diagonal" in orientation_data['crack_type']:
                severity_score += 2
                severity_factors.append("Orientación crítica")
            elif "Horizontal" in orientation_data['crack_type']:
                severity_score += 1
                severity_factors.append("Orientación preocupante")
        
        # Clasificación final
        if severity_score >= 5:
            final_severity = "CRÍTICA"
            action_required = "Evaluación estructural inmediata"
        elif severity_score >= 3:
            final_severity = "ALTA"
            action_required = "Monitoreo intensivo y reparación"
        elif severity_score >= 1:
            final_severity = "MODERADA"
            action_required = "Monitoreo regular"
        else:
            final_severity = "BAJA"
            action_required = "Inspección periódica"
        
        return {
            'total_contours': len(contours),
            'total_crack_area': total_crack_area,
            'total_crack_length': total_crack_length,
            'crack_density': crack_density,
            'coverage_percentage': coverage_percentage,
            'contour_details': contour_metrics,
            'severity_analysis': {
                'score': severity_score,
                'factors': severity_factors,
                'classification': final_severity,
                'action_required': action_required
            },
            'image_dimensions': (w, h),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def generate_comprehensive_analysis(self, image_path, save_name):
        """Generar análisis completo de fisura real"""
        print(f"🔬 Análisis completo: {save_name}")
        
        # Cargar imagen
        image_bgr = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Redimensionar manteniendo aspecto
        image_resized = cv2.resize(image_rgb, (128, 128))
        
        # Análisis mejorado de fisuras
        contours, processed, debug_images = self.enhanced_crack_detection(image_resized)
        
        # Análisis de orientación detallado
        orientation_analysis = self.analyze_crack_orientation_detailed(contours, image_resized.shape)
        
        # Métricas avanzadas
        advanced_metrics = self.calculate_advanced_metrics(contours, image_resized.shape, orientation_analysis)
        
        # Análisis con clasificador original para comparación
        original_result = self.classifier.analyze_single_image(str(image_path), show_analysis=False)
        
        # Crear visualización completa
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(4, 5, hspace=0.4, wspace=0.3)
        
        # Fila 1: Preprocesamiento
        # 1. Imagen original
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image_resized)
        ax1.set_title('1. Imagen Original', fontweight='bold', fontsize=10)
        ax1.axis('off')
        
        # 2. Filtro bilateral
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(debug_images['enhanced'], cmap='gray')
        ax2.set_title('2. CLAHE Mejorado', fontweight='bold', fontsize=10)
        ax2.axis('off')
        
        # 3. Edges combinados
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(debug_images['combined'], cmap='gray')
        ax3.set_title('3. Canny Combinado', fontweight='bold', fontsize=10)
        ax3.axis('off')
        
        # 4. Resultado morfológico
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.imshow(processed, cmap='gray')
        ax4.set_title('4. Morfología Final', fontweight='bold', fontsize=10)
        ax4.axis('off')
        
        # 5. Contornos detectados
        contour_img = image_resized.copy()
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        for i, contour in enumerate(contours[:5]):
            color = colors[i % len(colors)]
            cv2.drawContours(contour_img, [contour], -1, color, 2)
            
        ax5 = fig.add_subplot(gs[0, 4])
        ax5.imshow(contour_img)
        ax5.set_title(f'5. Fisuras ({len(contours)} detectadas)', fontweight='bold', fontsize=10)
        ax5.axis('off')
        
        # Fila 2: Análisis de orientación
        if orientation_analysis and orientation_analysis['individual_cracks']:
            # Histograma de orientaciones
            ax6 = fig.add_subplot(gs[1, 0:2])
            angles = [crack['average_angle'] for crack in orientation_analysis['individual_cracks']]
            ax6.hist(angles, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            ax6.axvline(orientation_analysis['average_angle'], color='red', 
                       linestyle='--', linewidth=2, label=f'Promedio: {orientation_analysis["average_angle"]:.1f}°')
            ax6.set_xlabel('Ángulo de Orientación (grados)')
            ax6.set_ylabel('Número de Fisuras')
            ax6.set_title('Distribución de Orientaciones', fontweight='bold')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            
            # Rosa de orientaciones (diagrama polar)
            ax7 = fig.add_subplot(gs[1, 2], projection='polar')
            angles_rad = [np.radians(angle + 90) for angle in angles]  # +90 para orientación correcta
            ax7.scatter(angles_rad, [1]*len(angles_rad), c='red', s=50, alpha=0.7)
            ax7.set_title('Rosa de Orientaciones', fontweight='bold', pad=20)
            ax7.set_ylim(0, 1.2)
            
            # Imagen con orientaciones marcadas
            orientation_img = image_resized.copy()
            for i, contour in enumerate(contours[:5]):
                if i < len(orientation_analysis['individual_cracks']):
                    crack_data = orientation_analysis['individual_cracks'][i]
                    
                    # Dibujar contorno
                    cv2.drawContours(orientation_img, [contour], -1, (255, 0, 0), 2)
                    
                    # Dibujar línea de orientación
                    if len(contour) >= 5:
                        ellipse = cv2.fitEllipse(contour)
                        center = (int(ellipse[0][0]), int(ellipse[0][1]))
                        angle = np.radians(crack_data['average_angle'])
                        length = 20
                        end_point = (int(center[0] + length * np.cos(angle)),
                                   int(center[1] + length * np.sin(angle)))
                        cv2.line(orientation_img, center, end_point, (0, 255, 255), 2)
                        cv2.putText(orientation_img, f'{crack_data["average_angle"]:.0f}°', 
                                  (center[0]+5, center[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
            
            ax8 = fig.add_subplot(gs[1, 3:5])
            ax8.imshow(orientation_img)
            ax8.set_title('Orientaciones Marcadas', fontweight='bold', fontsize=10)
            ax8.axis('off')
        
        # Fila 3: Métricas detalladas
        ax9 = fig.add_subplot(gs[2, 0:2])
        ax9.axis('off')
        
        if advanced_metrics:
            metrics_text = f"""MÉTRICAS AVANZADAS:

Fisuras detectadas: {advanced_metrics['total_contours']}
Área total: {advanced_metrics['total_crack_area']:.1f} px²
Longitud total: {advanced_metrics['total_crack_length']:.1f} px
Densidad: {advanced_metrics['crack_density']:.4f}
Cobertura: {advanced_metrics['coverage_percentage']:.2f}%

SEVERIDAD: {advanced_metrics['severity_analysis']['classification']}
Puntuación: {advanced_metrics['severity_analysis']['score']}/8
Factores: {', '.join(advanced_metrics['severity_analysis']['factors'][:2])}"""
        else:
            metrics_text = "No se detectaron métricas avanzadas"
            
        ax9.text(0.05, 0.95, metrics_text, transform=ax9.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Comparación con clasificador original
        ax10 = fig.add_subplot(gs[2, 2:4])
        ax10.axis('off')
        
        comparison_text = f"""COMPARACIÓN MÉTODOS:

CLASIFICADOR ORIGINAL:
• Detección: {'SÍ' if original_result['has_crack'] else 'NO'}
• Probabilidad: {original_result['detection_probability']:.3f}
• Severidad: {original_result['severity'].upper()}
• Método: {original_result.get('method', 'N/A')}

ANÁLISIS MEJORADO:
• Fisuras: {len(contours)} detectadas
• Orientación: {orientation_analysis['crack_type'] if orientation_analysis else 'N/A'}
• Riesgo: {orientation_analysis['structural_risk'][:20] + '...' if orientation_analysis else 'N/A'}"""
        
        ax10.text(0.05, 0.95, comparison_text, transform=ax10.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # Recomendaciones
        ax11 = fig.add_subplot(gs[2, 4])
        ax11.axis('off')
        
        if advanced_metrics:
            recommendations = f"""RECOMENDACIONES:

{advanced_metrics['severity_analysis']['action_required']}

Factores críticos:
{chr(10).join(['• ' + factor for factor in advanced_metrics['severity_analysis']['factors']])}"""
        else:
            recommendations = "No se requieren recomendaciones especiales"
            
        ax11.text(0.05, 0.95, recommendations, transform=ax11.transAxes, fontsize=9,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # Fila 4: Detalles por fisura
        if orientation_analysis and len(orientation_analysis['individual_cracks']) > 0:
            # Tabla de fisuras individuales
            ax12 = fig.add_subplot(gs[3, :])
            ax12.axis('off')
            
            # Crear tabla
            crack_data = []
            headers = ['ID', 'Área (px²)', 'Longitud (px)', 'Ángulo (°)', 'Tipo', 'Ancho Est. (px)']
            
            for i, crack in enumerate(orientation_analysis['individual_cracks'][:8]):  # Máximo 8
                crack_data.append([
                    f"F{i+1}",
                    f"{crack['contour_area']:.1f}",
                    f"{crack['contour_length']:.1f}",
                    f"{crack['average_angle']:.1f}",
                    orientation_analysis['crack_type'][:10],
                    f"{advanced_metrics['contour_details'][i]['estimated_width']:.1f}" if i < len(advanced_metrics['contour_details']) else "N/A"
                ])
            
            table = ax12.table(cellText=crack_data, colLabels=headers, 
                              cellLoc='center', loc='center',
                              colWidths=[0.08, 0.15, 0.15, 0.12, 0.25, 0.15])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
            
            # Colorear header
            for i in range(len(headers)):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Título principal
        title = f'ANÁLISIS COMPLETO DE FISURAS REALES - {Path(image_path).name}'
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        # Guardar imagen
        save_path = self.results_dir / f"{save_name}_complete.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Guardar datos JSON
        complete_data = {
            "image_info": {"filename": Path(image_path).name},
            "original_classifier": original_result,
            "enhanced_detection": {
                "contours_found": len(contours),
                "detection_method": "Multi-Canny + Enhanced Morphology"
            },
            "orientation_analysis": orientation_analysis,
            "advanced_metrics": advanced_metrics
        }
        
        json_path = self.results_dir / f"{save_name}_complete.json"
        with open(json_path, 'w') as f:
            json.dump(complete_data, f, indent=2, default=str)
        
        print(f"✅ Análisis completo guardado: {save_path}")
        return str(save_path)

def main():
    """Analizar 5 imágenes con fisuras reales"""
    print("🔬 ANÁLISIS ESPECIALIZADO DE FISURAS REALES")
    print("=" * 60)
    
    analyzer = RealCrackAnalyzer()
    
    # Buscar imágenes con fisuras
    crack_dir = Path("data/processed/sdnet2018_prepared/test/crack")
    crack_images = list(crack_dir.glob("*.jpg"))
    
    # Seleccionar 5 imágenes específicas que probablemente tengan fisuras visibles
    selected_images = crack_images[:5]
    
    print(f"📸 Analizando {len(selected_images)} imágenes con fisuras reales...")
    
    for i, img_path in enumerate(selected_images, 1):
        print(f"\n{'='*50}")
        print(f"🔍 ANÁLISIS {i}/5: {img_path.name}")
        print(f"{'='*50}")
        
        save_name = f"real_crack_{i:02d}_{img_path.stem}"
        
        try:
            result_path = analyzer.generate_comprehensive_analysis(img_path, save_name)
            print(f"✅ Completado: {Path(result_path).name}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n🎉 ANÁLISIS COMPLETADO")
    print(f"📁 Ubicación: {analyzer.results_dir}")
    print(f"🎯 ¡Listo para mostrar fisuras reales con orientaciones!")

if __name__ == "__main__":
    main()
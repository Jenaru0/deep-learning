"""
Clasificador de Severidad de Fisuras
====================================

Sistema de clasificación de severidad de fisuras basado en criterios técnicos
de literatura especializada en ingeniería estructural y patología de edificaciones.

Referencias:
- Mohan & Poobal (2018): Computer vision based crack detection methodologies
- Zhang et al. (2018): Automated crack detection and classification methods
- Cha et al. (2017): Deep learning‐based crack damage detection
"""

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CrackSeverityClassifier:
    """
    Clasificador de severidad de fisuras basado en análisis morfológico
    y criterios técnicos de literatura especializada.
    """
    
    def __init__(self, model_path="models/simple_crack_detector.keras"):
        """Inicializar clasificador"""
        self.model_path = Path(model_path)
        self.detection_model = None
        self.load_detection_model()
        
        # Criterios de severidad basados en literatura técnica
        self.severity_criteria = {
            "sin_fisura": {
                "descripcion": "No se detecta fisura",
                "ancho_min": 0.0,
                "ancho_max": 0.0,
                "densidad_max": 0.0,
                "riesgo": "Ninguno",
                "accion": "Ninguna acción requerida",
                "color": "green"
            },
            "superficial": {
                "descripcion": "Fisura superficial - No compromete integridad",
                "ancho_min": 0.0,
                "ancho_max": 0.3,  # mm (equivalente aprox. en píxeles)
                "densidad_max": 0.1,
                "riesgo": "Bajo - Estético",
                "accion": "Monitoreo periódico, sellado preventivo",
                "color": "green"
            },
            "moderada": {
                "descripcion": "Fisura moderada - Requiere atención",
                "ancho_min": 0.3,
                "ancho_max": 3.0,  # mm
                "densidad_max": 0.3,
                "riesgo": "Medio - Funcional",
                "accion": "Reparación programada, análisis de causas",
                "color": "orange"
            },
            "estructural": {
                "descripcion": "Fisura estructural - Compromete integridad",
                "ancho_min": 3.0,
                "ancho_max": float('inf'),
                "densidad_max": float('inf'),
                "riesgo": "Alto - Estructural",
                "accion": "Intervención inmediata, evaluación estructural",
                "color": "red"
            }
        }
        
        # Crear directorio de resultados
        self.results_dir = Path("results/severity_analysis")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def load_detection_model(self):
        """Cargar modelo de detección de fisuras"""
        try:
            if self.model_path.exists():
                print(f"Cargando modelo: {self.model_path}")
                self.detection_model = tf.keras.models.load_model(str(self.model_path))
                print("✅ Modelo cargado exitosamente")
            else:
                print(f"❌ Error: No se encuentra el modelo en {self.model_path}")
                print("Ejecuta primero el entrenamiento del modelo")
                raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}")
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            raise
    
    def preprocess_image(self, image_path):
        """Preprocesar imagen para análisis"""
        # Cargar imagen
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        
        # Convertir BGR a RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Redimensionar para el modelo (128x128 como se entrenó)
        image_resized = cv2.resize(image_rgb, (128, 128), interpolation=cv2.INTER_AREA)
        
        # Normalizar para el modelo (0-1)
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        return image_resized, image_normalized
    
    def detect_crack_contours(self, image):
        """Detectar contornos de fisuras usando análisis morfológico mejorado"""
        # Convertir a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Aplicar filtro bilateral para preservar bordes importantes
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Mejorar contraste con CLAHE más conservador
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        enhanced = clahe.apply(bilateral)
        
        # Detección de bordes con parámetros más conservadores
        edges = cv2.Canny(enhanced, 50, 150, apertureSize=3)
        
        # Operaciones morfológicas más selectivas
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)
        
        # Eliminar ruido con operación de apertura
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrado más estricto de contornos
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Filtros múltiples:
            # 1. Área mínima y máxima
            if area < 30 or area > 5000:
                continue
                
            # 2. Relación aspecto (evitar formas muy redondas que no son fisuras)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio < 1.5:  # Las fisuras suelen ser alargadas
                continue
                
            # 3. Solidez (relación área/área_convex_hull)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                if solidity < 0.3:  # Evitar formas muy irregulares
                    continue
            
            # 4. Relación perímetro/área (las fisuras tienen alto perímetro vs área)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.5:  # Evitar formas demasiado circulares
                    continue
                    
            filtered_contours.append(contour)
        
        return filtered_contours, cleaned
    
    def analyze_crack_dimensions(self, contours, image_shape):
        """Analizar dimensiones y características de las fisuras"""
        if not contours:
            return None
        
        h, w = image_shape[:2]
        total_area = h * w
        
        # Análisis de todos los contornos
        total_crack_area = sum(cv2.contourArea(c) for c in contours)
        crack_density = total_crack_area / total_area
        
        # Análisis del contorno más grande (fisura principal)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Bounding box
        x, y, bbox_w, bbox_h = cv2.boundingRect(largest_contour)
        
        # Estimación de ancho (basado en área y longitud aproximada)
        arc_length = cv2.arcLength(largest_contour, False)
        estimated_width_px = cv2.contourArea(largest_contour) / arc_length if arc_length > 0 else 0
        
        # Análisis de orientación
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            angle = ellipse[2]
        else:
            angle = 0
        
        return {
            "total_contours": len(contours),
            "total_crack_area": total_crack_area,
            "crack_density": crack_density,
            "largest_contour_area": cv2.contourArea(largest_contour),
            "estimated_width_px": estimated_width_px,
            "bounding_box": (x, y, bbox_w, bbox_h),
            "arc_length": arc_length,
            "orientation_angle": angle,
            "image_dimensions": (w, h)
        }
    
    def classify_severity(self, crack_analysis):
        """Clasificar severidad basada en análisis morfológico mejorado"""
        if crack_analysis is None:
            return "sin_fisura", self.severity_criteria["sin_fisura"]
        
        # Parámetros para clasificación
        width_px = crack_analysis["estimated_width_px"]
        density = crack_analysis["crack_density"]
        total_contours = crack_analysis["total_contours"]
        largest_area = crack_analysis["largest_contour_area"]
        
        # Clasificación más conservadora y realista
        # Si hay muy poca densidad de fisura, probablemente es ruido
        if density < 0.005:
            severity = "sin_fisura"
        # Clasificación por múltiples criterios
        elif (density < 0.01 and width_px < 3.0 and total_contours < 5 and largest_area < 100):
            severity = "superficial"
        elif (density < 0.03 and width_px < 8.0 and total_contours < 15 and largest_area < 500):
            severity = "moderada"
        else:
            # Solo clasificar como estructural si hay evidencia clara
            if density > 0.05 or width_px > 10.0 or largest_area > 800:
                severity = "estructural"
            else:
                severity = "moderada"  # Ser conservador
        
        severity_info = self.severity_criteria[severity].copy()
        severity_info.update({
            "width_px": width_px,
            "density": density,
            "total_contours": total_contours,
            "largest_area": largest_area,
            "confidence": self._calculate_confidence(crack_analysis)
        })
        
        return severity, severity_info
    
    def _calculate_confidence(self, crack_analysis):
        """Calcular confianza de la clasificación"""
        density = crack_analysis["crack_density"]
        width_px = crack_analysis["estimated_width_px"]
        
        # Confianza basada en consistencia de parámetros
        if density < 0.005:
            confidence = 0.9  # Alta confianza en "sin fisura"
        elif density > 0.05:
            confidence = 0.8  # Alta confianza en detección
        else:
            confidence = max(0.4, min(0.8, density * 20))  # Confianza moderada
        
        return round(confidence, 3)
    
    def analyze_single_image(self, image_path, show_analysis=True):
        """Análisis completo de una imagen"""
        print(f"🔍 Analizando: {Path(image_path).name}")
        
        # Preprocesar imagen
        image_resized, image_normalized = self.preprocess_image(image_path)
        
        # Preparar para predicción del modelo
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        # 1. Detección de fisura con modelo CNN
        detection_prob = self.detection_model.predict(image_batch, verbose=0)[0][0]
        
        # Análisis más conservador: solo proceder si la probabilidad es alta
        has_crack_cnn = detection_prob > 0.7  # Umbral más alto para reducir falsos positivos
        
        results = {
            "detection_probability": float(detection_prob),
            "has_crack": False,
            "severity": "sin_fisura",
            "severity_info": self.severity_criteria["sin_fisura"],
            "crack_analysis": None,
            "method": "cnn_only"
        }
        
        # Solo si el CNN está muy confiado, proceder con análisis morfológico
        if has_crack_cnn:
            # 2. Análisis morfológico
            contours, processed = self.detect_crack_contours(image_resized)
            crack_analysis = self.analyze_crack_dimensions(contours, image_resized.shape)
            
            # 3. Validación cruzada: el análisis morfológico debe confirmar la detección
            if crack_analysis and crack_analysis["crack_density"] > 0.003:
                # 4. Clasificación de severidad
                severity, severity_info = self.classify_severity(crack_analysis)
                
                results.update({
                    "has_crack": True,
                    "severity": severity,
                    "severity_info": severity_info,
                    "crack_analysis": crack_analysis,
                    "method": "cnn_and_morphology"
                })
                
                # 5. Visualización
                if show_analysis:
                    self.visualize_analysis(image_resized, contours, results)
            else:
                # CNN detectó fisura pero morfología no la confirma -> falso positivo
                results.update({
                    "has_crack": False,
                    "severity": "sin_fisura",
                    "severity_info": self.severity_criteria["sin_fisura"],
                    "crack_analysis": crack_analysis,
                    "method": "cnn_rejected_by_morphology",
                    "note": "CNN detectó fisura pero análisis morfológico no la confirma"
                })
        else:
            # Probabilidad baja del CNN -> probablemente sin fisura
            results["note"] = f"Probabilidad CNN muy baja ({detection_prob:.3f})"
        
        return results
    
    def visualize_analysis(self, image, contours, results):
        """Visualizar análisis de severidad"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Imagen original
        axes[0].imshow(image)
        axes[0].set_title('Imagen Original')
        axes[0].axis('off')
        
        # Contornos detectados
        contour_img = image.copy()
        if contours:
            cv2.drawContours(contour_img, contours, -1, (255, 0, 0), 2)
        axes[1].imshow(contour_img)
        axes[1].set_title('Contornos Detectados')
        axes[1].axis('off')
        
        # Resultado de clasificación
        severity = results['severity']
        severity_info = results['severity_info']
        
        # Crear imagen de resultado
        result_img = image.copy()
        if results['has_crack'] and contours:
            # Dibujar bounding box
            if results['crack_analysis']:
                x, y, w, h = results['crack_analysis']['bounding_box']
                color_map = {'green': (0, 255, 0), 'orange': (255, 165, 0), 'red': (255, 0, 0)}
                color = color_map.get(severity_info.get('color', 'blue'), (0, 0, 255))
                cv2.rectangle(result_img, (x, y), (x+w, y+h), color, 2)
        
        axes[2].imshow(result_img)
        
        # Título con información de severidad
        if results['has_crack']:
            title = f"Severidad: {severity.upper()}\n{severity_info.get('riesgo', '')}"
            axes[2].set_title(title, color=severity_info.get('color', 'black'))
        else:
            axes[2].set_title('Sin Fisura Detectada', color='green')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Imprimir detalles
        print(f"\n📊 ANÁLISIS DETALLADO:")
        print(f"Probabilidad detección: {results['detection_probability']:.3f}")
        if results['has_crack']:
            print(f"Severidad: {severity.upper()}")
            print(f"Riesgo: {severity_info.get('riesgo', 'N/A')}")
            print(f"Acción: {severity_info.get('accion', 'N/A')}")
            if results['crack_analysis']:
                print(f"Ancho estimado: {results['crack_analysis']['estimated_width_px']:.1f} px")
                print(f"Densidad: {results['crack_analysis']['crack_density']:.3f}")
    
    def batch_analysis(self, image_folder, max_images=10):
        """Análisis por lotes de imágenes"""
        print(f"\n📁 ANÁLISIS POR LOTES")
        print("=" * 50)
        
        image_folder = Path(image_folder)
        image_files = list(image_folder.glob("*.jpg"))[:max_images]
        
        results_summary = {
            "sin_fisura": 0,
            "superficial": 0,
            "moderada": 0,
            "estructural": 0
        }
        
        detailed_results = []
        
        for img_file in image_files:
            print(f"Analizando: {img_file.name}")
            result = self.analyze_single_image(str(img_file), show_analysis=False)
            
            severity = result['severity']
            results_summary[severity] += 1
            detailed_results.append({
                "filename": img_file.name,
                "result": result
            })
        
        # Guardar resultados
        with open('results/severity_analysis/batch_results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        # Mostrar resumen
        print(f"\n📊 RESUMEN DE ANÁLISIS:")
        for severity, count in results_summary.items():
            percentage = (count / len(image_files)) * 100 if image_files else 0
            print(f"   {severity}: {count} ({percentage:.1f}%)")
        
        return results_summary, detailed_results

if __name__ == "__main__":
    # Crear clasificador
    classifier = CrackSeverityClassifier()
    
    print("🔍 CLASIFICADOR DE SEVERIDAD DE FISURAS")
    print("Basado en criterios técnicos de literatura especializada")
    print("\n📋 CRITERIOS DE CLASIFICACIÓN:")
    for severity, criteria in classifier.severity_criteria.items():
        print(f"   {severity.upper()}: {criteria['descripcion']}")
    
    # Ejemplo de uso
    print("\n✅ Clasificador listo para usar!")
    print("Uso: classifier.analyze_single_image('ruta/imagen.jpg')")
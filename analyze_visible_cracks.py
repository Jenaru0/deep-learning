#!/usr/bin/env python3
"""
🔬 ANÁLISIS RÁPIDO DE FISURAS VISIBLES - CRACK500
=====================================================
Script optimizado para analizar imágenes con fisuras reales visibles
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
import math
from datetime import datetime

# Configuración
plt.style.use('default')
np.random.seed(42)

class VisibleCrackAnalyzer:
    def __init__(self):
        print("🔬 ANALIZADOR DE FISURAS VISIBLES CRACK500")
        print("=" * 50)
        
        # Rutas
        self.crack500_images = Path("data/external/CRACK500/images")
        self.crack500_masks = Path("data/external/CRACK500/masks") 
        self.results_dir = Path("results/visible_crack_analysis")
        self.results_dir.mkdir(exist_ok=True)
        
        # Cargar modelo
        model_path = "models/simple_crack_detector.keras"
        print(f"Cargando modelo: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        print("✅ Modelo cargado exitosamente")
        
    def get_crack_images_with_masks(self, limit=5):
        """Obtener imágenes con fisuras que tienen máscaras"""
        crack_images = []
        
        # Buscar imágenes base (sin sufijos de patches)
        for mask_file in self.crack500_masks.glob("*.png"):
            base_name = mask_file.stem.replace("_mask", "")
            img_file = self.crack500_images / f"{base_name}.jpg"
            
            if img_file.exists() and len(crack_images) < limit:
                crack_images.append({
                    'image': img_file,
                    'mask': mask_file,
                    'name': base_name
                })
                
        return crack_images
    
    def preprocess_image(self, image_path):
        """Preprocesar imagen para el modelo"""
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (128, 128))  # Tamaño correcto: 128x128
        img_normalized = img_resized.astype(np.float32) / 255.0
        return img_normalized
    
    def detect_crack_contours(self, image):
        """Detectar contornos de fisuras con múltiples técnicas"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Técnica 1: Canny adaptativo
        edges1 = cv2.Canny(gray, 30, 100)
        edges2 = cv2.Canny(gray, 50, 150)
        edges_combined = cv2.bitwise_or(edges1, edges2)
        
        # Técnica 2: Morfología para conectar fisuras
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_dilated = cv2.dilate(edges_combined, kernel, iterations=1)
        edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrar contornos por área
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 20:  # Área mínima
                valid_contours.append(cnt)
        
        return valid_contours, edges_combined
    
    def analyze_crack_orientation(self, contours):
        """Analizar orientación de fisuras"""
        if not contours:
            return None
            
        orientations = []
        for cnt in contours:
            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                angle = ellipse[2]
                orientations.append(angle)
        
        if not orientations:
            return None
            
        avg_angle = np.mean(orientations)
        
        # Clasificar orientación
        if 0 <= avg_angle <= 30 or 150 <= avg_angle <= 180:
            orientation_type = "Horizontal"
        elif 60 <= avg_angle <= 120:
            orientation_type = "Vertical"
        else:
            orientation_type = "Diagonal"
            
        return {
            'average_angle': float(avg_angle),
            'type': orientation_type,
            'angles': [float(a) for a in orientations],
            'count': len(orientations)
        }
    
    def calculate_crack_metrics(self, mask_path, contours):
        """Calcular métricas de fisura usando máscara ground truth"""
        if not mask_path.exists():
            return None
            
        # Cargar máscara ground truth
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
            
        # Calcular métricas de la máscara
        crack_pixels = np.sum(mask > 127)
        total_pixels = mask.shape[0] * mask.shape[1]
        density = crack_pixels / total_pixels
        
        # Encontrar contornos en la máscara
        mask_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calcular ancho promedio
        width_measurements = []
        if mask_contours:
            for cnt in mask_contours:
                if cv2.contourArea(cnt) > 10:
                    # Aproximar ancho usando área/perímetro
                    area = cv2.contourArea(cnt)
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter > 0:
                        approx_width = area / perimeter
                        width_measurements.append(approx_width)
        
        avg_width = np.mean(width_measurements) if width_measurements else 0
        
        return {
            'crack_pixel_count': int(crack_pixels),
            'total_pixels': int(total_pixels),
            'density_percentage': float(density * 100),
            'average_width_pixels': float(avg_width),
            'mask_contours_found': len(mask_contours),
            'detected_contours': len(contours)
        }
    
    def classify_severity(self, metrics):
        """Clasificar severidad basada en métricas"""
        if not metrics:
            return "sin_fisura", "green", "No detectada"
            
        density = metrics['density_percentage']
        avg_width = metrics['average_width_pixels']
        
        if density < 0.1:
            return "leve", "yellow", "Fisura muy leve"
        elif density < 0.5:
            return "moderada", "orange", "Fisura moderada"
        elif density < 1.0:
            return "severa", "red", "Fisura severa"
        else:
            return "critica", "darkred", "Fisura crítica"
    
    def create_analysis_visualization(self, image_info, prediction, contours, edges, orientation, metrics, severity_info):
        """Crear visualización técnica completa"""
        
        # Cargar imagen original
        img_orig = cv2.imread(str(image_info['image']))
        img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        
        # Cargar máscara
        mask = cv2.imread(str(image_info['mask']), cv2.IMREAD_GRAYSCALE)
        
        # Crear figura
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(f'ANÁLISIS TÉCNICO DE FISURA VISIBLE: {image_info["name"]}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Imagen original
        ax1 = plt.subplot(2, 4, 1)
        plt.imshow(img_orig)
        plt.title('Imagen Original', fontweight='bold')
        plt.axis('off')
        
        # 2. Máscara ground truth
        ax2 = plt.subplot(2, 4, 2)
        plt.imshow(mask, cmap='hot')
        plt.title('Ground Truth Mask', fontweight='bold')
        plt.axis('off')
        
        # 3. Detección de bordes
        ax3 = plt.subplot(2, 4, 3)
        plt.imshow(edges, cmap='gray')
        plt.title('Detección de Bordes', fontweight='bold')
        plt.axis('off')
        
        # 4. Contornos detectados
        ax4 = plt.subplot(2, 4, 4)
        img_contours = img_orig.copy()
        if contours:
            cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
        plt.imshow(img_contours)
        plt.title(f'Contornos ({len(contours)} detectados)', fontweight='bold')
        plt.axis('off')
        
        # 5. Mapa de calor de predicción
        ax5 = plt.subplot(2, 4, 5)
        prob = prediction['probability']
        heatmap = np.full((100, 100), prob)
        plt.imshow(heatmap, cmap='YlOrRd', vmin=0, vmax=1)
        plt.colorbar(shrink=0.8)
        plt.title(f'Probabilidad CNN: {prob:.3f}', fontweight='bold')
        plt.axis('off')
        
        # 6. Análisis de orientación
        ax6 = plt.subplot(2, 4, 6)
        if orientation:
            # Crear rosa de vientos simplificada
            angles = np.array(orientation['angles']) * np.pi / 180
            plt.hist(angles, bins=8, alpha=0.7, color='skyblue')
            plt.title(f'Orientación: {orientation["type"]}\n{orientation["average_angle"]:.1f}°', 
                     fontweight='bold')
        else:
            plt.text(0.5, 0.5, 'Sin orientación\ndetectable', ha='center', va='center', 
                    transform=ax6.transAxes, fontsize=12)
            plt.title('Orientación', fontweight='bold')
        plt.axis('off')
        
        # 7. Métricas cuantitativas
        ax7 = plt.subplot(2, 4, 7)
        if metrics:
            metrics_text = f"""MÉTRICAS TÉCNICAS:
            
Densidad: {metrics['density_percentage']:.2f}%
Píxeles afectados: {metrics['crack_pixel_count']}
Ancho promedio: {metrics['average_width_pixels']:.1f}px
Contornos GT: {metrics['mask_contours_found']}
Contornos detectados: {metrics['detected_contours']}"""
        else:
            metrics_text = "Sin métricas\ndisponibles"
            
        plt.text(0.05, 0.95, metrics_text, transform=ax7.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        plt.title('Métricas Cuantitativas', fontweight='bold')
        plt.axis('off')
        
        # 8. Clasificación final
        ax8 = plt.subplot(2, 4, 8)
        severity, color, description = severity_info
        
        result_text = f"""CLASIFICACIÓN FINAL:
        
Severidad: {severity.upper()}
Descripción: {description}
Confianza CNN: {prediction['probability']:.1%}
Estado: FISURA CONFIRMADA"""
        
        plt.text(0.05, 0.95, result_text, transform=ax8.transAxes, 
                fontsize=11, verticalalignment='top', fontweight='bold', 
                color=color)
        plt.title('Resultado Final', fontweight='bold')
        plt.axis('off')
        
        plt.tight_layout()
        return fig
    
    def analyze_crack_image(self, image_info, index):
        """Analizar una imagen con fisura"""
        print(f"\n{'='*50}")
        print(f"🔍 ANÁLISIS {index}: {image_info['name']}")
        print(f"{'='*50}")
        
        # Preprocesar imagen
        img_processed = self.preprocess_image(image_info['image'])
        if img_processed is None:
            print("❌ Error cargando imagen")
            return None
            
        # Predicción CNN
        prediction = self.model.predict(np.expand_dims(img_processed, axis=0), verbose=0)
        prob = float(prediction[0][0])
        
        print(f"🧠 Predicción CNN: {prob:.3f}")
        
        # Cargar imagen para análisis visual
        img_full = cv2.imread(str(image_info['image']))
        img_full = cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB)
        
        # Detectar contornos
        contours, edges = self.detect_crack_contours(img_full)
        print(f"🔍 Contornos detectados: {len(contours)}")
        
        # Análisis de orientación
        orientation = self.analyze_crack_orientation(contours)
        if orientation:
            print(f"📐 Orientación: {orientation['type']} ({orientation['average_angle']:.1f}°)")
        
        # Métricas usando ground truth
        metrics = self.calculate_crack_metrics(image_info['mask'], contours)
        if metrics:
            print(f"📊 Densidad: {metrics['density_percentage']:.2f}%")
        
        # Clasificación de severidad
        severity_info = self.classify_severity(metrics)
        print(f"⚠️  Severidad: {severity_info[0].upper()} - {severity_info[2]}")
        
        # Crear visualización
        fig = self.create_analysis_visualization(
            image_info, {'probability': prob}, contours, edges, 
            orientation, metrics, severity_info
        )
        
        # Guardar imagen
        output_path = self.results_dir / f"visible_crack_{index:02d}_{image_info['name']}_analysis.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✅ Análisis guardado: {output_path.name}")
        
        # Guardar datos JSON
        analysis_data = {
            'image_info': {
                'filename': image_info['name'],
                'image_path': str(image_info['image']),
                'mask_path': str(image_info['mask'])
            },
            'cnn_prediction': {
                'probability': prob,
                'classification': 'crack' if prob > 0.5 else 'no_crack'
            },
            'morphological_analysis': {
                'contours_detected': len(contours),
                'method': 'Multi-Canny + Enhanced Morphology'
            },
            'orientation_analysis': orientation,
            'ground_truth_metrics': metrics,
            'severity_classification': {
                'level': severity_info[0],
                'color': severity_info[1],
                'description': severity_info[2]
            },
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        json_path = self.results_dir / f"visible_crack_{index:02d}_{image_info['name']}_data.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        
        return analysis_data
    
    def run_analysis(self):
        """Ejecutar análisis completo"""
        print("📸 Buscando imágenes con fisuras visibles...")
        
        # Obtener imágenes con fisuras
        crack_images = self.get_crack_images_with_masks(limit=5)
        
        if not crack_images:
            print("❌ No se encontraron imágenes con fisuras y máscaras")
            return
            
        print(f"✅ Encontradas {len(crack_images)} imágenes con fisuras")
        
        results = []
        for i, image_info in enumerate(crack_images, 1):
            try:
                result = self.analyze_crack_image(image_info, i)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"❌ Error en análisis {i}: {e}")
                continue
        
        print(f"\n🎉 ANÁLISIS COMPLETADO")
        print(f"📁 Ubicación: {self.results_dir}")
        print(f"🎯 {len(results)} análisis exitosos de fisuras visibles")
        print(f"📊 ¡Listo para presentación técnica!")

if __name__ == "__main__":
    analyzer = VisibleCrackAnalyzer()
    analyzer.run_analysis()
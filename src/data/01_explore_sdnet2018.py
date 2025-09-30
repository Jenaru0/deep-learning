"""
🔍 Script de Análisis Exploratorio del Dataset SDNET2018
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import random

class SDNET2018DatasetAnalyzer:
    def __init__(self, dataset_path="C:/Users/jonna/OneDrive/Escritorio/DEEP LEARNING/dataset/SDNET2018"):
        self.dataset_path = Path(dataset_path)
        self.stats = defaultdict(dict)
        
    def analyze_dataset_structure(self):
        """Analizar estructura completa del dataset"""
        print("🔍 ANÁLISIS DEL DATASET SDNET2018")
        print("=" * 50)
        
        structure = {
            "D": {"CD": 0, "UD": 0, "description": "Bridge Decks"},
            "P": {"CP": 0, "UP": 0, "description": "Pavements"}, 
            "W": {"CW": 0, "UW": 0, "description": "Walls"}
        }
        
        total_images = 0
        total_cracks = 0
        total_no_cracks = 0
        
        for surface_type in structure.keys():
            surface_path = self.dataset_path / surface_type
            
            for crack_type in ["C", "U"]:
                folder_name = f"{crack_type}{surface_type}"
                folder_path = surface_path / folder_name
                
                if folder_path.exists():
                    count = len(list(folder_path.glob("*.jpg")))
                    structure[surface_type][folder_name] = count
                    total_images += count
                    
                    if crack_type == "C":
                        total_cracks += count
                    else:
                        total_no_cracks += count
                        
                    print(f"📁 {folder_name}: {count:,} imágenes")
        
        print(f"\n📊 RESUMEN TOTAL:")
        print(f"Total imágenes: {total_images:,}")
        print(f"Con fisuras: {total_cracks:,} ({total_cracks/total_images*100:.1f}%)")
        print(f"Sin fisuras: {total_no_cracks:,} ({total_no_cracks/total_images*100:.1f}%)")
        
        return structure, total_images, total_cracks, total_no_cracks
    
    def sample_images_analysis(self, n_samples=5):
        """Analizar muestra de imágenes de cada categoría"""
        print(f"\n🖼️ ANÁLISIS DE MUESTRA ({n_samples} imágenes por categoría)")
        print("=" * 50)
        
        categories = ["D/CD", "D/UD", "P/CP", "P/UP", "W/CW", "W/UW"]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, category in enumerate(categories):
            folder_path = self.dataset_path / category
            if folder_path.exists():
                images = list(folder_path.glob("*.jpg"))
                if images:
                    # Seleccionar imagen aleatoria
                    sample_img = random.choice(images)
                    
                    # Leer y mostrar imagen
                    img = cv2.imread(str(sample_img))
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    axes[i].imshow(img_rgb)
                    axes[i].set_title(f"{category}\n{sample_img.name}")
                    axes[i].axis('off')
                    
                    # Analizar dimensiones
                    h, w, c = img_rgb.shape
                    print(f"📸 {category}: {w}x{h}x{c} - {sample_img.name}")
        
        plt.tight_layout()
        plt.savefig("data/processed/sdnet2018_samples.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def analyze_image_properties(self, sample_size=100):
        """Analizar propiedades de las imágenes"""
        print(f"\n📏 ANÁLISIS DE PROPIEDADES DE IMÁGENES (muestra: {sample_size})")
        print("=" * 50)
        
        properties = {
            "widths": [], "heights": [], "channels": [],
            "file_sizes": [], "categories": []
        }
        
        categories = ["D/CD", "D/UD", "P/CP", "P/UP", "W/CW", "W/UW"]
        
        for category in categories:
            folder_path = self.dataset_path / category
            if folder_path.exists():
                images = list(folder_path.glob("*.jpg"))
                sample_images = random.sample(images, min(sample_size//6, len(images)))
                
                for img_path in sample_images:
                    # Leer imagen
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        h, w, c = img.shape
                        file_size = img_path.stat().st_size / 1024  # KB
                        
                        properties["widths"].append(w)
                        properties["heights"].append(h)
                        properties["channels"].append(c)
                        properties["file_sizes"].append(file_size)
                        properties["categories"].append(category)
        
        # Crear DataFrame para análisis
        df = pd.DataFrame(properties)
        
        # Estadísticas descriptivas
        print("\n📊 ESTADÍSTICAS DE DIMENSIONES:")
        print(f"Ancho: {df['widths'].min()}-{df['widths'].max()}, promedio: {df['widths'].mean():.1f}")
        print(f"Alto: {df['heights'].min()}-{df['heights'].max()}, promedio: {df['heights'].mean():.1f}")
        print(f"Tamaño archivo: {df['file_sizes'].min():.1f}-{df['file_sizes'].max():.1f} KB, promedio: {df['file_sizes'].mean():.1f}")
        
        # Visualizar distribuciones
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        axes[0,0].hist(df['widths'], bins=20, alpha=0.7)
        axes[0,0].set_title('Distribución de Anchos')
        axes[0,0].set_xlabel('Ancho (px)')
        
        axes[0,1].hist(df['heights'], bins=20, alpha=0.7)
        axes[0,1].set_title('Distribución de Altos')
        axes[0,1].set_xlabel('Alto (px)')
        
        axes[1,0].hist(df['file_sizes'], bins=20, alpha=0.7)
        axes[1,0].set_title('Distribución de Tamaños')
        axes[1,0].set_xlabel('Tamaño (KB)')
        
        sns.countplot(data=df, x='categories', ax=axes[1,1])
        axes[1,1].set_title('Muestras por Categoría')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig("data/processed/sdnet2018_properties.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        return df
    
    def create_unified_structure(self):
        """Crear estructura unificada para el proyecto"""
        print(f"\n🔄 CREANDO ESTRUCTURA UNIFICADA")
        print("=" * 50)
        
        # Crear carpetas de destino
        unified_path = Path("data/processed/sdnet2018_unified")
        unified_path.mkdir(parents=True, exist_ok=True)
        
        (unified_path / "crack").mkdir(exist_ok=True)
        (unified_path / "no_crack").mkdir(exist_ok=True)
        
        # Mapear categorías
        crack_folders = ["D/CD", "P/CP", "W/CW"]
        no_crack_folders = ["D/UD", "P/UP", "W/UW"]
        
        print("📁 Estructura unificada creada:")
        print(f"   📂 {unified_path}/crack/")
        print(f"   📂 {unified_path}/no_crack/")
        
        return str(unified_path)

if __name__ == "__main__":
    # Ejecutar análisis completo
    analyzer = SDNET2018DatasetAnalyzer()
    
    # 1. Analizar estructura
    structure, total, cracks, no_cracks = analyzer.analyze_dataset_structure()
    
    # 2. Analizar muestras
    analyzer.sample_images_analysis()
    
    # 3. Analizar propiedades
    properties_df = analyzer.analyze_image_properties()
    
    # 4. Crear estructura unificada
    unified_path = analyzer.create_unified_structure()
    
    print(f"\n✅ ANÁLISIS COMPLETADO")
    print(f"📊 Dataset listo para entrenamiento en: {unified_path}")
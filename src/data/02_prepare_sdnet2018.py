"""
🔧 Script de Preparación de Datos SDNET2018
Organiza y prepara el dataset para entrenamiento
"""

import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter
import random

class SDNET2018DataPreparator:
    def __init__(self, 
                 source_path="C:/Users/jonna/OneDrive/Escritorio/DEEP LEARNING/dataset/SDNET2018",
                 output_path="data/processed/sdnet2018_prepared"):
        self.source_path = Path(source_path)
        self.output_path = Path(output_path)
        self.train_ratio = 0.7
        self.val_ratio = 0.15
        self.test_ratio = 0.15
        
    def create_output_structure(self):
        """Crear estructura de carpetas para el dataset preparado"""
        print("📁 CREANDO ESTRUCTURA DE CARPETAS")
        print("=" * 50)
        
        # Crear estructura principal
        splits = ['train', 'validation', 'test']
        classes = ['crack', 'no_crack']
        
        for split in splits:
            for class_name in classes:
                folder_path = self.output_path / split / class_name
                folder_path.mkdir(parents=True, exist_ok=True)
                print(f"   📂 {folder_path}")
        
        print(f"\n✅ Estructura creada en: {self.output_path}")
        
    def collect_all_images(self):
        """Recolectar todas las imágenes y sus etiquetas"""
        print(f"\n📊 RECOLECTANDO IMÁGENES")
        print("=" * 50)
        
        all_images = []
        
        # Mapeo de carpetas
        crack_folders = {
            "D/CD": "deck_crack",
            "P/CP": "pavement_crack", 
            "W/CW": "wall_crack"
        }
        
        no_crack_folders = {
            "D/UD": "deck_no_crack",
            "P/UP": "pavement_no_crack",
            "W/UW": "wall_no_crack"
        }
        
        # Recolectar imágenes con fisuras
        for folder, surface_type in crack_folders.items():
            folder_path = self.source_path / folder
            if folder_path.exists():
                images = list(folder_path.glob("*.jpg"))
                for img_path in images:
                    all_images.append({
                        'path': str(img_path),
                        'filename': img_path.name,
                        'label': 'crack',
                        'surface_type': surface_type,
                        'binary_label': 1
                    })
                print(f"   📸 {folder}: {len(images):,} imágenes")
        
        # Recolectar imágenes sin fisuras
        for folder, surface_type in no_crack_folders.items():
            folder_path = self.source_path / folder
            if folder_path.exists():
                images = list(folder_path.glob("*.jpg"))
                for img_path in images:
                    all_images.append({
                        'path': str(img_path),
                        'filename': img_path.name,
                        'label': 'no_crack',
                        'surface_type': surface_type,
                        'binary_label': 0
                    })
                print(f"   📸 {folder}: {len(images):,} imágenes")
        
        # Crear DataFrame
        df = pd.DataFrame(all_images)
        
        print(f"\n📊 RESUMEN TOTAL:")
        print(f"Total imágenes: {len(df):,}")
        print(f"Con fisuras: {len(df[df['label'] == 'crack']):,}")
        print(f"Sin fisuras: {len(df[df['label'] == 'no_crack']):,}")
        
        return df
    
    def balance_dataset(self, df, max_samples_per_class=15000):
        """Balancear el dataset para evitar sobreajuste"""
        print(f"\n⚖️ BALANCEANDO DATASET")
        print("=" * 50)
        
        # Contar muestras por clase
        class_counts = df['label'].value_counts()
        print(f"Antes del balanceo:")
        for class_name, count in class_counts.items():
            print(f"   {class_name}: {count:,}")
        
        # Balancear limitando la clase mayoritaria
        crack_df = df[df['label'] == 'crack']
        no_crack_df = df[df['label'] == 'no_crack']
        
        # Si hay demasiadas muestras sin fisuras, tomar una muestra aleatoria
        if len(no_crack_df) > max_samples_per_class:
            no_crack_df = no_crack_df.sample(n=max_samples_per_class, random_state=42)
            print(f"   Limitando 'no_crack' a {max_samples_per_class:,} muestras")
        
        # Combinar datasets balanceados
        balanced_df = pd.concat([crack_df, no_crack_df], ignore_index=True)
        
        # Mezclar el dataset
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\nDespués del balanceo:")
        class_counts = balanced_df['label'].value_counts()
        for class_name, count in class_counts.items():
            print(f"   {class_name}: {count:,}")
        
        return balanced_df
    
    def split_dataset(self, df):
        """Dividir dataset en train/validation/test"""
        print(f"\n📊 DIVIDIENDO DATASET")
        print("=" * 50)
        
        # División estratificada
        X = df[['path', 'filename', 'surface_type']]
        y = df[['label', 'binary_label']]
        
        # Primera división: train vs (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(self.val_ratio + self.test_ratio), 
            stratify=y['label'], random_state=42
        )
        
        # Segunda división: val vs test
        val_size = self.val_ratio / (self.val_ratio + self.test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1-val_size), 
            stratify=y_temp['label'], random_state=42
        )
        
        # Crear DataFrames finales
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        print(f"📊 División completada:")
        print(f"   🏋️ Entrenamiento: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
        print(f"   ✅ Validación: {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
        print(f"   🧪 Prueba: {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def copy_images_to_splits(self, train_df, val_df, test_df):
        """Copiar imágenes a las carpetas correspondientes"""
        print(f"\n📁 COPIANDO IMÁGENES A CARPETAS")
        print("=" * 50)
        
        splits_data = {
            'train': train_df,
            'validation': val_df,
            'test': test_df
        }
        
        copy_count = 0
        
        for split_name, split_df in splits_data.items():
            print(f"\n📋 Procesando {split_name}...")
            
            for _, row in split_df.iterrows():
                source_path = Path(row['path'])
                
                # Determinar carpeta de destino
                class_folder = 'crack' if row['label'] == 'crack' else 'no_crack'
                dest_folder = self.output_path / split_name / class_folder
                
                # Crear nombre único para evitar conflictos
                surface_prefix = row['surface_type'].split('_')[0][:1].upper()  # D, P, W
                new_filename = f"{surface_prefix}_{row['filename']}"
                dest_path = dest_folder / new_filename
                
                # Copiar archivo
                if source_path.exists() and not dest_path.exists():
                    shutil.copy2(source_path, dest_path)
                    copy_count += 1
                    
                    if copy_count % 1000 == 0:
                        print(f"   📸 Copiadas: {copy_count:,} imágenes...")
        
        print(f"\n✅ Total imágenes copiadas: {copy_count:,}")
        
    def create_metadata_files(self, train_df, val_df, test_df):
        """Crear archivos de metadatos"""
        print(f"\n📄 CREANDO ARCHIVOS DE METADATOS")
        print("=" * 50)
        
        # Guardar DataFrames como CSV
        train_df.to_csv(self.output_path / 'train_metadata.csv', index=False)
        val_df.to_csv(self.output_path / 'validation_metadata.csv', index=False)
        test_df.to_csv(self.output_path / 'test_metadata.csv', index=False)
        
        # Crear resumen general
        summary = {
            'dataset_info': {
                'source': 'SDNET2018',
                'total_images': len(train_df) + len(val_df) + len(test_df),
                'train_samples': len(train_df),
                'validation_samples': len(val_df),
                'test_samples': len(test_df),
                'classes': ['crack', 'no_crack'],
                'image_size': '256x256x3',
                'format': 'JPG'
            }
        }
        
        import json
        with open(self.output_path / 'dataset_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"   📄 train_metadata.csv")
        print(f"   📄 validation_metadata.csv") 
        print(f"   📄 test_metadata.csv")
        print(f"   📄 dataset_summary.json")
        
    def prepare_complete_dataset(self):
        """Ejecutar preparación completa del dataset"""
        print("🚀 INICIANDO PREPARACIÓN COMPLETA DEL DATASET")
        print("=" * 60)
        
        # 1. Crear estructura de carpetas
        self.create_output_structure()
        
        # 2. Recolectar todas las imágenes
        df = self.collect_all_images()
        
        # 3. Balancear dataset
        balanced_df = self.balance_dataset(df)
        
        # 4. Dividir en splits
        train_df, val_df, test_df = self.split_dataset(balanced_df)
        
        # 5. Copiar imágenes
        self.copy_images_to_splits(train_df, val_df, test_df)
        
        # 6. Crear metadatos
        self.create_metadata_files(train_df, val_df, test_df)
        
        print(f"\n🎉 PREPARACIÓN COMPLETADA")
        print(f"📁 Dataset preparado en: {self.output_path}")
        print(f"📊 Listo para entrenamiento!")
        
        return self.output_path

if __name__ == "__main__":
    # Ejecutar preparación completa
    preparator = SDNET2018DataPreparator()
    dataset_path = preparator.prepare_complete_dataset()
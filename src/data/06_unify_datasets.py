"""
Unificación de Datasets: SDNET2018 + CRACK500
=============================================

Script para unificar ambos datasets en un formato estándar
y crear un dataset más grande y balanceado para entrenamiento.
"""

import shutil
import json
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

class DatasetUnifier:
    """Unificar SDNET2018 y CRACK500 en un dataset combinado"""
    
    def __init__(self):
        self.sdnet_path = Path("data/processed/sdnet2018_prepared")
        self.crack500_path = Path("data/external/CRACK500")
        self.output_path = Path("data/processed/unified_dataset")
        self.target_size = (128, 128)
        
        # Crear directorios de salida
        self.output_path.mkdir(parents=True, exist_ok=True)
        
    def analyze_crack500_labels(self):
        """Analizar qué imágenes de CRACK500 tienen fisuras"""
        print("🔍 ANALIZANDO ETIQUETAS DE CRACK500")
        print("=" * 50)
        
        images_dir = self.crack500_path / "images"
        masks_dir = self.crack500_path / "masks"
        
        if not images_dir.exists():
            print("❌ Directorio de imágenes no encontrado")
            return []
        
        image_files = list(images_dir.glob("*.jpg"))
        print(f"📊 Total imágenes encontradas: {len(image_files)}")
        
        # Analizar cuáles tienen máscaras (fisuras)
        crack_images = []
        no_crack_images = []
        
        for img_file in image_files:
            # Buscar máscara correspondiente
            mask_name = img_file.stem + "_mask.png"
            mask_path = masks_dir / mask_name
            
            if mask_path.exists():
                # Verificar si la máscara tiene contenido (fisura real)
                try:
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask is not None and np.sum(mask > 0) > 100:  # Umbral mínimo
                        crack_images.append(img_file)
                    else:
                        no_crack_images.append(img_file)
                except:
                    no_crack_images.append(img_file)
            else:
                # Sin máscara = probablemente sin fisura
                no_crack_images.append(img_file)
        
        print(f"🔴 Imágenes CON fisuras: {len(crack_images)}")
        print(f"🟢 Imágenes SIN fisuras: {len(no_crack_images)}")
        
        return crack_images, no_crack_images
    
    def copy_sdnet2018_data(self):
        """Copiar datos de SDNET2018 al dataset unificado"""
        print("📂 COPIANDO DATOS DE SDNET2018")
        print("=" * 50)
        
        # Crear estructura de directorios
        for split in ['train', 'validation', 'test']:
            for label in ['crack', 'no_crack']:
                (self.output_path / split / label).mkdir(parents=True, exist_ok=True)
        
        copied_files = 0
        
        for split in ['train', 'validation', 'test']:
            for label in ['crack', 'no_crack']:
                source_dir = self.sdnet_path / split / label
                target_dir = self.output_path / split / label
                
                if source_dir.exists():
                    files = list(source_dir.glob("*.jpg"))
                    print(f"   {split}/{label}: {len(files)} archivos")
                    
                    for file_path in files:
                        target_path = target_dir / f"sdnet_{file_path.name}"
                        shutil.copy2(file_path, target_path)
                        copied_files += 1
        
        print(f"✅ SDNET2018 copiado: {copied_files} archivos")
        return copied_files
    
    def process_crack500_images(self, crack_images, no_crack_images, limit_per_class=3000):
        """Procesar y copiar imágenes de CRACK500"""
        print("🖼️ PROCESANDO IMÁGENES DE CRACK500")
        print("=" * 50)
        
        # Limitar cantidad para balancear dataset
        crack_sample = crack_images[:limit_per_class] if len(crack_images) > limit_per_class else crack_images
        no_crack_sample = no_crack_images[:limit_per_class] if len(no_crack_images) > limit_per_class else no_crack_images
        
        print(f"🔴 Procesando {len(crack_sample)} imágenes CON fisuras")
        print(f"🟢 Procesando {len(no_crack_sample)} imágenes SIN fisuras")
        
        # Procesar imágenes con fisuras
        processed_crack = self._process_image_set(crack_sample, "crack", "crack500_crack")
        processed_no_crack = self._process_image_set(no_crack_sample, "no_crack", "crack500_no_crack")
        
        return processed_crack, processed_no_crack
    
    def _process_image_set(self, image_list, label, prefix):
        """Procesar un conjunto de imágenes"""
        processed = []
        
        for i, img_path in enumerate(image_list):
            try:
                # Cargar imagen
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Redimensionar
                img_resized = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
                
                # Nombre de archivo
                new_name = f"{prefix}_{i:04d}.jpg"
                
                processed.append({
                    'original_path': str(img_path),
                    'new_name': new_name,
                    'label': label,
                    'processed_image': img_resized
                })
                
                if (i + 1) % 500 == 0:
                    print(f"   Procesadas: {i + 1}/{len(image_list)}")
                    
            except Exception as e:
                print(f"   ❌ Error procesando {img_path.name}: {e}")
                continue
        
        print(f"✅ {label}: {len(processed)} imágenes procesadas")
        return processed
    
    def create_unified_splits(self, processed_crack, processed_no_crack):
        """Crear splits de entrenamiento balanceados"""
        print("⚖️ CREANDO SPLITS BALANCEADOS")
        print("=" * 50)
        
        # Combinar datos procesados
        all_crack = processed_crack
        all_no_crack = processed_no_crack
        
        print(f"🔴 Total imágenes CON fisuras: {len(all_crack)}")
        print(f"🟢 Total imágenes SIN fisuras: {len(all_no_crack)}")
        
        # Balancear dataset
        min_count = min(len(all_crack), len(all_no_crack))
        balanced_crack = all_crack[:min_count]
        balanced_no_crack = all_no_crack[:min_count]
        
        print(f"⚖️ Balanceado a {min_count} imágenes por clase")
        
        # Crear splits para cada clase
        def create_splits_for_class(data_list, label):
            # 70% train, 15% validation, 15% test
            train_data, temp_data = train_test_split(data_list, test_size=0.3, random_state=42)
            val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
            
            return {
                'train': train_data,
                'validation': val_data,
                'test': test_data
            }
        
        crack_splits = create_splits_for_class(balanced_crack, 'crack')
        no_crack_splits = create_splits_for_class(balanced_no_crack, 'no_crack')
        
        # Guardar imágenes en directorios correspondientes
        total_saved = 0
        
        for split in ['train', 'validation', 'test']:
            for label, splits_data in [('crack', crack_splits), ('no_crack', no_crack_splits)]:
                split_data = splits_data[split]
                target_dir = self.output_path / split / label
                
                for item in split_data:
                    target_path = target_dir / item['new_name']
                    cv2.imwrite(str(target_path), item['processed_image'])
                    total_saved += 1
                
                print(f"   {split}/{label}: {len(split_data)} imágenes")
        
        print(f"✅ Total imágenes CRACK500 guardadas: {total_saved}")
        return total_saved
    
    def create_metadata(self):
        """Crear archivos de metadatos para el dataset unificado"""
        print("📋 CREANDO METADATOS")
        print("=" * 50)
        
        metadata = {}
        total_images = 0
        
        for split in ['train', 'validation', 'test']:
            split_data = {}
            split_total = 0
            
            for label in ['crack', 'no_crack']:
                label_dir = self.output_path / split / label
                if label_dir.exists():
                    files = list(label_dir.glob("*.jpg"))
                    split_data[label] = len(files)
                    split_total += len(files)
            
            split_data['total'] = split_total
            metadata[split] = split_data
            total_images += split_total
        
        metadata['dataset_info'] = {
            'name': 'SDNET2018_CRACK500_Unified',
            'total_images': total_images,
            'image_size': '128x128',
            'classes': ['crack', 'no_crack'],
            'source_datasets': ['SDNET2018', 'CRACK500'],
            'creation_date': '2025-09-26'
        }
        
        # Guardar metadatos
        metadata_path = self.output_path / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Mostrar resumen
        print("📊 DATASET UNIFICADO CREADO:")
        for split, data in metadata.items():
            if split != 'dataset_info':
                print(f"   {split.upper()}:")
                print(f"     🔴 Crack: {data['crack']}")
                print(f"     🟢 No Crack: {data['no_crack']}")
                print(f"     📈 Total: {data['total']}")
        
        print(f"\n📈 TOTAL GENERAL: {metadata['dataset_info']['total_images']} imágenes")
        print(f"📄 Metadatos guardados: {metadata_path}")
        
        return metadata
    
    def unify_datasets(self):
        """Proceso completo de unificación"""
        print("🚀 PROCESO COMPLETO DE UNIFICACIÓN")
        print("=" * 60)
        
        # Paso 1: Analizar CRACK500
        crack_images, no_crack_images = self.analyze_crack500_labels()
        
        # Paso 2: Copiar SDNET2018
        sdnet_files = self.copy_sdnet2018_data()
        
        # Paso 3: Procesar CRACK500
        processed_crack, processed_no_crack = self.process_crack500_images(
            crack_images, no_crack_images, limit_per_class=3000
        )
        
        # Paso 4: Crear splits balanceados
        crack500_files = self.create_unified_splits(processed_crack, processed_no_crack)
        
        # Paso 5: Crear metadatos
        metadata = self.create_metadata()
        
        print("\n" + "=" * 60)
        print("🎉 UNIFICACIÓN COMPLETADA")
        print("=" * 60)
        print(f"✅ Archivos SDNET2018: {sdnet_files}")
        print(f"✅ Archivos CRACK500: {crack500_files}")
        print(f"🎯 Dataset unificado: {metadata['dataset_info']['total_images']} imágenes")
        print(f"📁 Ubicación: {self.output_path}")
        
        return metadata

if __name__ == "__main__":
    unifier = DatasetUnifier()
    unifier.unify_datasets()
"""
📥 Descargador de Dataset CRACK500 usando KaggleHub
Método moderno y simplificado para descargar datasets de Kaggle
"""

import os
import shutil
import zipfile
from pathlib import Path
import kagglehub

class CRACK500Downloader:
    def __init__(self):
        self.dataset_id = "pauldavid22/crack50020220509t090436z001"
        self.project_data_dir = Path("data/external")
        self.crack500_dir = self.project_data_dir / "CRACK500"
        
    def download_crack500(self):
        """Descargar dataset CRACK500 usando kagglehub"""
        print("📥 DESCARGANDO DATASET CRACK500")
        print("=" * 50)
        print(f"Dataset ID: {self.dataset_id}")
        
        try:
            # Descargar usando kagglehub
            print("🔄 Iniciando descarga...")
            download_path = kagglehub.dataset_download(self.dataset_id)
            
            print(f"✅ Descarga completada!")
            print(f"📁 Ruta de descarga: {download_path}")
            
            # Verificar el contenido descargado
            self.analyze_downloaded_content(download_path)
            
            # Organizar en nuestra estructura
            organized_path = self.organize_dataset(download_path)
            
            return organized_path
            
        except Exception as e:
            print(f"❌ Error en la descarga: {e}")
            print("\n🔧 POSIBLES SOLUCIONES:")
            print("1. Verificar conexión a internet")
            print("2. Verificar que kagglehub esté instalado: pip install kagglehub")
            print("3. Verificar que el dataset sea público")
            return None
    
    def analyze_downloaded_content(self, download_path):
        """Analizar el contenido descargado"""
        print(f"\n🔍 ANALIZANDO CONTENIDO DESCARGADO")
        print("=" * 50)
        
        download_path = Path(download_path)
        
        # Listar archivos y carpetas
        total_size = 0
        file_count = 0
        folder_count = 0
        
        print(f"📂 Contenido en: {download_path}")
        
        for item in download_path.rglob("*"):
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                total_size += size_mb
                file_count += 1
                
                # Mostrar archivos principales
                if size_mb > 10:  # Archivos > 10MB
                    print(f"   📄 {item.name}: {size_mb:.1f} MB")
                    
            elif item.is_dir():
                folder_count += 1
                # Contar archivos en cada carpeta
                files_in_folder = len([f for f in item.glob("*") if f.is_file()])
                if files_in_folder > 0:
                    print(f"   📁 {item.name}/: {files_in_folder} archivos")
        
        print(f"\n📊 RESUMEN:")
        print(f"Total archivos: {file_count}")
        print(f"Total carpetas: {folder_count}")
        print(f"Tamaño total: {total_size:.1f} MB")
        
        return {
            "total_files": file_count,
            "total_folders": folder_count,
            "total_size_mb": total_size,
            "path": download_path
        }
    
    def organize_dataset(self, download_path):
        """Organizar dataset en nuestra estructura de proyecto"""
        print(f"\n📁 ORGANIZANDO DATASET EN ESTRUCTURA DEL PROYECTO")
        print("=" * 50)
        
        download_path = Path(download_path)
        
        # Crear directorio de destino
        self.crack500_dir.mkdir(parents=True, exist_ok=True)
        
        # Identificar estructura del dataset
        structure = self.identify_dataset_structure(download_path)
        
        if structure:
            # Copiar archivos organizadamente
            self.copy_organized_files(download_path, structure)
            
            print(f"✅ Dataset organizado en: {self.crack500_dir}")
            return self.crack500_dir
        else:
            print("⚠️ No se pudo identificar la estructura del dataset")
            # Copiar todo tal como está
            shutil.copytree(download_path, self.crack500_dir, dirs_exist_ok=True)
            print(f"📁 Dataset copiado completo en: {self.crack500_dir}")
            return self.crack500_dir
    
    def identify_dataset_structure(self, download_path):
        """Identificar la estructura del dataset CRACK500"""
        print("🔍 Identificando estructura...")
        
        download_path = Path(download_path)
        structure = {
            "images": [],
            "masks": [],
            "labels": [],
            "metadata": []
        }
        
        # Buscar patrones comunes
        for item in download_path.rglob("*"):
            if item.is_file():
                name_lower = item.name.lower()
                
                # Imágenes
                if any(ext in name_lower for ext in ['.jpg', '.jpeg', '.png']):
                    if 'mask' in name_lower or 'gt' in name_lower or 'label' in name_lower:
                        structure["masks"].append(item)
                    else:
                        structure["images"].append(item)
                
                # Metadatos
                elif any(ext in name_lower for ext in ['.csv', '.txt', '.json', '.xml']):
                    structure["metadata"].append(item)
        
        print(f"   📸 Imágenes encontradas: {len(structure['images'])}")
        print(f"   🎭 Máscaras encontradas: {len(structure['masks'])}")
        print(f"   📄 Metadatos encontrados: {len(structure['metadata'])}")
        
        return structure if any(structure.values()) else None
    
    def copy_organized_files(self, source_path, structure):
        """Copiar archivos de forma organizada"""
        print("📋 Copiando archivos organizadamente...")
        
        # Crear subcarpetas
        (self.crack500_dir / "images").mkdir(exist_ok=True)
        (self.crack500_dir / "masks").mkdir(exist_ok=True)
        (self.crack500_dir / "metadata").mkdir(exist_ok=True)
        
        copy_count = 0
        
        # Copiar imágenes
        for img_file in structure["images"]:
            dest_file = self.crack500_dir / "images" / img_file.name
            if not dest_file.exists():
                shutil.copy2(img_file, dest_file)
                copy_count += 1
        
        # Copiar máscaras
        for mask_file in structure["masks"]:
            dest_file = self.crack500_dir / "masks" / mask_file.name
            if not dest_file.exists():
                shutil.copy2(mask_file, dest_file)
                copy_count += 1
        
        # Copiar metadatos
        for meta_file in structure["metadata"]:
            dest_file = self.crack500_dir / "metadata" / meta_file.name
            if not dest_file.exists():
                shutil.copy2(meta_file, dest_file)
                copy_count += 1
        
        print(f"   ✅ {copy_count} archivos copiados")
    
    def create_summary_report(self, organized_path):
        """Crear reporte resumen del dataset"""
        print(f"\n📄 CREANDO REPORTE RESUMEN")
        print("=" * 50)
        
        summary = {
            "dataset_name": "CRACK500",
            "source": "Kaggle - pauldavid22/crack50020220509t090436z001",
            "download_date": str(Path().cwd()),
            "organized_path": str(organized_path),
            "structure": {}
        }
        
        # Contar archivos por categoría
        if organized_path.exists():
            for subfolder in ["images", "masks", "metadata"]:
                subfolder_path = organized_path / subfolder
                if subfolder_path.exists():
                    file_count = len(list(subfolder_path.glob("*")))
                    summary["structure"][subfolder] = file_count
                    print(f"   📁 {subfolder}/: {file_count} archivos")
        
        # Guardar reporte
        import json
        report_file = organized_path / "dataset_summary.json"
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📄 Reporte guardado: {report_file}")
        return summary
    
    def download_and_organize_complete(self):
        """Proceso completo de descarga y organización"""
        print("🚀 PROCESO COMPLETO DE DESCARGA CRACK500")
        print("=" * 60)
        
        # 1. Descargar dataset
        organized_path = self.download_crack500()
        
        if organized_path:
            # 2. Crear reporte
            summary = self.create_summary_report(organized_path)
            
            print(f"\n🎉 DESCARGA COMPLETADA")
            print(f"📁 Dataset disponible en: {organized_path}")
            print(f"📊 Total archivos organizados: {sum(summary['structure'].values())}")
            print(f"✅ Listo para integración con SDNET2018")
            
            return organized_path
        else:
            print(f"\n❌ DESCARGA FALLIDA")
            return None

if __name__ == "__main__":
    # Ejecutar descarga completa
    downloader = CRACK500Downloader()
    result_path = downloader.download_and_organize_complete()
    
    if result_path:
        print(f"\n📋 SIGUIENTE PASO:")
        print(f"Unificar CRACK500 con SDNET2018 para entrenamiento")
    else:
        print(f"\n🔧 REVISAR:")
        print(f"Verificar configuración de kagglehub")
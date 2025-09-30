"""
🔑 Configurador de API de Kaggle para CRACK500 Dataset
"""

import os
import json
from pathlib import Path
import getpass

class KaggleAPIConfigurator:
    def __init__(self):
        self.kaggle_dir = Path.home() / '.kaggle'
        self.credentials_file = self.kaggle_dir / 'kaggle.json'
        
    def setup_kaggle_credentials(self):
        """Configurar credenciales de Kaggle"""
        print("🔑 CONFIGURACIÓN DE API KAGGLE")
        print("=" * 50)
        print("Para descargar el dataset CRACK500, necesitas:")
        print("1. Una cuenta de Kaggle")
        print("2. Tu API Token de Kaggle")
        print("\n📋 PASOS PARA OBTENER API TOKEN:")
        print("1. Ve a kaggle.com e inicia sesión")
        print("2. Ve a 'Account' → 'API' → 'Create New API Token'")
        print("3. Se descargará 'kaggle.json'")
        print("4. Copia el contenido de ese archivo aquí")
        
        # Verificar si ya existe configuración
        if self.credentials_file.exists():
            print(f"\n✅ Archivo de credenciales encontrado: {self.credentials_file}")
            try:
                with open(self.credentials_file, 'r') as f:
                    credentials = json.load(f)
                    username = credentials.get('username', 'No encontrado')
                    print(f"📝 Usuario configurado: {username}")
                    
                response = input("\n¿Deseas usar estas credenciales existentes? (y/n): ")
                if response.lower() == 'y':
                    return True
            except Exception as e:
                print(f"⚠️ Error leyendo credenciales existentes: {e}")
        
        # Configurar nuevas credenciales
        return self.input_new_credentials()
    
    def input_new_credentials(self):
        """Solicitar nuevas credenciales"""
        print(f"\n📝 CONFIGURANDO NUEVAS CREDENCIALES")
        print("Ingresa tu información de Kaggle:")
        
        username = input("Usuario de Kaggle: ").strip()
        if not username:
            print("❌ El usuario no puede estar vacío")
            return False
            
        key = getpass.getpass("API Key (se ocultará al escribir): ").strip()
        if not key:
            print("❌ La API Key no puede estar vacía")
            return False
        
        # Crear directorio kaggle si no existe
        self.kaggle_dir.mkdir(exist_ok=True)
        
        # Crear archivo de credenciales
        credentials = {
            "username": username,
            "key": key
        }
        
        try:
            with open(self.credentials_file, 'w') as f:
                json.dump(credentials, f, indent=2)
            
            # Establecer permisos (solo lectura para el usuario)
            os.chmod(self.credentials_file, 0o600)
            
            print(f"✅ Credenciales guardadas en: {self.credentials_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error guardando credenciales: {e}")
            return False
    
    def test_kaggle_connection(self):
        """Probar conexión con Kaggle"""
        print(f"\n🔍 PROBANDO CONEXIÓN CON KAGGLE")
        print("=" * 50)
        
        try:
            import kaggle
            
            # Probar listando datasets
            print("Obteniendo información de usuario...")
            kaggle.api.authenticate()
            
            # Buscar el dataset específico
            print("Buscando dataset CRACK500...")
            datasets = kaggle.api.dataset_list(search="crack500")
            
            crack500_found = False
            for dataset in datasets:
                if "crack500" in dataset.ref.lower():
                    print(f"✅ Dataset encontrado: {dataset.ref}")
                    print(f"   📊 Título: {dataset.title}")
                    print(f"   📅 Actualizado: {dataset.lastUpdated}")
                    print(f"   📥 Descargas: {dataset.downloadCount}")
                    crack500_found = True
                    break
            
            if not crack500_found:
                print("⚠️ Dataset CRACK500 específico no encontrado en la búsqueda")
                print("Probando acceso directo al dataset...")
                
            return True
            
        except ImportError:
            print("❌ Librería kaggle no instalada")
            return False
        except Exception as e:
            print(f"❌ Error de conexión: {e}")
            print("\n🔧 POSIBLES SOLUCIONES:")
            print("1. Verificar que las credenciales sean correctas")
            print("2. Verificar conexión a internet")
            print("3. Verificar que el archivo kaggle.json tenga permisos correctos")
            return False
    
    def download_crack500_info(self):
        """Obtener información del dataset CRACK500"""
        print(f"\n📊 INFORMACIÓN DEL DATASET CRACK500")
        print("=" * 50)
        
        try:
            import kaggle
            
            dataset_name = "pauldavid22/crack50020220509t090436z001"
            
            # Obtener información del dataset
            dataset_info = kaggle.api.dataset_view(dataset_name)
            
            print(f"📋 Nombre: {dataset_info.title}")
            print(f"👤 Autor: {dataset_info.creatorName}")
            print(f"📅 Creado: {dataset_info.creationDate}")
            print(f"📅 Actualizado: {dataset_info.lastUpdated}")
            print(f"📥 Descargas: {dataset_info.downloadCount}")
            print(f"💾 Tamaño: {dataset_info.totalBytes / (1024*1024*1024):.2f} GB")
            print(f"📁 Archivos: {dataset_info.fileCount}")
            
            return dataset_info
            
        except Exception as e:
            print(f"❌ Error obteniendo información: {e}")
            return None
    
    def setup_complete_kaggle(self):
        """Configuración completa de Kaggle"""
        print("🚀 CONFIGURACIÓN COMPLETA DE KAGGLE API")
        print("=" * 60)
        
        # 1. Configurar credenciales
        if not self.setup_kaggle_credentials():
            print("❌ No se pudieron configurar las credenciales")
            return False
        
        # 2. Probar conexión
        if not self.test_kaggle_connection():
            print("❌ No se pudo establecer conexión con Kaggle")
            return False
        
        # 3. Obtener información del dataset
        dataset_info = self.download_crack500_info()
        
        if dataset_info:
            print(f"\n🎉 CONFIGURACIÓN COMPLETADA")
            print(f"✅ API de Kaggle configurada correctamente")
            print(f"✅ Dataset CRACK500 verificado")
            print(f"📊 Listo para descargar {dataset_info.totalBytes / (1024*1024):.1f} MB")
            return True
        else:
            print(f"\n⚠️ Configuración parcial completada")
            print(f"✅ API configurada, pero hay problemas accediendo al dataset")
            return False

if __name__ == "__main__":
    configurator = KaggleAPIConfigurator()
    success = configurator.setup_complete_kaggle()
    
    if success:
        print(f"\n📋 SIGUIENTE PASO:")
        print(f"Ejecutar descarga del dataset CRACK500")
    else:
        print(f"\n🔧 REVISAR CONFIGURACIÓN")
        print(f"Verificar credenciales y conexión")
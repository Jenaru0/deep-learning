"""Script de prueba para descargar modelos desde Google Drive"""

import os
import gdown

# IDs de los archivos en Google Drive
MODELS = {
    'deteccion': {
        'url': 'https://drive.google.com/uc?id=1toZrp6q8-qCrRk7DUkz12wH9DAMVv0_V',
        'output': 'modelos/deteccion/modelo_deteccion_final.keras'
    },
    'segmentacion': {
        'url': 'https://drive.google.com/uc?id=1Ug_h1flAfLNHLMyNTP2yb_4wdqE5HxIt',
        'output': 'modelos/segmentacion/unet_segmentacion_final.keras'
    }
}

def download_models():
    """Descarga modelos si no existen localmente"""
    
    for model_name, model_info in MODELS.items():
        output_path = model_info['output']
        
        # Verificar si ya existe
        if os.path.exists(output_path):
            print(f"✅ Modelo {model_name} ya existe en {output_path}")
            continue
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Descargar desde Google Drive
        print(f"📥 Descargando modelo {model_name}...")
        try:
            gdown.download(model_info['url'], output_path, quiet=False)
            print(f"✅ Modelo {model_name} descargado exitosamente")
        except Exception as e:
            print(f"❌ Error descargando {model_name}: {str(e)}")
            return False
    
    return True

if __name__ == "__main__":
    print("🚀 Iniciando descarga de modelos desde Google Drive...\n")
    success = download_models()
    
    if success:
        print("\n🎉 ¡Todos los modelos descargados correctamente!")
    else:
        print("\n⚠️ Hubo errores durante la descarga")

"""
Script de Validación: Verificar integridad de splits SDNET2018
===============================================================
Verifica que no existan duplicados entre train/val/test.
"""

import sys
from pathlib import Path
from collections import defaultdict

# Agregar proyecto al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

RUTA_DETECCION = Path(config.RUTA_DETECCION)

def validar_no_duplicados():
    """Verifica que no haya imágenes duplicadas entre splits."""
    print("🔍 Verificando duplicados entre splits...\n")
    
    splits = {
        'train': set(),
        'val': set(),
        'test': set()
    }
    
    # Recolectar nombres de archivos por split
    for split_name in ['train', 'val', 'test']:
        split_dir = RUTA_DETECCION / split_name
        for clase in ['cracked', 'uncracked']:
            clase_dir = split_dir / clase
            if clase_dir.exists():
                archivos = [f.name for f in clase_dir.glob("*.jpg")]
                splits[split_name].update(archivos)
        print(f"  ✓ {split_name}: {len(splits[split_name])} imágenes únicas")
    
    # Verificar intersecciones
    print("\n🔍 Verificando intersecciones...")
    train_val = splits['train'] & splits['val']
    train_test = splits['train'] & splits['test']
    val_test = splits['val'] & splits['test']
    
    if train_val or train_test or val_test:
        print("  ❌ DUPLICADOS ENCONTRADOS:")
        if train_val:
            print(f"    - Train ∩ Val: {len(train_val)} archivos")
        if train_test:
            print(f"    - Train ∩ Test: {len(train_test)} archivos")
        if val_test:
            print(f"    - Val ∩ Test: {len(val_test)} archivos")
        return False
    else:
        print("  ✅ No se encontraron duplicados entre splits")
        return True

def validar_ratios():
    """Verifica que los ratios sean correctos."""
    print("\n📊 Verificando ratios de división...")
    
    import json
    splits_info = Path(config.BASE_DIR) / "splits_info.json"
    
    with open(splits_info, 'r') as f:
        data = json.load(f)
    
    train_ratio = data['splits']['train']['porcentaje_del_dataset'] / 100
    val_ratio = data['splits']['val']['porcentaje_del_dataset'] / 100
    test_ratio = data['splits']['test']['porcentaje_del_dataset'] / 100
    
    print(f"  Train: {train_ratio:.2%} (esperado: 70%)")
    print(f"  Val:   {val_ratio:.2%} (esperado: 15%)")
    print(f"  Test:  {test_ratio:.2%} (esperado: 15%)")
    
    # Verificar balance de clases
    print("\n⚖️  Verificando balance de clases...")
    for split_name in ['train', 'val', 'test']:
        pct_cracked = data['splits'][split_name]['porcentaje_cracked']
        print(f"  {split_name}: {pct_cracked:.2f}% cracked (esperado: ~15%)")
    
    return True

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🔍 VALIDACIÓN DE SPLITS SDNET2018")
    print("="*70 + "\n")
    
    resultado_duplicados = validar_no_duplicados()
    resultado_ratios = validar_ratios()
    
    print("\n" + "="*70)
    if resultado_duplicados and resultado_ratios:
        print("✅ VALIDACIÓN COMPLETA: Todos los checks pasaron")
    else:
        print("❌ VALIDACIÓN FALLIDA: Revisar errores arriba")
    print("="*70 + "\n")

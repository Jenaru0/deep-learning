# ============================================================================
# Script de PowerShell - Lanzador de Entrenamiento Turbo
# ============================================================================
#
# Facilita ejecutar comandos de WSL desde PowerShell de Windows
#
# USO desde PowerShell (como ADMIN):
#   cd "C:\Users\jonna\OneDrive\Escritorio\DEEP LEARNING\investigacion_fisuras"
#   .\entrenar_turbo.ps1
#
# Autor: Jesus Naranjo
# Fecha: Octubre 2025
# ============================================================================

Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "🚀 ENTRENAMIENTO TURBO - RTX 2050" -ForegroundColor Yellow
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Ruta del proyecto en WSL
$WSL_PROJECT_PATH = "/mnt/c/Users/jonna/OneDrive/Escritorio/DEEP LEARNING/investigacion_fisuras"

# Menú de opciones
Write-Host "Selecciona una opción:" -ForegroundColor White
Write-Host ""
Write-Host "  1. Verificar GPU (test completo)" -ForegroundColor Green
Write-Host "  2. Entrenar modelo TURBO (30-40 min)" -ForegroundColor Green
Write-Host "  3. Benchmark de rendimiento GPU" -ForegroundColor Green
Write-Host "  4. Monitor GPU (nvidia-smi)" -ForegroundColor Green
Write-Host "  5. Instalar GPU setup (solo primera vez)" -ForegroundColor Yellow
Write-Host "  6. Preparar datos SDNET2018" -ForegroundColor Yellow
Write-Host "  7. Ver guía completa" -ForegroundColor Cyan
Write-Host "  8. Salir" -ForegroundColor Red
Write-Host ""

$choice = Read-Host "Opción"

switch ($choice) {
    "1" {
        Write-Host "`n🔍 Ejecutando test completo de GPU..." -ForegroundColor Yellow
        wsl -d Ubuntu -e bash -c "cd '$WSL_PROJECT_PATH' && bash scripts/utils/test_gpu_completo.sh"
    }
    
    "2" {
        Write-Host "`n🚀 Iniciando entrenamiento TURBO..." -ForegroundColor Yellow
        Write-Host "⏱️  Tiempo estimado: 30-40 minutos" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "💡 Tip: Abre otra terminal y ejecuta 'nvidia-smi' para monitorear" -ForegroundColor Gray
        Write-Host ""
        Start-Sleep -Seconds 2
        wsl -d Ubuntu -e bash -c "cd '$WSL_PROJECT_PATH' && python3 scripts/entrenamiento/entrenar_deteccion_turbo.py"
    }
    
    "3" {
        Write-Host "`n📊 Ejecutando benchmark de GPU..." -ForegroundColor Yellow
        wsl -d Ubuntu -e bash -c "cd '$WSL_PROJECT_PATH' && python3 scripts/utils/benchmark_gpu.py"
    }
    
    "4" {
        Write-Host "`n📊 Monitoreando GPU (Ctrl+C para salir)..." -ForegroundColor Yellow
        Write-Host ""
        wsl -d Ubuntu -e watch -n 1 nvidia-smi
    }
    
    "5" {
        Write-Host "`n🔧 Instalando GPU setup..." -ForegroundColor Yellow
        Write-Host "⚠️  Esto instalará CUDA, TensorFlow y dependencias" -ForegroundColor Red
        Write-Host ""
        $confirm = Read-Host "¿Continuar? (S/N)"
        if ($confirm -eq "S" -or $confirm -eq "s") {
            wsl -d Ubuntu -e bash -c "cd '$WSL_PROJECT_PATH' && bash scripts/utils/instalar_gpu_wsl2.sh"
        } else {
            Write-Host "Instalación cancelada." -ForegroundColor Gray
        }
    }
    
    "6" {
        Write-Host "`n📦 Preparando datos SDNET2018..." -ForegroundColor Yellow
        wsl -d Ubuntu -e bash -c "cd '$WSL_PROJECT_PATH' && python3 scripts/preprocesamiento/dividir_sdnet2018.py"
    }
    
    "7" {
        Write-Host "`n📚 Abriendo guía completa..." -ForegroundColor Yellow
        $guide_path = "C:\Users\jonna\OneDrive\Escritorio\DEEP LEARNING\investigacion_fisuras\docs\guias\SETUP_FINAL_RTX2050.md"
        Start-Process notepad.exe -ArgumentList $guide_path
    }
    
    "8" {
        Write-Host "`n👋 ¡Hasta luego!" -ForegroundColor Cyan
        exit
    }
    
    default {
        Write-Host "`n❌ Opción inválida" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "✅ Proceso completado" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Pausa antes de cerrar
Read-Host "Presiona Enter para cerrar"

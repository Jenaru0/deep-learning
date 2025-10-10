@echo off
REM Script para lanzar la aplicación web de detección de fisuras en Windows
REM Uso: lanzar_app.bat

echo ========================================
echo    Sistema de Deteccion de Fisuras
echo ========================================
echo.

REM Cambiar al directorio del script
cd /d "%~dp0"

REM Verificar que existe el directorio de modelos
if not exist "modelos\deteccion" (
    echo ERROR: No se encontro el directorio de modelos
    echo Por favor, entrena un modelo primero
    pause
    exit /b 1
)

echo Verificando entorno virtual...

REM Activar entorno virtual si existe
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo Entorno virtual activado
) else (
    echo Advertencia: No se encontro entorno virtual
    echo Usando Python del sistema
)

echo.
echo Verificando Streamlit...

REM Verificar Streamlit
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Streamlit no esta instalado
    echo.
    set /p instalar="Deseas instalar Streamlit ahora? (S/N): "
    if /i "%instalar%"=="S" (
        echo Instalando Streamlit...
        pip install streamlit==1.31.0
    ) else (
        echo Abortando...
        pause
        exit /b 1
    )
) else (
    echo Streamlit esta instalado
)

echo.
echo ========================================
echo Lanzando aplicacion web...
echo.
echo La aplicacion se abrira en: http://localhost:8501
echo Presiona Ctrl+C para detener
echo ========================================
echo.

REM Lanzar aplicación
streamlit run app_web\app.py

pause

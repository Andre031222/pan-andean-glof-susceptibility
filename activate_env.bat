@echo off
:: =============================================================================
:: Activar entorno virtual GLOF Andes (Windows)
:: =============================================================================

echo ==========================================================================
echo   ACTIVANDO ENTORNO VIRTUAL - PROYECTO GLOF ANDES
echo ==========================================================================
echo.

call glof\Scripts\activate.bat

if %errorlevel% equ 0 (
    echo [OK] Entorno activado
    echo.
    python --version
    echo.
    echo  Comandos utiles:
    echo    jupyter notebook        - Iniciar Jupyter Notebook
    echo    jupyter lab             - Iniciar JupyterLab
    echo    deactivate              - Desactivar entorno
    echo.
    echo  Primer uso: abrir notebooks\00_environment_setup.ipynb
) else (
    echo [ERROR] No se pudo activar el entorno.
    echo         Ejecuta primero: setup_env.bat
)

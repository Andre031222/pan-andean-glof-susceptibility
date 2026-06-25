@echo off
:: =============================================================================
:: Configurar entorno virtual GLOF Andes (Windows)
:: Requiere Python 3.11+ en PATH
:: =============================================================================

echo ==========================================================================
echo   CONFIGURACION DEL ENTORNO - PROYECTO GLOF ANDES
echo ==========================================================================
echo.

:: 1. Crear entorno virtual
echo [1/5] Creando entorno virtual en glof\...
python -m venv glof
if %errorlevel% neq 0 ( echo [ERROR] Python 3.11+ requerido & exit /b 1 )

:: 2. Activar
echo [2/5] Activando entorno...
call glof\Scripts\activate.bat

:: 3. Actualizar pip
echo [3/5] Actualizando pip...
python -m pip install --upgrade pip setuptools wheel

:: 4. Instalar dependencias base (sin GPU)
echo [4/5] Instalando dependencias...
pip install -r requirements.txt

:: 5. Kernel de Jupyter
echo [5/5] Registrando kernel Jupyter...
python -m ipykernel install --user --name=glof_andes --display-name="Python (GLOF Andes)"

echo.
echo ==========================================================================
echo   INSTALACION COMPLETADA
echo ==========================================================================
echo.
echo   Para activar el entorno en el futuro:
echo     activate_env.bat
echo.
echo   Para instalar soporte GPU (CUDA 12.x):
echo     pip install cupy-cuda12x
echo.
echo   Para iniciar Jupyter:
echo     jupyter notebook
echo.

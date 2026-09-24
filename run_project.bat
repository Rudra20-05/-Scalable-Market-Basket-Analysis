@echo off
REM ===========================================================
REM Scalable Market Basket Analysis - Windows Launcher
REM ===========================================================

:MENU
cls
echo ===========================================================
echo   Scalable Market Basket Analysis - Project Launcher
echo ===========================================================
echo.
echo   1. Generate dataset
echo   2. Run full analysis pipeline (preprocessing + FP-Growth)
echo   3. Run performance / scalability test
echo   4. Start Streamlit dashboard
echo   5. Install / update dependencies
echo   6. Exit
echo.
set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" goto GENERATE
if "%choice%"=="2" goto PIPELINE
if "%choice%"=="3" goto PERFORMANCE
if "%choice%"=="4" goto DASHBOARD
if "%choice%"=="5" goto INSTALL
if "%choice%"=="6" goto END
echo Invalid choice. Press any key to try again...
pause >nul
goto MENU

:GENERATE
echo.
set /p numtxn="Number of transactions to generate (e.g. 10000): "
python src\generate_dataset.py --num_transactions %numtxn%
echo.
pause
goto MENU

:PIPELINE
echo.
python src\run_pipeline.py
echo.
pause
goto MENU

:PERFORMANCE
echo.
python src\performance_test.py
echo.
pause
goto MENU

:DASHBOARD
echo.
echo Starting Streamlit dashboard... (press Ctrl+C in this window to stop)
streamlit run dashboard\app.py
pause
goto MENU

:INSTALL
echo.
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
pause
goto MENU

:END
echo Goodbye!
exit /b 0

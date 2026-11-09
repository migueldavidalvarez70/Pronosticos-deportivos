@echo off
title Lanzando QBETS...
cd /d "%~dp0"
python -m streamlit run app.py
pause
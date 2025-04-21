:: run_results.bat
@echo off
echo Please enter your defect description:
set /p query="> "
echo %query% > query.txt
python defect_assistant.py

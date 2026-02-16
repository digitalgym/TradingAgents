@echo off
REM Generate quant prompt for manual LLM testing
REM Usage: gen_prompt.bat XAUUSD H1

cd /d "%~dp0\.."
python scripts/generate_quant_prompt.py %*
pause

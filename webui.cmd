@ECHO OFF

REM setup global envvars & create python venv
CALL setenv.cmd
IF NOT %ERRORLEVEL% == 0 GOTO :EOF

REM stat webui service
%PYTHON% webui.py

:EOF

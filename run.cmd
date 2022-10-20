@ECHO OFF

REM setup global envvars & create python venv
CALL setenv.cmd
IF NOT %ERRORLEVEL% == 0 GOTO :EOF

REM start and daemon on all services
%PYTHON% run.py

:EOF

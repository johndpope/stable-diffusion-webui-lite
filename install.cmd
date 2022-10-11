@ECHO OFF

REM setup global envvars & create python venv
CALL setenv.cmd
IF NOT %ERRORLEVEL% == 0 GOTO :EOF

REM git clone repos & pip install packages
%PYTHON% install.py
ECHO.
ECHO Done!
ECHO.
PAUSE

:EOF

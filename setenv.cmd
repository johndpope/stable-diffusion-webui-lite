@ECHO OFF

REM user envvars
IF EXIST setenv-user.cmd CALL setenv-user.cmd

REM default envvars
IF NOT DEFINED GIT SET GIT=git.exe
IF NOT DEFINED PYTHON SET PYTHON=python.exe
IF NOT DEFINED VENV_DIR SET VENV_DIR=venv
IF NOT DEFINED COMMAND_OPTS SET COMMAND_OPTS=

REM echo envvars
ECHO global envvars:
ECHO   GIT          = %GIT%
ECHO   PYTHON       = %PYTHON%
ECHO   VENV_DIR     = %VENV_DIR%
ECHO   COMMAND_OPTS = %COMMAND_OPTS%

REM create tmp folder
SET TMPDIR=tmp
MKDIR %TMPDIR%

REM test init python bin
FOR %%p IN (%PYTHON%) DO SET PYTHON=%%~$PATH:p
ECHO init python bin: %PYTHON%
%PYTHON% -c "" >%TMPDIR%/stdout.txt 2>%TMPDIR%/stderr.txt
IF %ERRORLEVEL% == 0 GOTO :start_venv
ECHO error run init python bin %PYTHON%
GOTO :show_stdout_stderr

:start_venv
REM try activate venv
DIR %VENV_DIR%\Scripts\python.exe >%TMPDIR%/stdout.txt 2>%TMPDIR%/stderr.txt
IF %ERRORLEVEL% == 0 GOTO :activate_venv

REM create venv if not exists
FOR /F "delims=" %%i IN ('CALL %PYTHON% -c "import sys; print(sys.executable)"') DO SET PYTHON_BIN="%%i"
ECHO creating venv in directory "%VENV_DIR%" using %PYTHON_BIN%
%PYTHON_BIN% -m venv %VENV_DIR% >%TMPDIR%/stdout.txt 2>%TMPDIR%/stderr.txt
IF %ERRORLEVEL% == 0 GOTO :activate_venv
ECHO error create venv in directory %VENV_DIR%
GOTO :show_stdout_stderr

REM add venv to PATH & switch to venv python bin
:activate_venv
PATH %VENV_DIR%\Scripts;%PATH%
SET PYTHON=%~dp0%VENV_DIR%\Scripts\Python.exe
ECHO venv python bin: %PYTHON%

REM clean up & exit
RMDIR /S /Q %TMPDIR%
GOTO :EOF


REM handle errors
:show_stdout_stderr
ECHO.
ECHO exit code: %ERRORLEVEL%

FOR /F %%i in ("%TMPDIR%\stdout.txt") DO SET size=%%~zi
IF %size% EQU 0 GOTO :show_stderr
ECHO.
ECHO stdout:
TYPE %TMPDIR%\stdout.txt

:show_stderr
FOR /F %%i in ("%TMPDIR%\stderr.txt") DO SET size=%%~zi
IF %size% EQU 0 GOTO :EOF
ECHO.
ECHO stderr:
TYPE %TMPDIR%\stderr.txt

:EOF

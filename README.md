# Dynamic Positioning – AGX + Python Setup (Windows)

This project runs with **AGX Dynamics** and a matching **CPython** version. Follow the steps to get a working environment.

## Prerequisites 
- AGX installed (user install): `%LOCALAPPDATA%\Algoryx\AGX-2.40.1.5`
- Matching **Python (64-bit)** (typically 3.9.x) Note that the AGX Dynamics version used for this project is only compatible with python v3.9.x


## 1 Detect the required Python version

Open empty Command Prompt (windows search -> command prompt) and insert this command:

```bat
"%LOCALAPPDATA%\Algoryx\AGX-2.40.1.5\python-x64\python.exe" -c "import sys; print(sys.version)"
```
This will detect the required python version for the AGX installation.

## 2 Install required python version

For people new to terminal, \path\to\your\project means just the project directory. You can make an empty folder on your desktop and open the terminal by right-clicking the folder and pressing "open with terminal". Then you insert the following commands:

```bat
cd /d C:\path\to\your\project
py -3.9 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install numpy pyyaml pandas matplotlib
```

After installing the python version, you also have to set the correct installation as the Python Interperter. There is no use in having python v3.9.9 if vscode python interperter is v3.13.7.

## 3 Point the shell at AGX

In the same terminal session, enter this:

```bat
set "AGX_DIR=%LOCALAPPDATA%\Algoryx\AGX-2.40.1.5"
set "PATH=%AGX_DIR%\bin\x64;%PATH%"
set "PYTHONPATH=%AGX_DIR%\bin\x64\agxpy;%AGX_DIR%\data\python\modules;%AGX_DIR%\data\python"
```
This will set the environment variables for your system. If you are not comfortable with the terminal, you can navigate to the system variables window and change it there. 

## 4 Test and run

run this commands in the same terminal session:

```bat
python -c "import agx, agxSDK,agxPythonModules, tutorials, numpy; print('AGX OK')"
```
If terminal prints 'AGX OK', then you are ready to run this command:
```bat
python .\src\main.py
```
This will run whatever python code that resides in main.py, and open AGX Dynamics simulator. The main file can also be run from within vscode, but there may arise some issues that is not accounted for in this step by step guide (f.ex PythonModules import not resolving). A bypass from this error was to just run the python file from the terminal

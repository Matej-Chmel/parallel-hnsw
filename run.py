import functools
from pathlib import Path
import platform
import subprocess
import sys

class AppError(Exception):
	pass

def callScript(executable: Path, stem: str):
	if subprocess.call([executable, f"{stem}.py"], cwd=getScriptsDir()) != 0:
		raise AppError(f"Skript {stem} skončil chybou. Více informací viz výše.")

def checkPythonVersion():
	if sys.version_info.major != 3 or sys.version_info.minor < 7:
		raise AppError(f"Python 3.7 je minimální vyžadovaná verze. Spuštěná verze: {sys.version}.")

@functools.cache
def getRepoDir():
	return Path(__file__).absolute().parent

@functools.cache
def getScriptsDir():
	return getRepoDir() / "src" / "scripts"

def getVirtualEnvExecutable(repoDir: Path):
	p = repoDir / ".venv" / "Scripts" / "python"

	if onWindows():
		p = p.with_suffix(".exe")

	return p.absolute()

def onWindows():
	return platform.system().strip().lower() == "windows"

def run():
	checkPythonVersion()
	callScript(sys.executable, "buildProject")

	executable = getVirtualEnvExecutable(getRepoDir())

	if not executable.exists():
		raise AppError("Virtuální prostředí nebylo sestaveno.")

	callScript(executable, "plotSmall")

def main():
	try:
		run()
	except (AppError, subprocess.SubprocessError) as e:
		print(f"[CHYBA] {e}")

if __name__ == "__main__":
	main()

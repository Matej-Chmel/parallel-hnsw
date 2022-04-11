from pathlib import Path
import platform
import subprocess
import sys

class AppError(Exception):
	pass

def checkPythonVersion():
	if sys.version_info.major != 3 or sys.version_info.minor < 7:
		raise AppError(f"Python 3.7 je minimální vyžadovaná verze. Spuštěná verze: {sys.version}.")

def getVirtualEnvExecutable(repoDir: Path):
	p = repoDir / ".venv" / "Scripts" / "python"

	if onWindows():
		p = p.with_suffix(".exe")

	return p.absolute()

def onWindows():
	return platform.system().strip().lower() == "windows"

def run():
	checkPythonVersion()
	repoDir = Path(__file__).parent
	scriptsDir = repoDir / "src" / "scripts"
	subprocess.call([sys.executable, "buildProject.py"], cwd=scriptsDir)
	executable = getVirtualEnvExecutable(repoDir)

	if not executable.exists():
		raise AppError("Virtuální prostředí nebylo sestaveno.")

	subprocess.call([executable, "plotSmall.py"], cwd=scriptsDir)

def main():
	try:
		run()
	except (AppError, subprocess.SubprocessError) as e:
		print(f"[CHYBA] {e}")

if __name__ == "__main__":
	main()

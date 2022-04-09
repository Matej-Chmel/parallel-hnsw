from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import platform
import subprocess
import sys
import clean

class AppError(Exception):
	pass

@dataclass
class Args:
	clean: bool
	ignorePythonVersion: bool

def buildNativeLib(executable: Path, src: Path):
	print("Building build system for native library.")
	subprocess.call([executable, Path("scripts", "formatCMakeTemplates.py")], cwd=src)
	cmakeBuildDir = src / "cmakeBuild"
	cmakeBuildDir.mkdir(exist_ok=True)
	subprocess.call(["cmake", Path("..")], cwd=cmakeBuildDir)
	print("Build system for native library built.")

def buildVirtualEnv(src: Path):
	print("Building virtual environment.")
	subprocess.call([sys.executable, "-m", "venv", ".venv"], cwd=src)
	executable = getVirtualEnvExecutable(src)

	if not executable.exists():
		raise AppError("Python virtual environment executable not found.")

	cmdline = [executable, "-m", "pip", "install"]
	subprocess.call(cmdline + ["--upgrade", "pip"], cwd=src)
	subprocess.call(cmdline + ["-r", Path("scripts") / "requirements.txt"], cwd=src)
	print("Virtual environment built.")
	return executable

def checkPythonVersion(args: Args):
	if not args.ignorePythonVersion and (sys.version_info.major != 3 or sys.version_info.minor < 7):
		raise AppError("Python 3.7 is the minimum required version.")

def cleanProject(args: Args):
	if args.clean:
		print("Cleaning project.")
		clean.cleanProject(False)
		print("Project cleaned.")

def getArgs():
	p = ArgumentParser(
		"BUILD",
		"Builds Python virtual environment, bindings and build system for native library."
	)
	p.add_argument(
		"-c", "--clean", action="store_true",
		help="Cleans the project before the build."
	)
	p.add_argument(
		"-i", "--ignorePythonVersion", action="store_true",
		help="Skips Python version check."
	)
	args = p.parse_args()
	return Args(args.clean, args.ignorePythonVersion)

def getVirtualEnvExecutable(repoDir: Path):
	p = repoDir / ".venv" / "Scripts" / "python"

	if onWindows():
		p = p.with_suffix(".exe")

	return p

def onWindows():
	return platform.system().strip().lower() == "windows"

def run():
	args = getArgs()
	cleanProject(args)
	checkPythonVersion(args)
	src = Path(__file__).parent.parent
	executable = buildVirtualEnv(src)
	buildNativeLib(executable, src)
	print("Completed.")

def main():
	try:
		run()
	except AppError as e:
		print(f"[APP ERROR] {e}")
	except subprocess.SubprocessError as e:
		print(f"[SUBPROCESS ERROR] {e}")

if __name__ == "__main__":
	main()

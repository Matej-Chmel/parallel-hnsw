from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import clean

class AppError(Exception):
	pass

@dataclass
class Args:
	clean: bool
	ignorePythonVersion: bool

def buildBindings(executable: Path, repoDir: Path):
	print("Building bindings.")
	subprocess.call([executable, "-m", "pip", "install", "."], cwd=repoDir / "src")
	output = subprocess.check_output(
		[executable, "-c", "import parallel_hnsw as h; print(h.__doc__)"]
	).decode("utf-8")
	print(f"Module docstring: {output}")
	print("Bindings built.")

def buildNativeLib(executable: Path, repoDir: Path):
	print("Building build system for native library.")

	if shutil.which("cmake") is None:
		raise AppError("CMake is not installed or it can't be found in environment variable Path.")

	subprocess.call([executable, Path("src", "scripts", "formatCMakeTemplates.py")], cwd=repoDir)
	cmakeBuildDir = repoDir / "cmakeBuild"
	cmakeBuildDir.mkdir(exist_ok=True)
	subprocess.call(["cmake", Path("..")], cwd=cmakeBuildDir)
	print("Build system for native library built.")

def buildVirtualEnv(repoDir: Path):
	print("Building virtual environment.")
	subprocess.call([sys.executable, "-m", "venv", ".venv"], cwd=repoDir)
	executable = getVirtualEnvExecutable(repoDir)

	if not executable.exists():
		raise AppError("Python virtual environment executable not found.")

	cmdline = [executable, "-m", "pip", "install"]
	subprocess.call(cmdline + ["--upgrade", "pip"], cwd=repoDir)
	subprocess.call(cmdline + ["-r", Path("src", "scripts", "requirements.txt")], cwd=repoDir)
	print("Virtual environment built.")
	return executable

def checkPythonVersion(args: Args):
	if not args.ignorePythonVersion and (sys.version_info.major != 3 or sys.version_info.minor < 7):
		raise AppError("Python 3.7 is the minimum required version.")

def cleanProject(args: Args):
	if args.clean:
		print("Cleaning project.")
		clean.cleanProject()
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

	return p.absolute()

def onWindows():
	return platform.system().strip().lower() == "windows"

def run():
	args = getArgs()
	cleanProject(args)
	checkPythonVersion(args)
	repoDir = Path(__file__).parents[2]
	executable = buildVirtualEnv(repoDir)
	buildNativeLib(executable, repoDir)
	buildBindings(executable, repoDir)
	runBindingsTest(executable, repoDir)
	print("Completed.")

def runBindingsTest(executable: Path, repoDir: Path):
	print("Running test of bindings.")
	subprocess.call([executable, "bindingsTest.py"], cwd=repoDir / "src" / "scripts")
	print("Bindings tested.")

def main():
	try:
		run()
	except (AppError, subprocess.SubprocessError) as e:
		print(f"[ERROR] {e}")
		raise SystemExit(1)

if __name__ == "__main__":
	main()

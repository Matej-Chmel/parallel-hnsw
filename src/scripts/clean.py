from ntpath import join
from pathlib import Path
import shutil

def cleanProject(deleteVenv: bool):
	repoDir = Path(__file__).parents[2]
	src = Path("src")

	for path in [
		"__pycache__", "cmakeBuild", "CMakeLists.txt", src / "build", src / "dist",
		src / "parallel_hnsw.egg-info", src / "plots", src / "scripts" / "__pycache__"
	]:
		deleteFile(repoDir / path)

	if deleteVenv:
		deleteFile(repoDir / ".venv")

def deleteFile(p: Path):
	pathStr = f"[{p}] "

	if p.exists():
		try:
			if p.is_dir():
				shutil.rmtree(p)
			elif p.is_file():
				p.unlink()
			else:
				return print(f"{pathStr}Unknown file type.")
			print(f"{pathStr}Deleted.")
		except PermissionError:
			print(f"{pathStr}Permission denied.")
	else:
		print(f"{pathStr}Does not exist.")

def main():
	cleanProject(True)

if __name__ == "__main__":
	main()

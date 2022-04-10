from pathlib import Path
import shutil

def cleanProject(deleteVenv: bool):
	repoDir = Path(__file__).parents[2]
	src = repoDir / "src"

	for path in [
		"__pycache__", "cmakeBuild", src / "scripts" / "__pycache__",
		src / "build", src / "dist", src / "parallel_hnsw.egg-info"
	]:
		deleteDir(repoDir / path)

	if deleteVenv:
		deleteDir(repoDir / ".venv")

def deleteDir(p: Path):
	pathStr = f"[{p}] "

	if p.exists():
		try:
			shutil.rmtree(p)
			print(f"{pathStr}Deleted.")
		except PermissionError:
			print(f"{pathStr}Permission denied.")
	else:
		print(f"{pathStr}Does not exist.")

def main():
	cleanProject(True)

if __name__ == "__main__":
	main()

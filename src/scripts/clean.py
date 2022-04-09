from pathlib import Path
import shutil

def cleanProject(deleteVenv: bool):
	src = Path(__file__).parent.parent

	for path in ["__pycache__"]:
		deleteDir(src / path)

	if deleteVenv:
		deleteDir(src.parent / ".venv")

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

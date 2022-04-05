import json
from pathlib import Path
from Dataset import Dataset

def run():
	srcDir = Path(__file__).parent.parent
	dataDir = srcDir / "data"

	with (srcDir / "config" / "datasetGeneratorConfig.json").open("r", encoding="utf-8") as f:
		arr = json.load(f)

	for obj in arr:
		writeDataset(obj, dataDir)

def writeDataset(obj: dict, dataDir: Path):
	Dataset(
		obj["angular"], obj["dim"], obj["k"], obj["testCount"], obj["trainCount"], obj["seed"]
	).generateAndWrite(obj["name"], dataDir)

def main():
	try:
		run()
	except FileNotFoundError:
		print("Could not open configuration file.")
	except KeyError as e:
		print(f"Missing key {e.args[0]}")

if __name__ == "__main__":
	main()

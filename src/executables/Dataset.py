from dataclasses import dataclass
from io import BufferedWriter
import h5py as hdf
from pathlib import Path
import numpy as np
from sklearn.neighbors import NearestNeighbors
import struct

N = "\n"

def generateData(count: int, dim: int, seed: int):
	np.random.seed(seed)
	return np.random.rand(count, dim).astype(np.float32)

def getNeighbors(angular: bool, k: int, test: np.ndarray, train: np.ndarray):
	nn = NearestNeighbors(algorithm="brute", metric="cosine" if angular else "l2")
	nn.fit(train)
	return nn.kneighbors(test, n_neighbors=k, return_distance=False)

def numpyArrayToStr(arr: np.ndarray, name:str, format: str = ""):
	res = f"{name}[length {arr.shape[0] * arr.shape[1]}]{N}"

	for i in arr.flat:
		res += f"{i:{format}}{N}"

	return res

@dataclass
class Dataset:
	angular: bool
	dim: int
	k: int
	testCount: int
	trainCount: int
	seed: int

	def __post_init__(self):
		self.generated = False
		self.name: str = None

	def __str__(self):
		return (
			f"Dataset {self.name}: {'angular' if self.angular else 'euclidean'} space, "
			f"dimension = {self.dim}, trainCount = {self.trainCount}, testCount = {self.testCount}, k = {self.k}"
		)

	@classmethod
	def fromHDF(cls, p: Path):
		with hdf.File(p, "r") as f:
			d = Dataset(
				f.attrs["distance"] == "angular", f["train"].shape[1], f["neighbors"].shape[1],
				f["test"].shape[0], f["train"].shape[0], None
			)
			d.neighbors = f["neighbors"][:]
			d.test = f["test"][:]
			d.train = f["train"][:]
			d.generated = True
			d.name = p.name
			return d

	def generate(self):
		if self.generated:
			return

		self.train = generateData(self.trainCount, self.dim, self.seed)
		self.test = generateData(self.testCount, self.dim, self.seed + 1)
		self.neighbors = getNeighbors(self.angular, self.k, self.test, self.train).astype(np.uint32)
		self.generated = True

	def generateAndWrite(self, name: str, outputDir: Path):
		self.generate()
		outputDir.mkdir(exist_ok=True)
		path = outputDir / name
		binaryPath = path.with_suffix(".bin")
		hdfPath = path.with_suffix(".hdf5")

		with binaryPath.open("wb") as f:
			self.writeBinary(f)
		print(f"Wrote binary dataset to {binaryPath}.")

		with hdf.File(hdfPath, "w") as f:
			self.writeHDF(f)
		print(f"Wrote HDF dataset to {hdfPath}.")

	def getDescription(self):
		return "\n".join([
			f"angular: {self.angular}",
			f"dim: {self.dim}",
			f"k: {self.k}",
			f"testCount: {self.testCount}",
			f"trainCount: {self.trainCount}"
		]) + "\n" + numpyArrayToStr(self.neighbors, "neighbors") + numpyArrayToStr(self.test, "test", ".6f") + numpyArrayToStr(self.train, "train", ".6f")

	def writeBinary(self, f: BufferedWriter):
		f.write(struct.pack("<?IIII", self.angular, self.dim, self.k, self.testCount, self.trainCount))
		self.neighbors.flatten().tofile(f)
		self.test.flatten().tofile(f)
		self.train.flatten().tofile(f)

	def writeDescription(self, p: Path):
		with p.open("w", encoding="utf-8") as f:
			f.write(self.getDescription())

	def writeHDF(self, f: hdf.File):
		f.create_dataset("neighbors", data=self.neighbors)
		f.create_dataset("test", data=self.test)
		f.create_dataset("train", data=self.train)
		f.attrs["distance"] = "angular" if self.angular else "euclidean"
		f.attrs["point_type"] = "float"

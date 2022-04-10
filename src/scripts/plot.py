from dataclasses import dataclass
import parallel_hnsw as h
from matplotlib import pyplot as plt

class AppError(Exception):
	pass

@dataclass
class Point:
	x: float
	y: float

@dataclass
class Line:
	name: str
	points: list[Point]

	def getXArray(self):
		return [p.x for p in self.points]

	def getYArray(self):
		return [p.y for p in self.points]

	def plot(self):
		plt.plot(self.getXArray(), self.getYArray(), label=self.name)

class Stats:
	def __init__(self, b: h.Benchmark):
		b.run()
		self.build: h.BenchmarkStats = b.getBuildStats()
		self.name = f"paralelní-{b.workers}" if b.parallel else "sekvenční"
		self.query: dict[int, h.QueryBenchmarkStats] = b.getQueryStats()
		self.testCount = b.dataset.testCount
		self.trainCount = b.dataset.trainCount

	def getBuildPoint(self):
		return Point(self.trainCount, self.build.avg.total_seconds())

	def getRecallLine(self):
		return Line(self.name, sorted([
			Point(v.avgRecall, 1 / (v.avg.total_seconds() / self.testCount))
			for v in self.query.values()
		], key=lambda p: p.x))

def getAvailableSIMD():
	best = h.getBestSIMDType()

	if best == h.SIMDType.AVX512:
		return [h.SIMDType.AVX512, h.SIMDType.AVX, h.SIMDType.SSE, h.SIMDType.NONE]
	if best == h.SIMDType.AVX:
		return [h.SIMDType.AVX, h.SIMDType.SSE, h.SIMDType.NONE]
	if best == h.SIMDType.SSE:
		return [h.SIMDType.SSE, h.SIMDType.NONE]
	return [h.SIMDType.NONE]

def getBuildLine(statsList: list[Stats]):
	if not statsList:
		raise AppError("No stats.")
	if not all(s.name == statsList[0].name for s in statsList):
		raise AppError("Name mismatch.")
	return Line(statsList[0].name, sorted([s.getBuildPoint() for s in statsList], key=lambda p: p.x))

@dataclass
class BenchmarkList:
	dim: int
	efConstruction: int
	efSearchValues: list[int]
	mMax: int
	simdType: h.Space
	space: h.Space
	trainCounts: list[int]
	workerCounts: list[int]
	k: int = 10
	runs: int = 5

	def __post_init__(self):
		self.parStats: dict[int, dict[int, Stats]] = {w: {} for w in self.workerCounts}
		self.seqStats: dict[int, Stats] = {}

	def getMetricStr(self):
		if self.space == h.Space.EUCLIDEAN:
			return "eukleidovská vzdálenost"
		return "kosinusová podobnost"

	def plotBuild(self):
		plt.clf()
		getBuildLine(list(self.seqStats.values())).plot()

		for w in self.workerCounts:
			getBuildLine(list(self.parStats[w].values())).plot()

		plt.legend()
		plt.title(f"Dimenze: {self.dim}, metrika: {self.getMetricStr()}")
		plt.xlabel("Počet prvků při stavbě")
		plt.ylabel("Čas stavby (s)")
		plt.show()

	def plotRecall(self, trainCount: int):
		plt.clf()
		self.seqStats[trainCount].getRecallLine().plot()

		for w in self.workerCounts:
			self.parStats[w][trainCount].getRecallLine().plot()

		plt.legend()
		plt.title(f"Dimenze: {self.dim}, metrika: {self.getMetricStr()}, počet prvků: {trainCount}")
		plt.xlabel("Přesnost")
		plt.ylabel("Počet dotazů za sekundu (1/s)")
		plt.show()

	def run(self):
		for trainCount in self.trainCounts:
			dataset = h.Dataset(
				self.dim, self.k, 105, self.space, self.simdType, max(trainCount // 10, 1), trainCount
			)
			b = h.Benchmark(
				dataset, self.efConstruction, self.efSearchValues,
				200, self.mMax, False, self.runs
			)
			self.seqStats[trainCount] = Stats(b)

			for w in self.workerCounts:
				self.parStats[w][trainCount] = Stats(b.getParallel(w))

def main():
	b = BenchmarkList(
		25, 200, [10, 50, 100, 200, 400], 16, h.SIMDType.NONE, h.Space.EUCLIDEAN,
		[100, 500, 1000, 2000], [1, 2, 3, 4]
	)
	b.run()
	b.plotBuild()
	b.plotRecall(2000)

if __name__ == "__main__":
	main()

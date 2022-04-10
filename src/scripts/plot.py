from dataclasses import dataclass
from functools import cached_property
import parallel_hnsw as h
from matplotlib import pyplot as plt

class AppError(Exception):
	pass

@dataclass
class Config:
	dim: int
	efConstruction: int
	efSearchValues: list[int]
	mMax: int
	runs: int
	trainCounts: list[int]
	workerCounts: list[int]

	@cached_property
	def maxTrainCount(self):
		return max(self.trainCounts)

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

		if b.dataset.SIMD != h.SIMDType.NONE:
			self.name += f"-{h.SIMDTypeToStr(b.dataset.SIMD)}"

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
		return [h.SIMDType.AVX512, h.SIMDType.AVX, h.SIMDType.SSE,]
	if best == h.SIMDType.AVX:
		return [h.SIMDType.AVX, h.SIMDType.SSE]
	if best == h.SIMDType.SSE:
		return [h.SIMDType.SSE]
	return []

def getBuildLine(statsList: list[Stats]):
	if not statsList:
		raise AppError("No stats.")
	if not all(s.name == statsList[0].name for s in statsList):
		raise AppError("Name mismatch.")
	return Line(statsList[0].name, sorted([s.getBuildPoint() for s in statsList], key=lambda p: p.x))

def getMetric(space: h.Space):
	if space == h.Space.EUCLIDEAN:
		return "eukleidovská vzdálenost"
	return "kosinusová podobnost"

def plotBuild(dim: int, metric: str, *statsCol: list[Stats]):
	fig, _ = plt.subplots(figsize=(12, 7))

	for s in statsCol:
		getBuildLine(s).plot()

	plt.legend()
	plt.title(f"Dimenze: {dim}, metrika: {metric}")
	plt.xlabel("Počet prvků při stavbě")
	plt.ylabel("Čas stavby (s)")
	plt.show()

def plotRecall(dim: int, metric: str, trainCount: int, *stats: Stats):
	fig, _ = plt.subplots(figsize=(12, 7))

	for s in stats:
		s.getRecallLine().plot()

	plt.legend()
	plt.title(f"Dimenze: {dim}, metrika: {metric}, počet prvků: {trainCount}")
	plt.xlabel("Přesnost")
	plt.ylabel("Počet dotazů za sekundu (1/s)")
	plt.show()

@dataclass
class BenchmarkList:
	dim: int
	efConstruction: int
	efSearchValues: list[int]
	mMax: int
	runs: int
	simdType: h.SIMDType
	space: h.Space
	trainCounts: list[int]
	workerCounts: list[int]
	k: int = 10

	def __post_init__(self):
		self.parStats: dict[int, dict[int, Stats]] = {w: {} for w in self.workerCounts}
		self.seqStats: dict[int, Stats] = {}

	def getParStats(self, trainCount: int = None):
		return (
			[list(self.parStats[w].values()) for w in self.workerCounts]
			if trainCount is None
			else [self.parStats[w][trainCount] for w in self.workerCounts]
		)

	def getSeqStats(self, trainCount: int = None):
		return list(self.seqStats.values()) if trainCount is None else self.seqStats[trainCount]

	def plotBuild(self):
		plotBuild(self.dim, getMetric(self.space), self.getSeqStats(), *self.getParStats())

	def plotRecall(self, trainCount: int):
		plotRecall(
			self.dim, getMetric(self.space), trainCount, self.seqStats[trainCount],
			*[self.parStats[w][trainCount] for w in self.workerCounts]
		)

	def run(self):
		if len(self.seqStats) > 0:
			return

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

def getBenchmarkList(cfg: Config, simdType: h.SIMDType, space: h.Space):
	res = BenchmarkList(
		dim=cfg.dim, efConstruction=cfg.efConstruction, efSearchValues=cfg.efSearchValues,
		mMax=cfg.mMax, runs=cfg.runs, simdType=simdType, space=space, trainCounts=cfg.trainCounts,
		workerCounts=cfg.workerCounts
	)
	res.run()
	return res

def run(cfg: Config):
	runForSpace(cfg, h.Space.EUCLIDEAN)
	runForSpace(cfg, h.Space.ANGULAR)

def runForSpace(cfg: Config, space: h.Space):
	noSIMDBenchmarks = getBenchmarkList(cfg, h.SIMDType.NONE, space)
	noSIMDBenchmarks.plotBuild()
	noSIMDBenchmarks.plotRecall(cfg.maxTrainCount)

	availableSIMD = getAvailableSIMD()

	if not availableSIMD:
		return

	SIMDBenchmarks = {SIMD: getBenchmarkList(cfg, SIMD, space) for SIMD in availableSIMD}
	bestSIMDBenchmarks = SIMDBenchmarks[h.getBestSIMDType()]

	plotBuild(
		cfg.dim, getMetric(space),
		noSIMDBenchmarks.getSeqStats(),
		*[b.getSeqStats() for b in SIMDBenchmarks.values()]
	)
	plotRecall(
		cfg.dim, getMetric(space), cfg.maxTrainCount,
		noSIMDBenchmarks.getSeqStats(cfg.maxTrainCount),
		*[b.getSeqStats(cfg.maxTrainCount) for b in SIMDBenchmarks.values()]
	)
	plotBuild(
		cfg.dim, getMetric(space),
		bestSIMDBenchmarks.getSeqStats(),
		*noSIMDBenchmarks.getParStats()
	)
	plotRecall(
		cfg.dim, getMetric(space), cfg.maxTrainCount,
		bestSIMDBenchmarks.getSeqStats(cfg.maxTrainCount),
		*noSIMDBenchmarks.getParStats(cfg.maxTrainCount)
	)
	bestSIMDBenchmarks.plotBuild()
	bestSIMDBenchmarks.plotRecall(cfg.maxTrainCount)

def main():
	run(Config(
		dim=25, efConstruction=200, efSearchValues=[10, 20, 40, 80, 120, 200, 400, 600],
		mMax=16, runs=1, trainCounts=[100, 500, 1000, 2000], workerCounts=[1, 2, 3, 4]
	))

if __name__ == "__main__":
	main()

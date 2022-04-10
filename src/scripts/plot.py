from dataclasses import dataclass
from functools import cached_property
import parallel_hnsw as h
from pathlib import Path
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

def getCzechMetric(space: h.Space):
	if space == h.Space.EUCLIDEAN:
		return "eukleidovská vzdálenost"
	return "kosinusová podobnost"

def getEnglishMetric(space: h.Space):
	if space == h.Space.EUCLIDEAN:
		return "euclidean"
	return "angular"

def plot(
	lines: list[Line], dim: int, metric: str, xLabel: str, yLabel: str,
	plotsDir: Path, name: str, trainCount: int = None
):
	fig, _ = plt.subplots(figsize=(12, 7))

	for l in lines:
		l.plot()

	plt.legend()
	plt.title(
		f"Dimenze: {dim}, metrika: {metric}{f', počet prvků: {trainCount}' if trainCount else ''}"
	)
	plt.xlabel(xLabel)
	plt.ylabel(yLabel)
	fig.savefig(plotsDir / f"{name}.svg")
	plt.show()

def plotBuild(dim: int, metric: str, plotsDir: Path, name: str, *statsCol: list[Stats]):
	plot(
		[getBuildLine(s) for s in statsCol], dim=dim, metric=metric,
		xLabel="Počet prvků při stavbě", yLabel="Čas stavby (s)",
		plotsDir=plotsDir, name=name
	)

def plotRecall(dim: int, metric: str, trainCount: int, plotsDir: Path, name: str, *stats: Stats):
	plot(
		[s.getRecallLine() for s in stats], dim=dim, metric=metric,
		xLabel="Přesnost", yLabel="Počet dotazů za sekundu (1/s)",
		plotsDir=plotsDir, name=name, trainCount=trainCount
	)

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

	def plotBuild(self, plotsDir: Path, name: str):
		plotBuild(
			self.dim, getCzechMetric(self.space), plotsDir, name,
			self.getSeqStats(), *self.getParStats(),
		)

	def plotRecall(self, trainCount: int, plotsDir: Path, name: str):
		plotRecall(
			self.dim, getCzechMetric(self.space), trainCount, plotsDir, name,
			self.seqStats[trainCount], *[self.parStats[w][trainCount] for w in self.workerCounts]
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
	plotsDir = Path(__file__).parents[1] / "plots"
	plotsDir.mkdir(exist_ok=True)
	runForSpace(cfg, h.Space.EUCLIDEAN, plotsDir)
	runForSpace(cfg, h.Space.ANGULAR, plotsDir)

def runForSpace(cfg: Config, space: h.Space, plotsDir: Path):
	noSIMDBenchmarks = getBenchmarkList(cfg, h.SIMDType.NONE, space)
	noSIMDBenchmarks.plotBuild(plotsDir, f"no_simd_build_{getEnglishMetric(space)}")
	noSIMDBenchmarks.plotRecall(
		cfg.maxTrainCount, plotsDir,
		f"no_simd_recall_{getEnglishMetric(space)}"
	)

	availableSIMD = getAvailableSIMD()

	if not availableSIMD:
		return

	SIMDBenchmarks = {SIMD: getBenchmarkList(cfg, SIMD, space) for SIMD in availableSIMD}
	bestSIMDBenchmarks = SIMDBenchmarks[h.getBestSIMDType()]

	plotBuild(
		cfg.dim, getCzechMetric(space), plotsDir, f"simd_sequential_build_{getEnglishMetric(space)}",
		noSIMDBenchmarks.getSeqStats(),
		*[b.getSeqStats() for b in SIMDBenchmarks.values()]
	)
	plotRecall(
		cfg.dim, getCzechMetric(space), cfg.maxTrainCount, plotsDir,
		f"simd_sequential_recall_{getEnglishMetric(space)}",
		noSIMDBenchmarks.getSeqStats(cfg.maxTrainCount),
		*[b.getSeqStats(cfg.maxTrainCount) for b in SIMDBenchmarks.values()]
	)
	plotBuild(
		cfg.dim, getCzechMetric(space), plotsDir,
		f"bestSIMD_sequential_vs_noSIMD_parallel_build_{getEnglishMetric(space)}",
		bestSIMDBenchmarks.getSeqStats(),
		*noSIMDBenchmarks.getParStats()
	)
	plotRecall(
		cfg.dim, getCzechMetric(space), cfg.maxTrainCount, plotsDir,
		f"bestSIMD_sequential_vs_noSIMD_parallel_recall_{getEnglishMetric(space)}",
		bestSIMDBenchmarks.getSeqStats(cfg.maxTrainCount),
		*noSIMDBenchmarks.getParStats(cfg.maxTrainCount)
	)
	bestSIMDBenchmarks.plotBuild(plotsDir, f"best_simd_build_{getEnglishMetric(space)}")
	bestSIMDBenchmarks.plotRecall(cfg.maxTrainCount, plotsDir, f"best_simd_recall_{getEnglishMetric(space)}")

def main():
	run(Config(
		dim=25, efConstruction=200, efSearchValues=[10, 20, 40, 80, 120, 200, 400, 600],
		mMax=16, runs=1, trainCounts=[100, 500, 1000, 2000], workerCounts=[1, 2, 3, 4]
	))

if __name__ == "__main__":
	main()

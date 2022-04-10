from dataclasses import dataclass, field
from functools import cached_property
import parallel_hnsw as h
from pathlib import Path
from matplotlib import pyplot as plt

BACK_SLASH = "\\"
N = "\n"
TAB = "\t"

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

	def getTupleStr(self):
		return f"({self.x}, {self.y})"

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

@dataclass
class GroupPlot:
	lines: list[Line]
	metric: str

	def getLatex(self):
		res = f"{BACK_SLASH}nextgroupplot[title={self.metric}]{N}"

		for line in self.lines:
			res += f"{BACK_SLASH}addplot coordinates {{{N}% {line.name}{N}"

			for p in line.points:
				res += f"{TAB}{p.getTupleStr()}{N}"

			res += f"}};{N}"

		return res

	def getLegend(self):
		return ",".join([l.name for l in self.lines])

@dataclass
class GroupPlotPair:
	build: GroupPlot = None
	recall: GroupPlot = None

@dataclass
class Plot:
	bestDirection: str
	caption: str
	groups: list[GroupPlot]
	plotLabel: str
	xLabel: str
	yLabel: str
	shortCaption: str = None
	yModeLog: bool = False

	def getLatex(self, template: str):
		return template.replace("@GROUP_SIZE@", "2 by 1"
		).replace("@YMODE@", f"{N}ymode = log," if self.yModeLog else ""
		).replace("@XLABEL@", self.xLabel
		).replace("@YLABEL@", self.yLabel
		).replace("@GROUP_PLOTS@", "\n\n".join([g.getLatex() for g in self.groups])
		).replace("@LEGEND@", self.groups[0].getLegend()
		).replace("@SHORT_CAPTION@", f"[{self.shortCaption}]" if self.shortCaption is not None else ""
		).replace("@LONG_CAPTION@", self.caption
		).replace("@BEST_DIRECTION@", self.bestDirection
		).replace("@LABEL@", self.plotLabel)

	def writeLatex(self, plotsDir: Path, template: str):
		with (plotsDir / f"{self.plotLabel}.tex").open("w", encoding="utf-8") as f:
			f.write(self.getLatex(template))

@dataclass
class SpaceGroupPlots:
	bestSIMD: GroupPlotPair = field(default_factory=GroupPlotPair)
	noSIMD: GroupPlotPair = field(default_factory=GroupPlotPair)
	SIMDadv: GroupPlotPair = field(default_factory=GroupPlotPair)
	SIMDseq: GroupPlotPair = field(default_factory=GroupPlotPair)

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
	return GroupPlot(lines, metric)

def plotBuild(dim: int, metric: str, plotsDir: Path, name: str, *statsCol: list[Stats]):
	return plot(
		[getBuildLine(s) for s in statsCol], dim=dim, metric=metric,
		xLabel="Počet prvků při stavbě", yLabel="Čas stavby (s)",
		plotsDir=plotsDir, name=name
	)

def plotRecall(dim: int, metric: str, trainCount: int, plotsDir: Path, name: str, *stats: Stats):
	return plot(
		[s.getRecallLine() for s in stats], dim=dim, metric=metric,
		xLabel="Přesnost", yLabel="Počet dotazů za sekundu (1/s)",
		plotsDir=plotsDir, name=name, trainCount=trainCount
	)

def writeLatexPlots(
	angularPlots: SpaceGroupPlots, euclideanPlots: SpaceGroupPlots,
	plotsDir: Path, template: str
):
	buildPlot = Plot(
		bestDirection="Lepší výsledky směrem k dolnímu okraji grafu.",
		caption="Závislost času stavby na počtu prvků při stavbě.",
		groups=[euclideanPlots.noSIMD.build, angularPlots.noSIMD.build],
		plotLabel="NoSIMDBuild", xLabel="Počet prvků při stavbě",
		yLabel="Čas stavby (s)"
	)
	buildPlot.writeLatex(plotsDir, template)

	buildPlot.groups=[euclideanPlots.SIMDseq.build, angularPlots.SIMDseq.build]
	buildPlot.plotLabel = "SIMDseqBuild"
	buildPlot.writeLatex(plotsDir, template)

	buildPlot.groups=[euclideanPlots.SIMDadv.build, angularPlots.SIMDadv.build]
	buildPlot.plotLabel = "SIMDadvBuild"
	buildPlot.writeLatex(plotsDir, template)

	buildPlot.groups=[euclideanPlots.bestSIMD.build, angularPlots.bestSIMD.build]
	buildPlot.plotLabel = "bestSIMDBuild"
	buildPlot.writeLatex(plotsDir, template)

	recallPlot = Plot(
		bestDirection="Lepší výsledky směrem k pravému hornímu rohu grafu.",
		caption="Závislost počtu zpracovaných dotazů za sekundu na přesnosti.",
		groups=[euclideanPlots.noSIMD.recall, angularPlots.noSIMD.recall],
		plotLabel="NoSIMDRecall", xLabel="Přesnost",
		yLabel=r"Počet dotazů za sekundu $(\frac{1}{s})$"
	)
	recallPlot.writeLatex(plotsDir, template)

	recallPlot.groups=[euclideanPlots.SIMDseq.recall, angularPlots.SIMDseq.recall]
	recallPlot.plotLabel = "SIMDseqRecall"
	recallPlot.writeLatex(plotsDir, template)

	recallPlot.groups=[euclideanPlots.SIMDadv.recall, angularPlots.SIMDadv.recall]
	recallPlot.plotLabel = "SIMDadvRecall"
	recallPlot.writeLatex(plotsDir, template)

	recallPlot.groups=[euclideanPlots.bestSIMD.recall, angularPlots.bestSIMD.recall]
	recallPlot.plotLabel = "bestSIMDRecall"
	recallPlot.writeLatex(plotsDir, template)

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
		return plotBuild(
			self.dim, getCzechMetric(self.space), plotsDir, name,
			self.getSeqStats(), *self.getParStats(),
		)

	def plotRecall(self, trainCount: int, plotsDir: Path, name: str):
		return plotRecall(
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
	src = Path(__file__).parents[1]
	plotsDir = src / "plots"
	plotsDir.mkdir(exist_ok=True)
	euclideanPlots = runForSpace(cfg, h.Space.EUCLIDEAN, plotsDir)
	angularPlots = runForSpace(cfg, h.Space.ANGULAR, plotsDir)
	writeLatexPlots(
		angularPlots, euclideanPlots, plotsDir,
		(src / "templates" / "groupPlots.txt").read_text(encoding="utf-8")
	)

def runForSpace(cfg: Config, space: h.Space, plotsDir: Path):
	res = SpaceGroupPlots()
	noSIMDBenchmarks = getBenchmarkList(cfg, h.SIMDType.NONE, space)
	res.noSIMD.build = noSIMDBenchmarks.plotBuild(plotsDir, f"no_simd_build_{getEnglishMetric(space)}")
	res.noSIMD.recall = noSIMDBenchmarks.plotRecall(
		cfg.maxTrainCount, plotsDir,
		f"no_simd_recall_{getEnglishMetric(space)}"
	)

	availableSIMD = getAvailableSIMD()

	if not availableSIMD:
		return res

	SIMDBenchmarks = {SIMD: getBenchmarkList(cfg, SIMD, space) for SIMD in availableSIMD}
	bestSIMDBenchmarks = SIMDBenchmarks[h.getBestSIMDType()]

	res.SIMDseq.build = plotBuild(
		cfg.dim, getCzechMetric(space), plotsDir, f"simd_sequential_build_{getEnglishMetric(space)}",
		noSIMDBenchmarks.getSeqStats(),
		*[b.getSeqStats() for b in SIMDBenchmarks.values()]
	)
	res.SIMDseq.recall = plotRecall(
		cfg.dim, getCzechMetric(space), cfg.maxTrainCount, plotsDir,
		f"simd_sequential_recall_{getEnglishMetric(space)}",
		noSIMDBenchmarks.getSeqStats(cfg.maxTrainCount),
		*[b.getSeqStats(cfg.maxTrainCount) for b in SIMDBenchmarks.values()]
	)
	res.SIMDadv.build = plotBuild(
		cfg.dim, getCzechMetric(space), plotsDir,
		f"bestSIMD_sequential_vs_noSIMD_parallel_build_{getEnglishMetric(space)}",
		bestSIMDBenchmarks.getSeqStats(),
		*noSIMDBenchmarks.getParStats()
	)
	res.SIMDadv.recall = plotRecall(
		cfg.dim, getCzechMetric(space), cfg.maxTrainCount, plotsDir,
		f"bestSIMD_sequential_vs_noSIMD_parallel_recall_{getEnglishMetric(space)}",
		bestSIMDBenchmarks.getSeqStats(cfg.maxTrainCount),
		*noSIMDBenchmarks.getParStats(cfg.maxTrainCount)
	)
	res.bestSIMD.build = bestSIMDBenchmarks.plotBuild(
		plotsDir, f"best_simd_build_{getEnglishMetric(space)}"
	)
	res.bestSIMD.recall = bestSIMDBenchmarks.plotRecall(
		cfg.maxTrainCount, plotsDir, f"best_simd_recall_{getEnglishMetric(space)}"
	)
	return res

def main():
	run(Config(
		dim=25, efConstruction=200, efSearchValues=[10, 20, 40, 80, 120, 200, 400, 600],
		mMax=16, runs=1, trainCounts=[100, 500, 1000, 2000], workerCounts=[1, 2, 3, 4]
	))

if __name__ == "__main__":
	main()

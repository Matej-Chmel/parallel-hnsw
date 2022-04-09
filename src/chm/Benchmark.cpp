#include <numeric>
#include <sstream>
#include <stdexcept>
#include "Benchmark.hpp"

namespace chm {
	constexpr std::streamsize EF_SEARCH_WIDTH = 8;
	constexpr std::streamsize ELAPSED_PRETTY_WIDTH = 29;
	constexpr std::streamsize RECALL_WIDTH = 12;

	BenchmarkStats::BenchmarkStats(
		const chr::nanoseconds& avg, const chr::nanoseconds& max, const chr::nanoseconds& min
	) : avg(avg), max(max), min(min) {}

	QueryBenchmark::QueryBenchmark(const chr::nanoseconds& elapsed, const float recall)
		: elapsed(elapsed), recall(recall) {}

	QueryBenchmarkStats::QueryBenchmarkStats(
		const chr::nanoseconds& avg, const chr::nanoseconds& max, const chr::nanoseconds& min,
		const float avgRecall, const float maxRecall, const float minRecall
	) : BenchmarkStats(avg, max, min), avgRecall(avgRecall), maxRecall(maxRecall),
		minRecall(minRecall) {}

	Benchmark::Benchmark(
		const DatasetPtr& dataset, const uint efConstruction,
		const std::vector<uint>& efSearchValues, const uint k, const uint levelGenSeed,
		const uint mMax, const bool parallel, const size_t runsCount, const size_t workerCount
	) : cfg(efConstruction, mMax, dataset->trainCount), dataset(dataset), k(k), indexStr(""),
		levelGenSeed(levelGenSeed), parallel(parallel), runsCount(runsCount), workerCount(workerCount) {

		for(const auto& efSearch : efSearchValues)
			this->efsToBenchmarks[efSearch] = std::vector<QueryBenchmark>();
	}

	BenchmarkStats Benchmark::getBuildStats() const {
		return BenchmarkStats(
			std::accumulate(
				this->buildElapsed.begin(), this->buildElapsed.end(), chr::nanoseconds(0)
			) / this->buildElapsed.size(),
			*std::max_element(this->buildElapsed.begin(), this->buildElapsed.end()),
			*std::min_element(this->buildElapsed.begin(), this->buildElapsed.end())
		);
	}

	Benchmark Benchmark::getParallel(const size_t workerCount) const {
		std::vector<uint> efSearchValues;

		for(const auto& p : this->efsToBenchmarks)
			efSearchValues.emplace_back(p.first);

		return Benchmark(
			this->dataset, this->cfg.efConstruction, efSearchValues, this->k, this->levelGenSeed,
			this->cfg.mMax, true, this->runsCount, workerCount
		);
	}

	std::string Benchmark::getString() const {
		std::stringstream s;
		s << this->dataset->getString() << '\n' << this->indexStr << '\n' <<
			"Runs: " << this->runsCount;
		return s.str();
	}

	std::map<uint, QueryBenchmarkStats> Benchmark::getQueryStats() const {
		std::map<uint, QueryBenchmarkStats> res;

		for(const auto& p : this->efsToBenchmarks) {
			chr::nanoseconds sumElapsed(0);
			float sumRecall(0);

			for(const auto& q : p.second) {
				sumElapsed += q.elapsed;
				sumRecall += q.recall;
			}

			res[p.first] = QueryBenchmarkStats(
				sumElapsed / p.second.size(),
				std::max_element(p.second.begin(), p.second.end(), MaxQueryElapsedCmp())->elapsed,
				std::min_element(p.second.begin(), p.second.end(), MinQueryElapsedCmp())->elapsed,
				sumRecall / p.second.size(),
				std::max_element(p.second.begin(), p.second.end(), MaxQueryRecallCmp())->recall,
				std::min_element(p.second.begin(), p.second.end(), MinQueryRecallCmp())->recall
			);
		}

		return res;
	}

	void Benchmark::print(std::ostream& s) const {
		std::ios streamState(nullptr);
		streamState.copyfmt(s);

		s << this->getString() << '\n';

		const auto buildStats = this->getBuildStats();
		s << "Build avg: ";
		prettyPrint(buildStats.avg, s);
		s << "\nBuild best: ";
		prettyPrint(buildStats.min, s);
		s << "\nBuild worst: ";
		prettyPrint(buildStats.max, s);
		s << "\n\n";

		printField("EfSearch", s, EF_SEARCH_WIDTH);
		printField("Avg. recall", s, RECALL_WIDTH);
		printField("Max. recall", s, RECALL_WIDTH);
		printField("Min. recall", s, RECALL_WIDTH);
		printField("Avg. elapsed", s, ELAPSED_PRETTY_WIDTH);
		printField("Max. elapsed", s, ELAPSED_PRETTY_WIDTH);
		printField("Min. elapsed", s, ELAPSED_PRETTY_WIDTH);
		printField("\n", s, 1);

		for(const auto& p : this->getQueryStats()) {
			printField(p.first, s, EF_SEARCH_WIDTH);

			s << std::right << std::setw(RECALL_WIDTH);
			chm::print(p.second.avgRecall, s, 3);
			s << std::right << std::setw(RECALL_WIDTH);
			chm::print(p.second.maxRecall, s, 3);
			s << std::right << std::setw(RECALL_WIDTH);
			chm::print(p.second.minRecall, s, 3);

			s << std::right << std::setw(ELAPSED_PRETTY_WIDTH);
			prettyPrint(p.second.avg, s);
			s << std::right << std::setw(ELAPSED_PRETTY_WIDTH);
			prettyPrint(p.second.max, s);
			s << std::right << std::setw(ELAPSED_PRETTY_WIDTH);
			prettyPrint(p.second.min, s);
			printField("\n", s, 1);
		}

		s.copyfmt(streamState);
	}

	Benchmark& Benchmark::run() {
		Timer timer{};

		for(size_t buildIdx = 0; buildIdx < this->runsCount; buildIdx++) {
			timer.reset();
			auto index = this->dataset->getIndex(
				this->cfg.efConstruction, this->cfg.mMax, this->parallel,
				this->levelGenSeed, this->workerCount
			);
			this->dataset->build(index);
			this->buildElapsed.push_back(timer.getElapsed());

			const auto indexStr = index->getString();

			if(indexStr != this->indexStr) {
				if(this->indexStr == "")
					this->indexStr = indexStr;
				else
					throw std::runtime_error("Index string changed between runs.");
			}

			for(size_t queryIdx = 0; queryIdx < this->runsCount; queryIdx++)
				for(auto& p : this->efsToBenchmarks) {
					timer.reset();
					auto queryRes = this->dataset->query(index, p.first);
					const auto elapsed = timer.getElapsed();
					p.second.emplace_back(elapsed, this->dataset->getRecall(queryRes->getIDs()));
				}
		}

		return *this;
	}

	void prettyPrint(const chr::nanoseconds& elapsed, std::ostream& s) {
		chr::nanoseconds elapsedCopy = elapsed;
		std::stringstream strStream;

		print(convert<chr::hours>(elapsedCopy), strStream);
		strStream << ':';
		print(convert<chr::minutes>(elapsedCopy), strStream);
		strStream << ':';
		print(convert<chr::seconds>(elapsedCopy), strStream);
		strStream << '.';
		print(convert<chr::milliseconds>(elapsedCopy), strStream, 3);
		strStream << '.';
		print(convert<chr::microseconds>(elapsedCopy), strStream, 3);
		strStream << '.';
		print(elapsedCopy.count(), strStream, 3);

		s << strStream.str();
	}

	void print(const float number, std::ostream& s, const std::streamsize places) {
		std::ios streamState(nullptr);
		streamState.copyfmt(s);
		s << std::fixed << std::showpoint << std::setprecision(places) << number;
		s.copyfmt(streamState);
	}

	void print(const long long number, std::ostream& s, const std::streamsize places) {
		s << std::setfill('0') << std::setw(places) << number;
	}
}

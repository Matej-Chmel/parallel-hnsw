#include <sstream>
#include "RecallTable.hpp"

namespace chm {
	constexpr std::streamsize EF_SEARCH_WIDTH = 8;
	constexpr std::streamsize ELAPSED_PRETTY_WIDTH = 29;
	constexpr std::streamsize ELAPSED_WIDTH = 29;
	constexpr std::streamsize RECALL_WIDTH = 12;

	long long QueryBenchmark::getElapsedNum() const {
		return this->elapsed.count();
	}

	float QueryBenchmark::getRecall() const {
		return this->recall;
	}

	void QueryBenchmark::prettyPrintElapsed(std::ostream& s) const {
		std::stringstream strStream;
		prettyPrint(this->elapsed, strStream);
		s << strStream.str();
	}

	QueryBenchmark::QueryBenchmark(const uint efSearch) : efSearch(efSearch), elapsed(0), recall(0.f) {}

	void QueryBenchmark::setElapsed(const chr::nanoseconds& elapsed) {
		this->elapsed = elapsed;
	}

	void QueryBenchmark::setRecall(const float recall) {
		this->recall = recall;
	}

	RecallTableConfig RecallTableConfig::getOpposite() const {
		return RecallTableConfig(
			this->dataset,
			this->efConstruction,
			this->efSearchValues,
			this->mMax,
			!this->parallel,
			this->seed,
			this->workersNum
		);
	}

	RecallTableConfig::RecallTableConfig(
		const DatasetPtr& dataset, const uint efConstruction,
		const std::vector<uint>& efSearchValues, const uint mMax,
		const bool parallel, const uint seed, const uint workersNum
	) : dataset(dataset), efConstruction(efConstruction), efSearchValues(efSearchValues), mMax(mMax),
		parallel(parallel), seed(seed), workersNum(workersNum) {}

	RecallTableConfig::RecallTableConfig(
		const fs::path& datasetPath, const uint efConstruction,
		const std::vector<uint>& efSearchValues, const uint mMax,
		const bool parallel, const uint seed, const uint workersNum
	) : dataset(std::make_shared<Dataset>(datasetPath)), efConstruction(efConstruction),
		efSearchValues(efSearchValues), mMax(mMax), parallel(parallel), seed(seed),
		workersNum(workersNum) {}

	RecallTable::RecallTable(const RecallTableConfig& cfg) : buildElapsed(0), cfg(cfg), indexStr("") {}

	void RecallTable::print(std::ostream& s) const {
		std::ios streamState(nullptr);
		streamState.copyfmt(s);

		this->cfg.dataset->writeShortDescription(s);
		s << this->indexStr << '\n';
		s << "Build time: [";
		prettyPrint(this->buildElapsed, s);
		s << "], " << this->buildElapsed.count() << " ns\n\n";

		printField("EfSearch", s, EF_SEARCH_WIDTH);
		printField("Recall", s, RECALL_WIDTH);
		printField("Elapsed (pretty)", s, ELAPSED_PRETTY_WIDTH);
		printField("Elapsed (ns)", s, ELAPSED_WIDTH);
		printField("\n", s, 1);

		for(const auto& benchmark : this->benchmarks) {
			printField(benchmark.efSearch, s, EF_SEARCH_WIDTH);
			s << std::right << std::setw(RECALL_WIDTH);
			chm::print(benchmark.getRecall(), s, 3);
			s << std::right << std::setw(ELAPSED_PRETTY_WIDTH);
			benchmark.prettyPrintElapsed(s);
			printField(benchmark.getElapsedNum(), s, ELAPSED_WIDTH);
			printField("\n", s, 1);
		}

		s.copyfmt(streamState);
	}

	void RecallTable::run(std::ostream& s) {
		Timer timer{};
		this->benchmarks.clear();
		this->benchmarks.reserve(this->cfg.efSearchValues.size());

		s << "Building index.\n";

		timer.reset();
		IndexPtr index = this->cfg.dataset->getIndex(
			cfg.efConstruction, cfg.mMax, cfg.parallel, cfg.seed, cfg.workersNum
		);
		this->cfg.dataset->build(index);
		this->buildElapsed = timer.getElapsed();
		this->indexStr = index->getString();

		s << "Index built in ";
		prettyPrint(this->buildElapsed, s);
		s << "\n\n";

		for(const auto& efSearch : this->cfg.efSearchValues) {
			auto& benchmark = this->benchmarks.emplace_back(efSearch);
			s << "Querying with efSearch = " << efSearch << ".\n";

			timer.reset();
			const auto queryRes = this->cfg.dataset->query(index, efSearch);
			benchmark.setElapsed(timer.getElapsed());

			s << "Completed in ";
			benchmark.prettyPrintElapsed(s);
			s << "\nComputing recall.\n";

			timer.reset();
			benchmark.setRecall(this->cfg.dataset->getRecall(queryRes));
			const auto recallElapsed = timer.getElapsed();

			s << "Recall " << benchmark.getRecall() << " computed in ";
			prettyPrint(recallElapsed, s);
			s << "\n\n";
		}
	}

	chr::nanoseconds Timer::getElapsed() const {
		return chr::duration_cast<chr::nanoseconds>(chr::steady_clock::now() - this->start);
	}

	void Timer::reset() {
		this->start = chr::steady_clock::now();
	}

	Timer::Timer() {
		this->reset();
	}

	void prettyPrint(const chr::nanoseconds& elapsed, std::ostream& s) {
		chr::nanoseconds elapsedCopy = elapsed;
		std::ios streamState(nullptr);
		streamState.copyfmt(s);

		print(convert<chr::hours>(elapsedCopy), s);
		s << ':';
		print(convert<chr::minutes>(elapsedCopy), s);
		s << ':';
		print(convert<chr::seconds>(elapsedCopy), s);
		s << '.';
		print(convert<chr::milliseconds>(elapsedCopy), s, 3);
		s << '.';
		print(convert<chr::microseconds>(elapsedCopy), s, 3);
		s << '.';
		print(elapsedCopy.count(), s, 3);

		s.copyfmt(streamState);
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

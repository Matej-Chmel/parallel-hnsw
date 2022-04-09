#pragma once
#include <iomanip>
#include <ios>
#include <ostream>
#include <map>
#include "Dataset.hpp"

namespace chm {
	struct BenchmarkStats {
		chr::nanoseconds avg;
		chr::nanoseconds max;
		chr::nanoseconds min;

		BenchmarkStats(
			const chr::nanoseconds& avg, const chr::nanoseconds& max, const chr::nanoseconds& min
		);
	};

	struct QueryBenchmark {
		chr::nanoseconds elapsed;
		float recall;

		QueryBenchmark(const chr::nanoseconds& elapsed, const float recall);
	};

	struct QueryBenchmarkStats : public BenchmarkStats {
		float avgRecall;
		float maxRecall;
		float minRecall;

		QueryBenchmarkStats(
			const chr::nanoseconds& avg, const chr::nanoseconds& max, const chr::nanoseconds& min,
			const float avgRecall, const float maxRecall, const float minRecall
		);
	};

	struct MaxQueryElapsedCmp {
		constexpr bool operator()(const QueryBenchmark& a, const QueryBenchmark& b) {
			return a.elapsed > b.elapsed;
		}
	};

	struct MinQueryElapsedCmp {
		constexpr bool operator()(const QueryBenchmark& a, const QueryBenchmark& b) {
			return a.elapsed < b.elapsed;
		}
	};

	struct MaxQueryRecallCmp {
		constexpr bool operator()(const QueryBenchmark& a, const QueryBenchmark& b) {
			return a.recall > b.recall;
		}
	};

	struct MinQueryRecallCmp {
		constexpr bool operator()(const QueryBenchmark& a, const QueryBenchmark& b) {
			return a.recall < b.recall;
		}
	};

	class Benchmark {
		std::vector<chr::nanoseconds> buildElapsed;
		const IndexConfig cfg;
		const DatasetPtr dataset;
		std::map<uint, std::vector<QueryBenchmark>> efsToBenchmarks;
		std::string indexStr;
		const uint k;
		const uint levelGenSeed;
		const bool parallel;
		const size_t runsCount;
		const size_t workerCount;

		float getRecall(const uint efSearch) const;

	public:
		Benchmark(
			const DatasetPtr& dataset, const uint efConstruction,
			const std::vector<uint>& efSearchValues, const uint k, const uint levelGenSeed,
			const uint mMax, const bool parallel, const size_t runsCount, const size_t workerCount
		);
		BenchmarkStats getBuildStats() const;
		Benchmark getParallel(const size_t workerCount) const;
		std::string getString() const;
		std::map<uint, QueryBenchmarkStats> getQueryStats() const;
		void print(std::ostream& s) const;
		Benchmark& run();
	};

	template<typename T> long long convert(chr::nanoseconds& t);
	void prettyPrint(const chr::nanoseconds& elapsed, std::ostream& s);
	void print(const float number, std::ostream& s, const std::streamsize places = 2);
	void print(const long long number, std::ostream& s, const std::streamsize places = 2);
	template<typename T> void printField(const T& field, std::ostream& s, const std::streamsize width);

	template<typename T>
	inline long long convert(chr::nanoseconds& t) {
		const auto res = chr::duration_cast<T>(t);
		t -= res;
		return res.count();
	}

	template<typename T>
	inline void printField(const T& field, std::ostream& s, const std::streamsize width) {
		s << std::right << std::setw(width) << field;
	}
}

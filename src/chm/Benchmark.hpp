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

		BenchmarkStats() = default;
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

		QueryBenchmarkStats() = default;
		QueryBenchmarkStats(
			const chr::nanoseconds& avg, const chr::nanoseconds& max, const chr::nanoseconds& min,
			const float avgRecall, const float maxRecall, const float minRecall
		);
	};

	class Benchmark {
		std::vector<chr::nanoseconds> buildElapsed;
		const IndexConfig cfg;
		const DatasetPtr dataset;
		std::map<uint, std::vector<QueryBenchmark>> efsToBenchmarks;
		std::string indexStr;
		const uint levelGenSeed;
		const bool parallel;
		const size_t runsCount;
		const size_t workerCount;

	public:
		Benchmark(
			const DatasetPtr& dataset, const uint efConstruction,
			const std::vector<uint>& efSearchValues, const uint levelGenSeed,
			const uint mMax, const bool parallel, const size_t runsCount, const size_t workerCount = 1
		);
		BenchmarkStats getBuildStats() const;
		Benchmark getParallel(const size_t workerCount) const;
		std::string getString() const;
		std::map<uint, QueryBenchmarkStats> getQueryStats() const;
		void print(std::ostream& s) const;
		Benchmark& run(std::ostream& s);
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

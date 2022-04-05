#pragma once
#include <chrono>
#include "Dataset.hpp"

namespace chm {
	namespace chr = std::chrono;

	class QueryBenchmark {
		chr::nanoseconds elapsed;
		float recall;

	public:
		const uint efSearch;

		long long getElapsedNum() const;
		float getRecall() const;
		void prettyPrintElapsed(std::ostream& s) const;
		QueryBenchmark(const uint efSearch);
		void setElapsed(const chr::nanoseconds& elapsed);
		void setRecall(const float recall);
	};

	struct RecallTableConfig {
		const DatasetPtr dataset;
		const uint efConstruction;
		const std::vector<uint>& efSearchValues;
		const uint mMax;
		const bool parallel;
		const uint seed;
		const uint workersNum;

		RecallTableConfig getOpposite() const;
		RecallTableConfig(
			const DatasetPtr& dataset, const uint efConstruction,
			const std::vector<uint>& efSearchValues, const uint mMax,
			const bool parallel, const uint seed, const uint workersNum
		);
		RecallTableConfig(
			const fs::path& datasetPath, const uint efConstruction,
			const std::vector<uint>& efSearchValues, const uint mMax,
			const bool parallel, const uint seed, const uint workersNum
		);
	};

	class RecallTable {
		std::vector<QueryBenchmark> benchmarks;
		chr::nanoseconds buildElapsed;
		const RecallTableConfig cfg;
		std::string indexStr;

	public:
		RecallTable(const RecallTableConfig& cfg);
		void print(std::ostream& s) const;
		void run(std::ostream& s);
	};

	class Timer {
		chr::steady_clock::time_point start;

	public:
		chr::nanoseconds getElapsed() const;
		void reset();
		Timer();
	};

	void prettyPrint(const chr::nanoseconds& elapsed, std::ostream& s);
	void print(const float number, std::ostream& s, const std::streamsize places = 2);
	void print(const long long number, std::ostream& s, const std::streamsize places = 2);

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

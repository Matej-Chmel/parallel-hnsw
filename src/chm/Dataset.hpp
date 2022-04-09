#pragma once
#include <chrono>
#include "Index.hpp"

namespace chm {
	namespace chr = std::chrono;

	class BruteforceIndex {
		uint elemCount;
		std::vector<Node> nodes;
		Space space;

		void queryOne(const Element& e, const size_t k, const QueryResPtr& res);

	public:
		BruteforceIndex(
			const size_t dim, const size_t maxElemCount,
			const SIMDType simdType, const SpaceKind spaceKind
		);
		void push(const ArrayView<const float>& v);
		QueryResPtr queryBatch(const ArrayView<const float>& v, const size_t k);
	};

	class Dataset {
		chr::nanoseconds bruteforceElapsed;
		std::vector<uint> neighbors;
		std::vector<float> test;
		std::vector<float> train;

		void generate(std::vector<float>& v, const size_t count, const uint seed);

	public:
		const size_t dim;
		const size_t k;
		const SIMDType simdType;
		const SpaceKind spaceKind;
		const size_t testCount;
		const size_t trainCount;

		void build(const IndexPtr& index) const;
		Dataset(
			const size_t dim, const size_t k, const uint seed, const SpaceKind spaceKind,
			const SIMDType simdType, const size_t testCount, const size_t trainCount
		);
		chr::nanoseconds getBruteforceElapsed() const;
		IndexPtr getIndex(
			const uint efConstruction, const uint mMax, const bool parallel,
			const uint seed, const size_t workerCount
		) const;
		float getRecall(const ArrayView<const uint>& foundIDs) const;
		std::string getString() const;
		QueryResPtr query(const IndexPtr& index, const uint efSearch) const;
	};

	using DatasetPtr = std::shared_ptr<const Dataset>;

	class Timer {
		chr::steady_clock::time_point start;

	public:
		chr::nanoseconds getElapsed() const;
		void reset();
		Timer();
	};
}

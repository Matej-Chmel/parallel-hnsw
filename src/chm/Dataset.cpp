#include <sstream>
#include <stdexcept>
#include "Dataset.hpp"

namespace chm {
	void BruteforceIndex::queryOne(const Element& e, const size_t k, const QueryResPtr& res) {
		for(auto& n : this->nodes)
			n.dist = space.getDistance(e.data, n.id);

		std::sort(this->nodes.begin(), this->nodes.end(), FarHeapCmp());

		for(size_t i = 0; i < k; i++)
			res->set(e.id, i, this->nodes[i].dist, this->nodes[i].id);
	}

	BruteforceIndex::BruteforceIndex(
		const size_t dim, const size_t maxElemCount, const SIMDType simdType, const SpaceKind spaceKind
	) : elemCount(0), space(dim, spaceKind, uint(maxElemCount), simdType) {

		this->nodes.reserve(maxElemCount);
	}

	QueryResPtr BruteforceIndex::queryBatch(const ArrayView<const float>& v, const size_t k) {
		const auto queryCount = v.getElemCount();
		auto res = std::make_shared<QueryResults>(k, queryCount);

		for(size_t i = 0; i < queryCount; i++)
			this->queryOne(Element(v.getData(i), uint(i)), k, res);

		return res;
	}

	void BruteforceIndex::push(const ArrayView<const float>& v) {
		const auto elemCount = v.getElemCount();

		for(size_t i = 0; i < elemCount; i++) {
			const auto id = this->elemCount + uint(i);
			this->space.push(Element(v.getData(i), id));
			this->nodes.emplace_back(0.f, id);
		}
	}

	void Dataset::generate(std::vector<float>& v, const size_t count, const uint seed) {
		const auto componentCount = this->dim * count;
		std::uniform_real_distribution<float> dist{};
		std::default_random_engine gen(seed);

		v.reserve(componentCount);

		for(uint i = 0; i < componentCount; i++)
			v.emplace_back(dist(gen));
	}

	void Dataset::build(const IndexPtr& index) const {
		index->push(ArrayView<const float>(this->train.data(), this->dim, this->trainCount));
	}

	Dataset::Dataset(
		const size_t dim, const size_t k, const uint seed, const SpaceKind spaceKind,
		const SIMDType simdType, const size_t testCount, const size_t trainCount
	) : dim(dim), k(k), simdType(simdType), spaceKind(spaceKind), testCount(testCount),
		trainCount(trainCount) {

		this->generate(this->test, testCount, seed + 1);
		this->generate(this->train, trainCount, seed);

		Timer timer{};
		BruteforceIndex bf(this->dim, this->trainCount, simdType, this->spaceKind);
		bf.push(ArrayView<const float>(this->train.data(), this->dim, this->trainCount));
		const auto res = bf.queryBatch(
			ArrayView<const float>(this->test.data(), this->dim, this->testCount), this->k
		);
		this->bruteforceElapsed = timer.getElapsed();
		res->copyIDsTo(this->neighbors);
	}

	IndexPtr Dataset::getIndex(
		const uint efConstruction, const uint mMax, const bool parallel,
		const uint seed, const size_t workerCount
	) const {
		const IndexConfig cfg(efConstruction, mMax, uint(this->trainCount));
		if(parallel) {
			auto res = std::make_shared<ParallelIndex>(
				cfg, this->dim, seed, this->spaceKind, this->simdType
			);
			res->setWorkersNum(workerCount);
			return res;
		}
		return std::make_shared<SequentialIndex>(cfg, this->dim, seed, this->spaceKind, this->simdType);
	}

	chr::nanoseconds Dataset::getBruteforceElapsed() const {
		return this->bruteforceElapsed;
	}

	float Dataset::getRecall(const ArrayView<const uint>& foundIDs) const {
		if(!this->k)
			throw std::runtime_error("Bruteforce wasn't computed.");

		return chm::getRecall(
			ArrayView<const uint>(this->neighbors.data(), this->k, this->testCount), foundIDs
		);
	}

	std::string Dataset::getString() const {
		std::stringstream s;
		s << "Dataset: " << spaceKindToStr(this->spaceKind)
			<< " space, dimension = " << this->dim << ", trainCount = " << this->trainCount
			<< ", testCount = " << this->testCount << ", k = " << this->k;
		return s.str();
	}

	QueryResPtr Dataset::query(const IndexPtr& index, const uint efSearch) const {
		return index->queryBatch(
			ArrayView<const float>(this->test.data(), this->dim, this->testCount), efSearch, uint(this->k)
		);
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
}

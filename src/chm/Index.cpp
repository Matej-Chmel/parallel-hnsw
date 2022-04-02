#include <cmath>
#include <stdexcept>
#include "Index.hpp"

namespace chm {
	Node::Node() : dist(0.f), id(0) {};
	Node::Node(const float dist, const uint id) : dist(dist), id(id) {};

	std::vector<uint>::iterator NeighborsCopy::begin() {
		return this->n.begin();
	}

	void NeighborsCopy::clear() {
		throw std::runtime_error("NeighborsCopy object is read-only.");
	}

	std::vector<uint>::iterator NeighborsCopy::end() {
		return this->n.end();
	}

	uint NeighborsCopy::len() const {
		return uint(this->n.size());
	}

	NeighborsCopy::NeighborsCopy(const std::vector<uint>::iterator& lenIter) {
		auto beginIter = lenIter + 1;
		const auto len = *lenIter;
		this->n.reserve(len);

		for(size_t i = 0; i < len; i++)
			this->n.push_back(*(beginIter + i));
	}

	void NeighborsCopy::push(const uint id) {
		throw std::runtime_error("NeighborsCopy object is read-only.");
	}

	std::vector<uint>::iterator NeighborsView::begin() {
		return this->lenIter + 1;
	}

	void NeighborsView::clear() {
		*this->lenIter = 0;
	}

	std::vector<uint>::iterator NeighborsView::end() {
		return this->lenIter + 1 + *this->lenIter;
	}

	uint NeighborsView::len() const {
		return *this->lenIter;
	}

	NeighborsView::NeighborsView(const std::vector<uint>::iterator& lenIter) : lenIter(lenIter) {};

	void NeighborsView::push(const uint id) {
		*this->end() = id;
		(*this->lenIter)++;
	}

	std::vector<uint>::iterator Connections::getLenIter(const uint id, const uint lc) {
		return lc
			? this->upperLayers[id].begin() + this->maxLen * (lc - 1)
			: this->layer0.begin() + this->maxLen0 * id;
	}

	Connections::Connections(
		const uint maxElemCount, const uint mMax, const uint mMax0
	) : maxLen(mMax + 1), maxLen0(mMax0 + 1), upperLayers(maxElemCount) {

		this->layer0.resize(maxElemCount * this->maxLen0, 0);
		this->upperLayers.resize(maxElemCount);
	}

	NeighborsPtr Connections::getNeighbors(const uint id, const uint lc) {
		return std::make_shared<NeighborsView>(this->getLenIter(id, lc));
	}

	void Connections::init(const uint id, const uint level) {
		if(level)
			this->upperLayers[id].resize(this->maxLen * level, 0);
	}

	std::mutex& ThreadSafeConnections::getMutex(const uint id) {
		return this->mutexes[id];
	}

	NeighborsPtr ThreadSafeConnections::getNeighbors(const uint id, const uint lc) {
		std::unique_lock lock(this->getMutex(id));
		return std::make_shared<NeighborsCopy>(this->getLenIter(id, lc));
	}

	NeighborsView ThreadSafeConnections::getWritableNeighbors(const uint id, const uint lc) {
		return NeighborsView(this->getLenIter(id, lc));
	}

	ThreadSafeConnections::ThreadSafeConnections(
		const uint maxElemCount, const uint mMax, const uint mMax0
	) : Connections(maxElemCount, mMax, mMax0) {

		this->mutexes.resize(maxElemCount);
	}

	Element::Element(const float* const data, const uint id) : data(data), id(id) {}

	float Space::getNorm(const float* const data) const {
		float norm = 0.f;

		for(size_t i = 0; i < this->dim; i++)
			norm += data[i] * data[i];

		return sqrtf(norm);
	}

	float* Space::getData(const uint id) {
		return this->view.getData(id);
	}

	const float* const Space::getData(const uint id) const {
		return this->view.getData(id);
	}

	float Space::getDistance(const float* const aData, const float* const bData) const {
		return this->distFunc(aData, bData, this->dim);
	}

	float Space::getDistance(const float* const aData, const uint bID) const {
		return this->getDistance(aData, this->getData(bID));
	}

	float Space::getDistance(const uint aID, const float* const bData) const {
		return this->getDistance(this->getData(aID), bData);
	}

	float Space::getDistance(const uint aID, const uint bID) const {
		return this->getDistance(this->getData(aID), this->getData(bID));
	}

	void Space::normalizeData(const float* const data, float* const res) const {
		const auto invNorm = 1.f / (this->getNorm(data) + 1e-30f);

		for(size_t i = 0; i < this->dim; i++)
			res[i] = data[i] * invNorm;
	}

	void Space::push(const Element& q) {
		if(this->normalize)
			this->normalizeData(q.data, this->getData(q.id));
		else
			std::copy(q.data, q.data + this->dim, this->getData(q.id));
	}

	Space::Space(const size_t dim, const SpaceKind kind, const uint maxElemCount)
		: distFunc(kind == SpaceKind::EUCLIDEAN ? euclid : innerProd), dim(dim),
		elemData(maxElemCount * this->dim, 0.f), view(this->elemData.data(), this->dim, maxElemCount),
		normalize(kind == SpaceKind::ANGULAR) {}

	bool VisitedSet::isMarked(const uint id) const {
		return this->v[id];
	}

	void VisitedSet::mark(const uint id) {
		this->v[id] = true;
	}

	VisitedSet::VisitedSet(const uint elemCount, const uint epID) : v(elemCount, false) {
		this->mark(epID);
	}

	double IndexConfig::getML() const {
		return 1.0 / std::log(double(this->mMax));
	}

	IndexConfig::IndexConfig(const uint efConstruction, const uint mMax, const uint maxElemCount)
		: efConstruction(efConstruction), maxElemCount(maxElemCount), mMax(mMax), mMax0(mMax * 2) {}

	QueryResults::~QueryResults() {
		if(this->owningData) {
			delete[] this->distances.getData(0);
			delete[] this->ids.getData(0);
		}
	}

	void QueryResults::set(
		const size_t queryIdx, const size_t neighborIdx, const float dist, const uint id
	) {
		this->distances.setVal(queryIdx, neighborIdx, dist);
		this->ids.setVal(queryIdx, neighborIdx, id);
	}

	float QueryResults::getDistance(const size_t queryIdx, const size_t neighborIdx) {
		return this->distances.getVal(queryIdx, neighborIdx);
	}

	uint QueryResults::getID(const size_t queryIdx, const size_t neighborIdx) {
		return this->ids.getVal(queryIdx, neighborIdx);
	}

	size_t QueryResults::getK() const {
		return this->ids.getDim();
	}

	size_t QueryResults::getQueryCount() const {
		return this->ids.getElemCount();
	}

	void QueryResults::push(FarHeap& h, const size_t queryIdx) {
		for(auto neighborIdx = this->getK() - 1;; neighborIdx--) {
			const auto node = h.extractTop();
			this->set(queryIdx, neighborIdx, node.dist, node.id);

			if(!neighborIdx)
				break;
		}
	}

	QueryResults::QueryResults(const size_t k, const uint queryCount)
		: distances(new float[k * queryCount], k, queryCount),
		ids(new uint[k * queryCount], k, queryCount), owningData(true) {}

	NearHeap AbstractIndex::getNearHeap(NeighborsPtr n, const float* const q) {
		NearHeap res;

		for(const auto& id : *n)
			res.push(Node(this->space.getDistance(id, q), id));

		return res;
	}

	Node AbstractIndex::processLowerLayer(const Node& ep, const uint lc, const Element& q) {
		auto W = this->searchLowerLayer(this->cfg.efConstruction, ep, lc, q.data, false);
		const auto R = this->selectNeighbors(this->cfg.mMax, q.data, NearHeap(W));
		this->writeNeighbors(q.id, lc, R);
		const auto mLayer = lc ? this->cfg.mMax : this->cfg.mMax0;

		for(const auto& e : R) {
			const auto eData = this->space.getData(e.id);
			auto N = this->getConn()->getNeighbors(e.id, lc);
			auto nHeap = this->getNearHeap(N, eData);
			nHeap.push(Node(this->space.getDistance(eData, q.data), q.id));

			if(N->len() > mLayer) {
				const auto nRes = this->selectNeighbors(mLayer, eData, nHeap);
				this->writeNeighbors(e.id, lc, N, nRes);
			} else
				this->writeNeighbors(e.id, lc, N, nHeap);
		}

		return *std::min_element(R.cbegin(), R.cend(), FarHeapCmp());
	}

	FarHeap AbstractIndex::searchLowerLayer(
		const uint ef, const Node& ep, const uint lc, const float* const q, const bool s
	) {
		NearHeap C(ep);
		auto V = this->getVisitedSet(ep);
		FarHeap W(ep);

		while(C.len()) {
			auto c = C.extractTop();
			auto f = W.top();

			if(
				(s || W.len() == ef) &&
				this->space.getDistance(c.id, q) > this->space.getDistance(f.id, q)
			)
				break;

			const auto N = this->getConn()->getNeighbors(c.id, lc);

			for(const auto& eID : *N)
				if(!V->isMarked(eID)) {
					V->mark(eID);
					f = W.top();
					Node e(this->space.getDistance(eID, q), eID);

					if(W.len() < ef || f.dist > e.dist) {
						C.push(e);
						W.push(e);

						if(W.len() > ef)
							W.pop();
					}
				}
		}
	}

	Node AbstractIndex::searchUpperLayer(const Node& ep, const uint lc, const float* const q) {
		Node m = ep;
		uint prev{};

		do {
			const auto N = this->getConn()->getNeighbors(m.id, lc);
			prev = m.id;

			for(const auto& cand : *N) {
				const auto dist = this->space.getDistance(cand, q);

				if(dist < m.dist) {
					m.dist = dist;
					m.id = cand;
				}
			}

		} while(m.id != prev);

		return m;
	}

	std::vector<Node> AbstractIndex::selectNeighbors(
		const uint M, const float* const q, NearHeap& W
	) {
		std::vector<Node> R;
		R.reserve(M);

		if(W.len() <= M) {
			while(W.len())
				R.push_back(W.extractTop());
			return R;
		}

		while(W.len() && R.size() < M) {
			auto close = true;
			auto e = W.extractTop();

			for(const auto& r : R)
				if(this->space.getDistance(e.id, r.id) < e.dist) {
					close = false;
					break;
				}

			if(close)
				R.push_back(e);
		}

		return R;
	}

	void AbstractIndex::insertWithLevel(const Element& q, const uint l) {
		this->getConn()->init(q.id, l);
		this->space.push(q);

		Node ep(this->space.getDistance(this->entryID, q.data), this->entryID);
		const auto L = this->entryLevel;
		auto lc = L;

		while(lc > l) {
			ep = this->searchUpperLayer(ep, lc, q.data);
			lc--;
		}

		lc = std::min(L, l);

		for(;;) {
			ep = this->processLowerLayer(ep, lc, q);

			if(!lc)
				break;

			lc--;
		}
	}

	FarHeap AbstractIndex::query(const float* const q, const uint efSearch, const uint k) {
		const auto efMax = std::max(efSearch, k);
		Node ep(this->space.getDistance(this->entryID, q), this->entryID);
		const auto L = this->entryLevel;
		auto lc = L;

		while(lc > 0) {
			ep = this->searchUpperLayer(ep, lc, q);
			lc--;
		}

		auto W = this->searchLowerLayer(efMax, ep, 0, q, true);

		while(W.len() > k)
			W.pop();

		return W;
	}

	void AbstractIndex::setEntry(const uint id, const uint level) {
		this->entryID = id;
		this->entryLevel = level;
	}

	AbstractIndex::AbstractIndex(IndexConfig cfg, const size_t dim, const SpaceKind spaceKind)
		: cfg(cfg), elemCount(0), entryID(0), entryLevel(0),
		space(dim, spaceKind, this->cfg.maxElemCount) {}

	Element ThreadSafeFloatView::getNextElement() {
		std::unique_lock<std::mutex> lock(this->m);
		Element res(this->v.getData(this->currID), this->currID);
		this->currID++;
		return res;
	}

	ThreadSafeFloatView::ThreadSafeFloatView(const ArrayView<float>& v) : currID(0), v(v) {}

	uint LevelGenerator::getNextLevel() {
		return uint(-std::log(this->dist(this->gen)) * this->mL);
	}

	LevelGenerator::LevelGenerator(const double mL, const uint seed)
		: dist(0.0, 1.0), gen(seed), mL(mL) {}

	Connections* ParallelIndex::getConn() {
		return &this->conn;
	}

	VisitedPtr ParallelIndex::getVisitedSet(const Node& ep) {
		return std::make_shared<VisitedSet>(this->cfg.maxElemCount, ep.id);
	}

	void ParallelIndex::writeNeighbors(const uint id, const uint lc, const std::vector<Node>& R) {
		std::unique_lock lock(this->conn.getMutex(id));
		auto N = this->conn.getWritableNeighbors(id, lc);
		N.clear();

		for(const auto& r : R)
			N.push(r.id);
	}

	void ParallelIndex::writeNeighbors(
		const uint id, const uint lc, NeighborsPtr, const std::vector<Node>& R
	) {
		this->writeNeighbors(id, lc, R);
	}

	void ParallelIndex::writeNeighbors(const uint id, const uint lc, NeighborsPtr, NearHeap& R) {
		std::unique_lock lock(this->conn.getMutex(id));
		auto N = this->conn.getWritableNeighbors(id, lc);
		N.clear();

		while(R.len())
			N.push(R.extractTop().id);
	}

	ParallelIndex::ParallelIndex(
		const IndexConfig& cfg, const size_t dim, const uint levelGenSeed,
		const SpaceKind spaceKind
	) : AbstractIndex(cfg, dim, spaceKind),
		conn(this->cfg.maxElemCount, this->cfg.mMax, this->cfg.mMax0), levelGenSeed(levelGenSeed) {}

	void ParallelIndex::push(const ArrayView<float>& v) {

	}

	QueryResPtr ParallelIndex::queryBatch(
		const ArrayView<float>& v, const uint efSearch, const uint k
	) {

	}

	ParallelInsertWorker::ParallelInsertWorker(
		ParallelIndex* const index, ThreadSafeFloatView* const view, const uint levelGenSeed
	) : index(index), levelGenSeed(levelGenSeed), view(view) {}

	void ParallelInsertWorker::run() {

	}

	std::thread ParallelInsertWorker::start() {
		return std::thread(&ParallelInsertWorker::run, this);
	}

	ParallelQueryWorker::ParallelQueryWorker(
		ParallelIndex* const index, ThreadSafeFloatView* const view,
		const uint efSearch, const uint k
	) : efSearch(efSearch), index(index), k(k), view(view) {}

	void ParallelQueryWorker::run() {

	}

	std::thread ParallelQueryWorker::start() {
		return std::thread(&ParallelQueryWorker::run, this);
	}

	Connections* SequentialIndex::getConn() {
		return &this->conn;
	}

	VisitedPtr SequentialIndex::getVisitedSet(const Node& ep) {
		return std::make_shared<VisitedSet>(this->elemCount, ep.id);
	}

	void SequentialIndex::writeNeighbors(const uint id, const uint lc, const std::vector<Node>& R) {
		this->writeNeighbors(id, lc, this->conn.getNeighbors(id, lc), R);
	}

	void SequentialIndex::writeNeighbors(
		const uint id, const uint lc, NeighborsPtr N, const std::vector<Node>& R
	) {
		N->clear();

		for(const auto& r : R)
			N->push(r.id);
	}

	void SequentialIndex::writeNeighbors(const uint id, const uint lc, NeighborsPtr N, NearHeap& R) {
		N->clear();

		while(R.len())
			N->push(R.extractTop().id);
	}

	void SequentialIndex::push(const ArrayView<float>& v) {
		size_t i = 0;

		if(!this->elemCount) {
			this->elemCount = 1;
			this->entryLevel = this->gen.getNextLevel();
			i = 1;
			this->conn.init(0, this->entryLevel);
			this->space.push(Element(v.getData(0), 0));
		}

		for(; i < v.getElemCount(); i++) {
			this->insertWithLevel(Element(v.getData(i), this->elemCount), this->gen.getNextLevel());
			this->elemCount++;
		}
	}

	QueryResPtr SequentialIndex::queryBatch(
		const ArrayView<float>& v, const uint efSearch, const uint k
	) {
		auto res = std::make_shared<QueryResults>(k, v.getElemCount());

		if(this->space.normalize) {
			std::vector<float> normQuery(this->space.dim, 0.f);

			for(size_t queryIdx = 0; queryIdx < v.getElemCount(); queryIdx++) {
				this->space.normalizeData(v.getData(queryIdx), normQuery.data());
				res->push(this->query(normQuery.data(), efSearch, k), queryIdx);
			}
		} else {
			for(size_t queryIdx = 0; queryIdx < v.getElemCount(); queryIdx++) {
				res->push(this->query(v.getData(queryIdx), efSearch, k), queryIdx);
			}
		}

		return res;
	}

	SequentialIndex::SequentialIndex(
		const IndexConfig& cfg, const size_t dim, const uint levelGenSeed,
		const SpaceKind spaceKind
	) : AbstractIndex(cfg, dim, spaceKind),
		conn(this->cfg.maxElemCount, this->cfg.mMax, this->cfg.mMax0),
		gen(this->cfg.getML(), levelGenSeed) {}

	float getRecall(const ArrayView<uint>& correctIDs, const ArrayView<uint>& foundIDs) {
		size_t hits = 0;
		std::unordered_set<uint> correctSet;
		correctSet.reserve(correctIDs.getDim());

		for(size_t queryIdx = 0; queryIdx < correctIDs.getElemCount(); queryIdx++) {
			correctIDs.fillSet(correctSet, queryIdx);

			for(size_t neighborIdx = 0; neighborIdx < correctIDs.getDim(); neighborIdx++)
				if(correctSet.find(foundIDs.getVal(queryIdx, neighborIdx)) != correctSet.end())
					hits++;
		}

		return float(hits) / float(correctIDs.getComponentCount());
	}
}

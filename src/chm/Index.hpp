#pragma once
#include <algorithm>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_set>
#include <vector>
#include "distances.hpp"

namespace chm {
	using uint = unsigned int;

	struct Node {
		float dist;
		uint id;

		Node();
		Node(const float dist, const uint id);
	};

	struct FarHeapCmp {
		constexpr bool operator()(const Node& a, const Node& b) const noexcept {
			return a.dist < b.dist;
		}
	};

	struct NearHeapCmp {
		constexpr bool operator()(const Node& a, const Node& b) const noexcept {
			return a.dist > b.dist;
		}
	};

	template<class Cmp>
	class Heap {
		std::vector<Node> nodes;

	public:
		Node extractTop();
		template<class OtherCmp> Heap(Heap<OtherCmp>& o);
		Heap() = default;
		Heap(const Node& n);
		size_t len() const;
		void pop();
		void push(const Node& n);
		Node top();
	};

	using FarHeap = Heap<FarHeapCmp>;
	using NearHeap = Heap<NearHeapCmp>;

	class NeighborsInterface {
	public:
		virtual std::vector<uint>::iterator begin() = 0;
		virtual void clear() = 0;
		virtual std::vector<uint>::iterator end() = 0;
		virtual uint len() const = 0;
		virtual void push(const uint id) = 0;
	};

	using NeighborsPtr = std::shared_ptr<NeighborsInterface>;

	class NeighborsCopy : public NeighborsInterface {
		std::vector<uint> n;

	public:
		std::vector<uint>::iterator begin() override;
		void clear() override;
		std::vector<uint>::iterator end() override;
		uint len() const override;
		NeighborsCopy(const std::vector<uint>::iterator& lenIter);
		void push(const uint id) override;
	};

	class NeighborsView : public NeighborsInterface {
		std::vector<uint>::iterator lenIter;

	public:
		std::vector<uint>::iterator begin() override;
		void clear() override;
		std::vector<uint>::iterator end() override;
		uint len() const override;
		NeighborsView(const std::vector<uint>::iterator& lenIter);
		void push(const uint id) override;
	};

	class Connections {
	protected:
		std::vector<uint> layer0;
		const size_t maxLen;
		const size_t maxLen0;
		std::vector<std::vector<uint>> upperLayers;

	protected:
		std::vector<uint>::iterator getLenIter(const uint id, const uint lc);

	public:
		Connections(const uint maxElemCount, const uint mMax, const uint mMax0);
		virtual NeighborsPtr getNeighbors(const uint id, const uint lc);
		void init(const uint id, const uint level);
	};

	class ThreadSafeConnections : public Connections {
		std::vector<std::mutex> mutexes;

	public:
		std::mutex& getMutex(const uint id);
		NeighborsPtr getNeighbors(const uint id, const uint lc) override;
		NeighborsView getWritableNeighbors(const uint id, const uint lc);
		ThreadSafeConnections(const uint maxElemCount, const uint mMax, const uint mMax0);
	};

	template<typename T>
	class ArrayView {
		T* data;
		size_t dim;
		size_t elemCount;

	public:
		ArrayView(T* data, const size_t dim, const size_t elemCount);
		void fillSet(std::unordered_set<uint>& set, const size_t elemIdx) const;
		T* getData(const size_t elemIdx);
		const T* const getData(const size_t elemIdx) const;
		size_t getDim() const;
		size_t getElemCount() const;
		size_t getComponentCount() const;
		T getVal(const size_t elemIdx, const size_t valIdx) const;
		void setVal(const size_t elemIdx, const size_t valIdx, const T val);
	};

	struct Element {
		const float* const data;
		uint id;

		Element(const float* const data, const uint id);
	};

	enum class SpaceKind {
		ANGULAR,
		EUCLIDEAN,
		INNER_PRODUCT
	};

	class Space {
		DistFunc distFunc;
		std::vector<float> elemData;
		ArrayView<float> view;

		float getNorm(const float* const data) const;

	public:
		const size_t dim;
		const bool normalize;

		float* getData(const uint id);
		const float* const getData(const uint id) const;
		float getDistance(const float* const aData, const float* const bData) const;
		float getDistance(const float* const aData, const uint bID) const;
		float getDistance(const uint aID, const float* const bData) const;
		float getDistance(const uint aID, const uint bID) const;
		void normalizeData(const float* const data, float* const res) const;
		void push(const Element& e);
		Space(const size_t dim, const SpaceKind kind, const uint maxElemCount);
	};

	class VisitedSet {
		std::vector<bool> v;

	public:
		bool isMarked(const uint id) const;
		void mark(const uint id);
		VisitedSet(const uint elemCount, const uint epID);
	};

	using VisitedPtr = std::shared_ptr<VisitedSet>;

	struct IndexConfig {
		const uint efConstruction;
		const uint maxElemCount;
		const uint mMax;
		const uint mMax0;

		double getML() const;
		IndexConfig(const uint efConstruction, const uint mMax, const uint maxElemCount);
	};

	class QueryResults {
		ArrayView<float> distances;
		ArrayView<uint> ids;
		bool owningData;

		void set(const size_t queryIdx, const size_t neighborIdx, const float dist, const uint id);

	public:
		~QueryResults();
		float getDistance(const size_t queryIdx, const size_t neighborIdx);
		uint getID(const size_t queryIdx, const size_t neighborIdx);
		size_t getK() const;
		size_t getQueryCount() const;
		void push(FarHeap& h, const size_t queryIdx);
		QueryResults(const size_t k, const uint queryCount);
	};

	using QueryResPtr = std::shared_ptr<QueryResults>;

	class AbstractIndex {
		NearHeap getNearHeap(NeighborsPtr n, const float* const q);
		Node processLowerLayer(const Node& ep, const uint lc, const Element& q);
		FarHeap searchLowerLayer(
			const uint ef, const Node& ep, const uint lc, const float* const q, const bool s
		);
		Node searchUpperLayer(const Node& ep, const uint lc, const float* const q);
		std::vector<Node> selectNeighbors(const uint M, const float* const q, NearHeap& W);

	protected:
		IndexConfig cfg;
		uint elemCount;
		uint entryID;
		uint entryLevel;
		Space space;

		virtual Connections* getConn() = 0;
		virtual VisitedPtr getVisitedSet(const Node& ep) = 0;
		void insertWithLevel(const Element& q, const uint l);
		FarHeap query(const float* const q, const uint efSearch, const uint k);
		void setEntry(const uint id, const uint level);
		virtual void writeNeighbors(const uint id, const uint lc, const std::vector<Node>& R) = 0;
		virtual void writeNeighbors(
			const uint id, const uint lc, NeighborsPtr N, const std::vector<Node>& R
		) = 0;
		virtual void writeNeighbors(const uint id, const uint lc, NeighborsPtr N, NearHeap& R) = 0;

	public:
		AbstractIndex(IndexConfig cfg, const size_t dim, const SpaceKind spaceKind);
		virtual void push(const ArrayView<float>& v) = 0;
		virtual QueryResPtr queryBatch(
			const ArrayView<float>& v, const uint efSearch, const uint k
		) = 0;
	};

	class ThreadSafeFloatView {
		size_t currID;
		std::mutex m;
		const ArrayView<float> v;

	public:
		Element getNextElement();
		ThreadSafeFloatView(const ArrayView<float>& v);
	};

	class LevelGenerator {
		std::uniform_real_distribution<double> dist;
		std::default_random_engine gen;
		const double mL;

	public:
		uint getNextLevel();
		LevelGenerator(const double mL, const uint seed);
	};

	class ParallelIndex : public AbstractIndex {
		ThreadSafeConnections conn;
		std::mutex entryPointMutex;
		uint levelGenSeed;

		Connections* getConn() override;
		VisitedPtr getVisitedSet(const Node& ep) override;
		void writeNeighbors(const uint id, const uint lc, const std::vector<Node>& R) override;
		void writeNeighbors(
			const uint id, const uint lc, NeighborsPtr, const std::vector<Node>& R
		) override;
		void writeNeighbors(const uint id, const uint lc, NeighborsPtr, NearHeap& R) override;

	public:
		ParallelIndex(
			const IndexConfig& cfg, const size_t dim, const uint levelGenSeed,
			const SpaceKind spaceKind
		);
		void push(const ArrayView<float>& v) override;
		QueryResPtr queryBatch(const ArrayView<float>& v, const uint efSearch, const uint k) override;
	};

	class ParallelInsertWorker {
		ParallelIndex* const index;
		const uint levelGenSeed;
		ThreadSafeFloatView* const view;

	public:
		ParallelInsertWorker(
			ParallelIndex* const index, ThreadSafeFloatView* const view, const uint levelGenSeed
		);
		void run();
		std::thread start();
	};

	class ParallelQueryWorker {
		const uint efSearch;
		ParallelIndex* const index;
		const uint k;
		ThreadSafeFloatView* const view;

	public:
		ParallelQueryWorker(
			ParallelIndex* const index, ThreadSafeFloatView* const view,
			const uint efSearch, const uint k
		);
		void run();
		std::thread start();
	};

	class SequentialIndex : public AbstractIndex {
		Connections conn;
		LevelGenerator gen;

		Connections* getConn() override;
		VisitedPtr getVisitedSet(const Node& ep) override;
		void writeNeighbors(const uint id, const uint lc, const std::vector<Node>& R) override;
		void writeNeighbors(
			const uint id, const uint lc, NeighborsPtr N, const std::vector<Node>& R
		) override;
		void writeNeighbors(const uint id, const uint lc, NeighborsPtr N, NearHeap& R) override;

	public:
		void push(const ArrayView<float>& v) override;
		QueryResPtr queryBatch(const ArrayView<float>& v, const uint efSearch, const uint k) override;
		SequentialIndex(
			const IndexConfig& cfg, const size_t dim, const uint levelGenSeed,
			const SpaceKind spaceKind
		);
	};

	float getRecall(const ArrayView<uint>& correctIDs, const ArrayView<uint>& foundIDs);

	template<class Cmp>
	inline Node Heap<Cmp>::extractTop() {
		auto res = this->top();
		this->pop();
		return res;
	}

	template<class Cmp>
	template<class OtherCmp>
	inline Heap<Cmp>::Heap(Heap<OtherCmp>& o) {
		this->reserve(o.len());

		while(o.len())
			this->push(o.extractTop());
	}

	template<class Cmp>
	inline Heap<Cmp>::Heap(const Node& n) {
		this->push(n);
	}

	template<class Cmp>
	inline size_t Heap<Cmp>::len() const {
		return this->nodes.size();
	}

	template<class Cmp>
	inline void Heap<Cmp>::pop() {
		std::pop_heap(this->nodes.begin(), this->nodes.end(), Cmp());
		this->nodes.pop_back();
	}

	template<class Cmp>
	inline void Heap<Cmp>::push(const Node& n) {
		this->nodes.push_back(n);
		std::push_heap(this->nodes.begin(), this->nodes.end(), Cmp());
	}

	template<class Cmp>
	inline Node Heap<Cmp>::top() {
		return this->nodes.front();
	}

	template<typename T>
	inline ArrayView<T>::ArrayView(T* data, const size_t dim, const size_t elemCount)
		: data(data), dim(dim), elemCount(elemCount) {}

	template<typename T>
	inline void ArrayView<T>::fillSet(std::unordered_set<uint>& set, const size_t elemIdx) const {
		set.clear();

		for(size_t i = 0; i < this->dim; i++)
			set.insert(this->getVal(elemIdx, i));
	}

	template<typename T>
	inline size_t ArrayView<T>::getComponentCount() const {
		return this->dim * this->elemCount;
	}

	template<typename T>
	inline T* ArrayView<T>::getData(const size_t elemIdx) {
		return this->data + elemIdx * this->dim;
	}

	template<typename T>
	inline const T* const ArrayView<T>::getData(const size_t elemIdx) const {
		return this->data + elemIdx * this->dim;
	}

	template<typename T>
	inline size_t ArrayView<T>::getDim() const {
		return this->dim;
	}

	template<typename T>
	inline size_t ArrayView<T>::getElemCount() const {
		return this->elemCount;
	}

	template<typename T>
	inline T ArrayView<T>::getVal(const size_t elemIdx, const size_t valIdx) const {
		return this->data[elemIdx * this->dim + valIdx];
	}

	template<typename T>
	inline void ArrayView<T>::setVal(const size_t elemIdx, const size_t valIdx, const T val) {
		this->data[elemIdx * this->dim + valIdx] = val;
	}
}

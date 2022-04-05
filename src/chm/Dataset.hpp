#pragma once
#include <filesystem>
#include <iomanip>
#include <ios>
#include <ostream>
#include "Index.hpp"

namespace chm {
	namespace fs = std::filesystem;

	class Dataset {
		size_t dim;
		uint k;
		std::string name;
		SpaceKind space;
		uint testCount;
		uint trainCount;
		std::vector<uint> neighbors;
		std::vector<float> test, train;

	public:
		void build(const IndexPtr& index) const;
		Dataset(const fs::path& p);
		IndexPtr getIndex(
			const uint efConstruction, const uint mMax, const bool parallel,
			const uint seed, const size_t workersNum
		) const;
		float getRecall(const QueryResPtr& results) const;
		bool isAngular() const;
		QueryResPtr query(const IndexPtr& index, const uint efSearch) const;
		void writeLongDescription(const fs::path& p) const;
		void writeLongDescription(std::ostream& s) const;
		void writeShortDescription(std::ostream& s) const;
	};

	using DatasetPtr = std::shared_ptr<const Dataset>;

	void throwCouldNotOpen(const fs::path& p);

	template<typename T>
	void readBinary(std::ifstream& s, T& value) {
		s.read(reinterpret_cast<std::ifstream::char_type*>(&value), sizeof(T));
	}

	template<typename T>
	void readBinary(std::ifstream& s, std::vector<T>& v, const size_t len) {
		v.resize(len);
		s.read(reinterpret_cast<std::ifstream::char_type*>(v.data()), len * sizeof(T));
	}

	template<typename T>
	void writeDescription(std::ostream& s, const std::vector<T>& v, const std::string& name) {
		s << name << "[length " << v.size() << "]\n";

		for(const auto& item : v)
			s << item << '\n';
	}

	template<typename T>
	void writeDescription(
		std::ostream& s, const std::vector<T>& v, const std::string& name,
		const std::streamsize decimalPlaces
	) {
		std::ios streamState(nullptr);
		streamState.copyfmt(s);

		s << name << "[length " << v.size() << "]\n"
			<< std::fixed << std::setprecision(decimalPlaces) << std::showpoint;

		for(const auto& item : v)
			s << item << '\n';

		s.copyfmt(streamState);
	}
}

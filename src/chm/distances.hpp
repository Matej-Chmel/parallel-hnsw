#pragma once
#include <functional>

namespace chm {
	using DistFunc = std::function<float(const float*, const float*, const size_t)>;

	static float euclid(const float* node, const float* query, const size_t dim) {
		auto res = 0.f;

		for(size_t i = 0; i < dim; i++) {
			const auto diff = node[i] - query[i];
			res += diff * diff;
		}

		return res;
	}

	static float innerProdSum(const float* node, const float* query, const size_t dim) {
		auto res = 0.f;

		for(size_t i = 0; i < dim; i++)
			res += node[i] * query[i];

		return res;
	}

	static float innerProd(const float* node, const float* query, const size_t dim) {
		return 1.f - innerProdSum(node, query, dim);
	}
}

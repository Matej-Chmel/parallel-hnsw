#pragma once
#include <stdexcept>
#include "DistanceFunction.hpp"

namespace chm {
	static float euclideanDistance(
		const float* node, const float* query, const size_t dim,
		const size_t, const size_t, const size_t
	) {
		auto res = 0.f;

		for(size_t i = 0; i < dim; i++) {
			const auto diff = node[i] - query[i];
			res += diff * diff;
		}

		return res;
	}

	FunctionInfo euc(euclideanDistance, "euc");

	#if defined(AVX_CAPABLE)
		static float euclideanDistance16AVX(
			const float* node, const float* query, const size_t,
			const size_t, const size_t dim16, const size_t
		) {
			__m256 diff, v1, v2;
			const float* end = node + dim16;
			float PORTABLE_ALIGN32 tmp[8];
			__m256 sum = _mm256_set1_ps(0);

			while (node < end) {
				v1 = _mm256_loadu_ps(node);
				node += 8;
				v2 = _mm256_loadu_ps(query);
				query += 8;
				diff = _mm256_sub_ps(v1, v2);
				sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

				v1 = _mm256_loadu_ps(node);
				node += 8;
				v2 = _mm256_loadu_ps(query);
				query += 8;
				diff = _mm256_sub_ps(v1, v2);
				sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
			}

			_mm256_store_ps(tmp, sum);
			return
				tmp[0] + tmp[1] + tmp[2] + tmp[3] +
				tmp[4] + tmp[5] + tmp[6] + tmp[7];
		}

		static float euclideanDistance16ResidualAVX(
			const float* node, const float* query, const size_t,
			const size_t dim4, const size_t dim16, const size_t dimLeft
		) {
			const float front = euclideanDistance16AVX(node, query, 0, dim4, dim16, 0);
			const float back = euclideanDistance(node + dim16, query + dim16, dimLeft, 0, 0, 0);
			return front + back;
		}

		FunctionInfo euc16AVX(euclideanDistance16AVX, "euc16AVX");
		FunctionInfo euc16RAVX(euclideanDistance16ResidualAVX, "euc16RAVX");
	#endif

	#if defined(AVX512_CAPABLE)
		static float euclideanDistance16AVX512(
			const float* node, const float* query, const size_t,
			const size_t, const size_t dim16, const size_t
		) {
			__m512 diff, v1, v2;
			const float* end = node + dim16;
			__m512 sum = _mm512_set1_ps(0);
			float PORTABLE_ALIGN64 tmp[16];

			while (node < end) {
				v1 = _mm512_loadu_ps(node);
				node += 16;
				v2 = _mm512_loadu_ps(query);
				query += 16;
				diff = _mm512_sub_ps(v1, v2);
				sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
			}

			_mm512_store_ps(tmp, sum);
			return
				tmp[0] + tmp[1] + tmp[2] + tmp[3] +
				tmp[4] + tmp[5] + tmp[6] + tmp[7] +
				tmp[8] + tmp[9] + tmp[10] + tmp[11] +
				tmp[12] + tmp[13] + tmp[14] + tmp[15];
		}

		static float euclideanDistance16ResidualAVX512(
			const float* node, const float* query, const size_t,
			const size_t dim4, const size_t dim16, const size_t dimLeft
		) {
			const float front = euclideanDistance16AVX512(node, query, 0, dim4, dim16, 0);
			const float back = euclideanDistance(node + dim16, query + dim16, dimLeft, 0, 0, 0);
			return front + back;
		}

		FunctionInfo euc16AVX512(euclideanDistance16AVX512, "euc16AVX512");
		FunctionInfo euc16RAVX512(euclideanDistance16ResidualAVX512, "euc16RAVX512");
	#endif

	#if defined(SSE_CAPABLE)
		static float euclideanDistance16SSE(
			const float* node, const float* query, const size_t,
			const size_t, const size_t dim16, const size_t
		) {
			__m128 diff, v1, v2;
			const float* end = node + dim16;
			__m128 sum = _mm_set1_ps(0);
			float PORTABLE_ALIGN32 tmp[8];

			while (node < end) {
				v1 = _mm_loadu_ps(node);
				node += 4;
				v2 = _mm_loadu_ps(query);
				query += 4;
				diff = _mm_sub_ps(v1, v2);
				sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

				v1 = _mm_loadu_ps(node);
				node += 4;
				v2 = _mm_loadu_ps(query);
				query += 4;
				diff = _mm_sub_ps(v1, v2);
				sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

				v1 = _mm_loadu_ps(node);
				node += 4;
				v2 = _mm_loadu_ps(query);
				query += 4;
				diff = _mm_sub_ps(v1, v2);
				sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

				v1 = _mm_loadu_ps(node);
				node += 4;
				v2 = _mm_loadu_ps(query);
				query += 4;
				diff = _mm_sub_ps(v1, v2);
				sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
			}

			_mm_store_ps(tmp, sum);
			return tmp[0] + tmp[1] + tmp[2] + tmp[3];
		}

		static float euclideanDistance4SSE(
			const float* node, const float* query, const size_t,
			const size_t dim4, const size_t, const size_t
		) {
			__m128 diff, v1, v2;
			const float* end = node + dim4;
			__m128 sum = _mm_set1_ps(0);
			float PORTABLE_ALIGN32 tmp[8];

			while (node < end) {
				v1 = _mm_loadu_ps(node);
				node += 4;
				v2 = _mm_loadu_ps(query);
				query += 4;
				diff = _mm_sub_ps(v1, v2);
				sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
			}

			_mm_store_ps(tmp, sum);
			return tmp[0] + tmp[1] + tmp[2] + tmp[3];
		}

		static float euclideanDistance4ResidualSSE(
			const float* node, const float* query, const size_t,
			const size_t dim4, const size_t dim16, const size_t dimLeft
		) {
			const float front = euclideanDistance4SSE(node, query, 0, dim4, dim16, 0);
			const float back = euclideanDistance(node + dim4, query + dim4, dimLeft, 0, 0, 0);
			return front + back;
		}

		static float euclideanDistance16ResidualSSE(
			const float* node, const float* query, const size_t,
			const size_t dim4, const size_t dim16, const size_t dimLeft
		) {
			const float front = euclideanDistance16SSE(node, query, 0, dim4, dim16, 0);
			const float back = euclideanDistance(node + dim16, query + dim16, dimLeft, 0, 0, 0);
			return front + back;
		}

		FunctionInfo euc16SSE(euclideanDistance16SSE, "euc16SSE");
		FunctionInfo euc4SSE(euclideanDistance4SSE, "euc4SSE");
		FunctionInfo euc4RSSE(euclideanDistance4ResidualSSE, "euc4RSSE");
		FunctionInfo euc16RSSE(euclideanDistance16ResidualSSE, "euc16RSSE");
	#endif

	inline DistanceInfo getEuclideanInfo(
		const size_t dim, const size_t dim4, const size_t dim16, SIMDType type
	) {
		#if defined(SIMD_CAPABLE)
			if(type == SIMDType::NONE)
				return DistanceInfo(0, euc);

			if(type == SIMDType::BEST)
				type = getBestSIMDType();

			if(dim % 16 == 0)
				switch(type) {
					case SIMDType::AVX:
						#if defined(AVX_CAPABLE)
							return DistanceInfo(0, euc16AVX);
						#else
							throw std::runtime_error("This CPU doesn't support AVX.");
						#endif
					case SIMDType::AVX512:
						#if defined(AVX512_CAPABLE)
							return DistanceInfo(0, euc16AVX512);
						#else
							throw std::runtime_error("This CPU doesn't support AVX512.");
						#endif
					case SIMDType::SSE:
						#if defined(SSE_CAPABLE)
							return DistanceInfo(0, euc16SSE);
						#else
							throw std::runtime_error("This CPU doesn't support SSE.");
						#endif
					default:
						throw std::runtime_error("Unknown SIMD type.");
				}
			else if(dim % 4 == 0)
				#if defined(SSE_CAPABLE)
					return DistanceInfo(0, euc4SSE);
				#else
					throw std::runtime_error("This CPU doesn't support SSE.");
				#endif
			else if (dim > 16) {
				const auto dimLeft = dim - dim16;

				switch(type) {
					case SIMDType::AVX:
						#if defined(AVX_CAPABLE)
							return DistanceInfo(dimLeft, euc16RAVX);
						#else
							throw std::runtime_error("This CPU doesn't support AVX.");
						#endif
					case SIMDType::AVX512:
						#if defined(AVX512_CAPABLE)
							return DistanceInfo(dimLeft, euc16RAVX512);
						#else
							throw std::runtime_error("This CPU doesn't support AVX512.");
						#endif
					case SIMDType::SSE:
						#if defined(SSE_CAPABLE)
							return DistanceInfo(dimLeft, euc16RSSE);
						#else
							throw std::runtime_error("This CPU doesn't support SSE.");
						#endif
					default:
						throw std::runtime_error("Unknown SIMD type.");
				}
			}
			else if (dim > 4)
				#if defined(SSE_CAPABLE)
					return DistanceInfo(dim - dim4, euc4RSSE);
				#else
					throw std::runtime_error("This CPU doesn't support SSE.");
				#endif
		#endif

		return DistanceInfo(0, euc);
	}
}

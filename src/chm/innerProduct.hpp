#pragma once
#include "DistanceFunction.hpp"

namespace chm {
	static float innerProductSum(const float* node, const float* query, const size_t dim) {
		auto res = 0.f;

		for(size_t i = 0; i < dim; i++)
			res += node[i] * query[i];

		return res;
	}

	static float innerProduct(
		const float* node, const float* query, const size_t dim,
		const size_t, const size_t, const size_t
	) {
		return 1.f - innerProductSum(node, query, dim);
	}

	FunctionInfo ip(innerProduct, "ip");

	#if defined(AVX_CAPABLE)
		static float innerProductSum16AVX(const float* node, const float* query, const size_t dim16) {
			const float* end = node + dim16;
			__m256 sum = _mm256_set1_ps(0);
			float PORTABLE_ALIGN32 tmp[8];

			while(node < end) {
				__m256 v1 = _mm256_loadu_ps(node);
				node += 8;
				__m256 v2 = _mm256_loadu_ps(query);
				query += 8;
				sum = _mm256_add_ps(sum, _mm256_mul_ps(v1, v2));

				v1 = _mm256_loadu_ps(node);
				node += 8;
				v2 = _mm256_loadu_ps(query);
				query += 8;
				sum = _mm256_add_ps(sum, _mm256_mul_ps(v1, v2));
			}

			_mm256_store_ps(tmp, sum);
			return
				tmp[0] + tmp[1] + tmp[2] + tmp[3] +
				tmp[4] + tmp[5] + tmp[6] + tmp[7];
		}

		static float innerProduct16AVX(
			const float* node, const float* query, const size_t,
			const size_t, const size_t dim16, const size_t
		) {
			return 1.f - innerProductSum16AVX(node, query, dim16);
		}

		static float innerProduct16ResidualAVX(
			const float* node, const float* query, const size_t,
			const size_t dim4, const size_t dim16, const size_t dimLeft
		) {
			const float front = innerProductSum16AVX(node, query, dim16);
			const float back = innerProductSum(node + dim16, query + dim16, dimLeft);
			return 1.f - (front + back);
		}

		static float innerProductSum4AVX(
			const float* node, const float* query, const size_t dim4, const size_t dim16
		) {
			const float* end4 = node + dim4;
			const float* end16 = node + dim16;
			__m256 sum = _mm256_set1_ps(0);
			float PORTABLE_ALIGN32 tmp[8];

			while(node < end16) {
				__m256 v1 = _mm256_loadu_ps(node);
				node += 8;
				__m256 v2 = _mm256_loadu_ps(query);
				query += 8;
				sum = _mm256_add_ps(sum, _mm256_mul_ps(v1, v2));

				v1 = _mm256_loadu_ps(node);
				node += 8;
				v2 = _mm256_loadu_ps(query);
				query += 8;
				sum = _mm256_add_ps(sum, _mm256_mul_ps(v1, v2));
			}

			__m128 v1, v2;
			__m128 prod = _mm_add_ps(_mm256_extractf128_ps(sum, 0), _mm256_extractf128_ps(sum, 1));

			while(node < end4) {
				v1 = _mm_loadu_ps(node);
				node += 4;
				v2 = _mm_loadu_ps(query);
				query += 4;
				prod = _mm_add_ps(prod, _mm_mul_ps(v1, v2));
			}

			_mm_store_ps(tmp, prod);
			return tmp[0] + tmp[1] + tmp[2] + tmp[3];
		}

		static float innerProduct4AVX(
			const float* node, const float* query, const size_t,
			const size_t dim4, const size_t dim16, const size_t
		) {
			return 1.f - innerProductSum4AVX(node, query, dim4, dim16);
		}

		static float innerProduct4ResidualAVX(
			const float* node, const float* query, const size_t,
			const size_t dim4, const size_t dim16, const size_t dimLeft
		) {
			const float front = innerProductSum4AVX(node, query, dim4, dim16);
			const float back = innerProductSum(node + dim4, query + dim4, dimLeft);
			return 1.f - (front + back);
		}

		FunctionInfo ip16AVX(innerProduct16AVX, "ip16AVX");
		FunctionInfo ip16RAVX(innerProduct16ResidualAVX, "ip16RAVX");
		FunctionInfo ip4AVX(innerProduct4AVX, "ip4AVX");
		FunctionInfo ip4RAVX(innerProduct4ResidualAVX, "ip4RAVX");
	#endif

	#if defined(AVX512_CAPABLE)
		static float innerProductSum16AVX512(
			const float* node, const float* query, const size_t dim16
		) {
			const float* end = node + dim16;
			__m512 sum = _mm512_set1_ps(0);
			float PORTABLE_ALIGN64 tmp[16];

			while(node < end) {
				__m512 v1 = _mm512_loadu_ps(node);
				node += 16;
				__m512 v2 = _mm512_loadu_ps(query);
				query += 16;
				sum = _mm512_add_ps(sum, _mm512_mul_ps(v1, v2));
			}

			_mm512_store_ps(tmp, sum);
			return
				tmp[0] + tmp[1] + tmp[2] + tmp[3] +
				tmp[4] + tmp[5] + tmp[6] + tmp[7] +
				tmp[8] + tmp[9] + tmp[10] + tmp[11] +
				tmp[12] + tmp[13] + tmp[14] + tmp[15];
		}

		static float innerProduct16AVX512(
			const float* node, const float* query, const size_t,
			const size_t, const size_t dim16, const size_t
		) {
			return 1.f - innerProductSum16AVX512(node, query, dim16);
		}

		static float innerProduct16ResidualAVX512(
			const float* node, const float* query, const size_t,
			const size_t dim4, const size_t dim16, const size_t dimLeft
		) {
			const float front = innerProductSum16AVX512(node, query, dim16);
			const float back = innerProductSum(node + dim16, query + dim16, dimLeft);
			return 1.f - (front + back);
		}

		FunctionInfo ip16AVX512(innerProduct16AVX512, "ip16AVX512");
		FunctionInfo ip16RAVX512(innerProduct16ResidualAVX512, "ip16RAVX512");
	#endif

	#if defined(SSE_CAPABLE)
		static float innerProductSum16SSE(
			const float* node, const float* query, const size_t dim16
		) {
			const float* end = node + dim16;
			float PORTABLE_ALIGN32 tmp[8];
			__m128 sum = _mm_set1_ps(0);
			__m128 v1, v2;

			while(node < end) {
				v1 = _mm_loadu_ps(node);
				node += 4;
				v2 = _mm_loadu_ps(query);
				query += 4;
				sum = _mm_add_ps(sum, _mm_mul_ps(v1, v2));

				v1 = _mm_loadu_ps(node);
				node += 4;
				v2 = _mm_loadu_ps(query);
				query += 4;
				sum = _mm_add_ps(sum, _mm_mul_ps(v1, v2));

				v1 = _mm_loadu_ps(node);
				node += 4;
				v2 = _mm_loadu_ps(query);
				query += 4;
				sum = _mm_add_ps(sum, _mm_mul_ps(v1, v2));

				v1 = _mm_loadu_ps(node);
				node += 4;
				v2 = _mm_loadu_ps(query);
				query += 4;
				sum = _mm_add_ps(sum, _mm_mul_ps(v1, v2));
			}

			_mm_store_ps(tmp, sum);
			return tmp[0] + tmp[1] + tmp[2] + tmp[3];
		}

		static float innerProduct16SSE(
			const float* node, const float* query, const size_t,
			const size_t, const size_t dim16, const size_t
		) {
			return 1.f - innerProductSum16SSE(node, query, dim16);
		}

		static float innerProduct16ResidualSSE(
			const float* node, const float* query, const size_t,
			const size_t dim4, const size_t dim16, const size_t dimLeft
		) {
			const float front = innerProductSum16SSE(node, query, dim16);
			const float back = innerProductSum(node + dim16, query + dim16, dimLeft);
			return 1.f - (front + back);
		}

		static float innerProductSum4SSE(
			const float* node, const float* query, const size_t dim4, const size_t dim16
		) {
			const float* end4 = node + dim4;
			const float* end16 = node + dim16;
			float PORTABLE_ALIGN32 tmp[8];
			__m128 sum = _mm_set1_ps(0);
			__m128 v1, v2;

			while(node < end16) {
				v1 = _mm_loadu_ps(node);
				node += 4;
				v2 = _mm_loadu_ps(query);
				query += 4;
				sum = _mm_add_ps(sum, _mm_mul_ps(v1, v2));

				v1 = _mm_loadu_ps(node);
				node += 4;
				v2 = _mm_loadu_ps(query);
				query += 4;
				sum = _mm_add_ps(sum, _mm_mul_ps(v1, v2));

				v1 = _mm_loadu_ps(node);
				node += 4;
				v2 = _mm_loadu_ps(query);
				query += 4;
				sum = _mm_add_ps(sum, _mm_mul_ps(v1, v2));

				v1 = _mm_loadu_ps(node);
				node += 4;
				v2 = _mm_loadu_ps(query);
				query += 4;
				sum = _mm_add_ps(sum, _mm_mul_ps(v1, v2));
			}

			while(node < end4) {
				v1 = _mm_loadu_ps(node);
				node += 4;
				v2 = _mm_loadu_ps(query);
				query += 4;
				sum = _mm_add_ps(sum, _mm_mul_ps(v1, v2));
			}

			_mm_store_ps(tmp, sum);
			return tmp[0] + tmp[1] + tmp[2] + tmp[3];
		}

		static float innerProduct4SSE(
			const float* node, const float* query, const size_t,
			const size_t dim4, const size_t dim16, const size_t
		) {
			return 1.f - innerProductSum4SSE(node, query, dim4, dim16);
		}

		static float innerProduct4ResidualSSE(
			const float* node, const float* query, const size_t,
			const size_t dim4, const size_t dim16, const size_t dimLeft
		) {
			const float front = innerProductSum4SSE(node, query, dim4, dim16);
			const float back = innerProductSum(node + dim4, query + dim4, dimLeft);
			return 1.f - (front + back);
		}

		FunctionInfo ip16SSE(innerProduct16SSE, "ip16SSE");
		FunctionInfo ip16RSSE(innerProduct16ResidualSSE, "ip16RSSE");
		FunctionInfo ip4SSE(innerProduct4SSE, "ip4SSE");
		FunctionInfo ip4RSSE(innerProduct4ResidualSSE, "ip4RSSE");
	#endif

	inline DistanceInfo getInnerProductInfo(
		const size_t dim, const size_t dim4, const size_t dim16, SIMDType type
	) {
		#if defined(SIMD_CAPABLE)
			if(type == SIMDType::NONE)
				return DistanceInfo(0, ip);

			if(type == SIMDType::BEST)
				type = getBestSIMDType();

			if(dim % 16 == 0)
				switch(type) {
					case SIMDType::AVX:
						#if defined(AVX_CAPABLE)
							return DistanceInfo(0, ip16AVX);
						#else
							throw std::runtime_error("This CPU doesn't support AVX.");
						#endif
					case SIMDType::AVX512:
						#if defined(AVX512_CAPABLE)
							return DistanceInfo(0, ip16AVX512);
						#else
							throw std::runtime_error("This CPU doesn't support AVX512.");
						#endif
					case SIMDType::SSE:
						#if defined(SSE_CAPABLE)
							return DistanceInfo(0, ip16SSE);
						#else
							throw std::runtime_error("This CPU doesn't support SSE.");
						#endif
					default:
						throw std::runtime_error("Unknown SIMD type.");
				}
			else if(dim % 4 == 0)
				switch(type) {
					case SIMDType::AVX:
					case SIMDType::AVX512:
						#if defined(AVX_CAPABLE)
							return DistanceInfo(0, ip4AVX);
						#else
							throw std::runtime_error("This CPU doesn't support AVX.");
						#endif
					case SIMDType::SSE:
						#if defined(SSE_CAPABLE)
							return DistanceInfo(0, ip4SSE);
						#else
							throw std::runtime_error("This CPU doesn't support SSE.");
						#endif
					default:
						throw std::runtime_error("Unknown SIMD type.");
				}
			else if (dim > 16) {
				const auto dimLeft = dim - dim16;

				switch(type) {
					case SIMDType::AVX:
						#if defined(AVX_CAPABLE)
							return DistanceInfo(dimLeft, ip16RAVX);
						#else
							throw std::runtime_error("This CPU doesn't support AVX.");
						#endif
					case SIMDType::AVX512:
						#if defined(AVX512_CAPABLE)
							return DistanceInfo(dimLeft, ip16RAVX512);
						#else
							throw std::runtime_error("This CPU doesn't support AVX512.");
						#endif
					case SIMDType::SSE:
						#if defined(SSE_CAPABLE)
							return DistanceInfo(dimLeft, ip16RSSE);
						#else
							throw std::runtime_error("This CPU doesn't support SSE.");
						#endif
					default:
						throw std::runtime_error("Unknown SIMD type.");
				}
			}
			else if (dim > 4) {
				const auto dimLeft = dim - dim4;

				switch(type) {
					case SIMDType::AVX:
					case SIMDType::AVX512:
						#if defined(AVX_CAPABLE)
							return DistanceInfo(dimLeft, ip4RAVX);
						#else
							throw std::runtime_error("This CPU doesn't support AVX.");
						#endif
					case SIMDType::SSE:
						#if defined(SSE_CAPABLE)
							return DistanceInfo(dimLeft, ip4RSSE);
						#else
							throw std::runtime_error("This CPU doesn't support SSE.");
						#endif
					default:
						throw std::runtime_error("Unknown SIMD type.");
				}
			}
		#endif

		return DistanceInfo(0, ip);
	}
}

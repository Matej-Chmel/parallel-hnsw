#pragma once
#include <string>

#if defined(SIMD_CAPABLE)
	#include <immintrin.h>

	#ifdef _MSC_VER
		#include <intrin.h>
		#include <stdexcept>
	#else
		#include <x86intrin.h>
	#endif

	#if defined(__GNUC__)
		#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
		#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
	#else
		#define PORTABLE_ALIGN32 __declspec(align(32))
		#define PORTABLE_ALIGN64 __declspec(align(64))
	#endif
#endif

namespace chm {
	typedef float (*DistanceFunction)(
		const float*, const float*, const size_t,
		const size_t, const size_t, const size_t
	);

	enum class SIMDType {
		AVX,
		AVX512,
		BEST,
		NONE,
		SSE
	};

	SIMDType getBestSIMDType();
	SIMDType getSIMDType(std::string s);

	struct FunctionInfo {
		const DistanceFunction f;
		const char* const name;

		FunctionInfo(const DistanceFunction f, const char* const name);
	};

	struct DistanceInfo {
		const size_t dimLeft;
		const FunctionInfo funcInfo;

		DistanceInfo(const size_t dimLeft, const FunctionInfo funcInfo);
	};
}

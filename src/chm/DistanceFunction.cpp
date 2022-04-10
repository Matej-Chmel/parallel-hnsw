#include <algorithm>
#include <cctype>
#include <stdexcept>
#include "DistanceFunction.hpp"

namespace chm {
	SIMDType getBestSIMDType() {
		#if defined(AVX512_CAPABLE)
			return SIMDType::AVX512;
		#elif defined(AVX_CAPABLE)
			return SIMDType::AVX;
		#elif defined(SSE_CAPABLE)
			return SIMDType::SSE;
		#else
			return SIMDType::NONE;
		#endif
	}

	SIMDType getSIMDType(std::string s) {
		std::transform(s.begin(), s.end(), s.begin(), ::tolower);

		if(s == "avx")
			return SIMDType::AVX;
		if(s == "avx512")
			return SIMDType::AVX512;
		if(s == "best")
			return SIMDType::BEST;
		if(s == "none")
			return SIMDType::NONE;
		if(s == "sse")
			return SIMDType::SSE;
		throw std::runtime_error("Invalid SIMD type.");
	}

	std::string SIMDTypeToStr(const SIMDType s) {
		switch(s) {
			case SIMDType::AVX:
				return "avx";
			case SIMDType::AVX512:
				return "avx512";
			case SIMDType::BEST:
				return SIMDTypeToStr(getBestSIMDType());
			case SIMDType::NONE:
				return "none";
			case SIMDType::SSE:
				return "sse";
			default:
				throw std::runtime_error("Invalid SIMD type.");
		}
	}

	FunctionInfo::FunctionInfo(const DistanceFunction f, const char* const name) : f(f), name(name) {}

	DistanceInfo::DistanceInfo(const size_t dimLeft, const FunctionInfo funcInfo)
		: dimLeft(dimLeft), funcInfo(funcInfo) {}
}

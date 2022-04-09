import cpufeature as cf

class SIMDCapability:
	def __init__(self):
		info: dict = cf.CPUFeature
		osAVX = info.get("OS_AVX", False)
		self.avx = info.get("AVX", False) and osAVX
		self.avx2 = info.get("AVX2", False)
		self.avx512 = info.get("AVX512f", False) and info.get("OS_AVX512", False)
		self.sse = info.get("SSE", False)
		self.any = self.avx or self.avx2 or self.avx512 or self.sse

	def __str__(self):
		N = "\n"
		return (
			f"SIMD capability:{N}"
			f"Any: {self.any}{N}"
			f"AVX: {self.avx}{N}"
			f"AVX2: {self.avx2}{N}"
			f"AVX-512: {self.avx512}{N}"
			f"SSE: {self.sse}"
		)

	def getMacros(self):
		res = []

		if self.any:
			res.append("SIMD_CAPABLE")

			if self.avx or self.avx2:
				res.append("AVX_CAPABLE")
			if self.avx512:
				res.append("AVX512_CAPABLE")
			if self.sse:
				res.append("SSE_CAPABLE")

		return res

	def getMsvcArchFlag(self):
		if self.avx512:
			return "/arch:AVX512"
		if self.avx2:
			return "/arch:AVX2"
		if self.avx:
			return "/arch:AVX"
		return None

def main():
	cf.print_features()
	print()
	print(SIMDCapability())

if __name__ == "__main__":
	main()

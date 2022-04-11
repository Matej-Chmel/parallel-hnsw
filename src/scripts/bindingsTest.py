import parallel_hnsw as h

def main():
	dataset = h.Dataset(
		25, 10, 104, h.Space.ANGULAR, h.SIMDType.BEST, 200, 20000
	)
	b = h.Benchmark(
		dataset, 200,
		[10, 20, 40, 80, 120, 300, 500],
		200, 16, False, 2
	)

	b.run(True).print()
	b.getParallel(2).run(True).print()
	print(b)
	print(b.dataset)

if __name__ == "__main__":
	main()

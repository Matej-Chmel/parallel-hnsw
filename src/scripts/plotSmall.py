import plot

def main():
	plot.run(plot.Config(
		dim=25, efConstruction=200, efSearchValues=[10, 20, 40, 80, 120, 200, 400, 600],
		mMax=16, runs=2,
		trainCounts=[500, 1000, 1500, 2000],
		workerCounts=[1, 2, 3, 4]
	))

if __name__ == "__main__":
	main()

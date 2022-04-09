#include <cstdlib>
#include <iostream>
#include "chm/Benchmark.hpp"

int main() {
	using namespace chm;

	try {
		const auto dataset = std::make_shared<Dataset>(
			4, 3, 100, SpaceKind::EUCLIDEAN, SIMDType::BEST, 50, 200
		);
		Benchmark b(dataset, 8, {3, 4, 6}, 10, 101, 2, false, 1, 1);
		b.run().print(std::cout);

	} catch(const std::exception& e) {
		std::cerr << "[ERROR] " << e.what() << '\n';
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

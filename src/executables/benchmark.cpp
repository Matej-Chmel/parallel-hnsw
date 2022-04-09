#include <cstdlib>
#include <iostream>
#include "chm/Benchmark.hpp"

int main() {
	using namespace chm;

	try {
		const auto dataset = std::make_shared<Dataset>(
			25, 10, 104, SpaceKind::ANGULAR, SIMDType::BEST, 200, 20000
		);
		Benchmark b(
			dataset, 200,
			{10, 20, 40, 80, 120, 300, 500},
			200, 16, false, 2
		);

		b.run(std::cout).print(std::cout);
		b.getParallel(2).run(std::cout).print(std::cout);

	} catch(const std::exception& e) {
		std::cerr << "[ERROR] " << e.what() << '\n';
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

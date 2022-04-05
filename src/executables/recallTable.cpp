#include <cstdlib>
#include <filesystem>
#include <iostream>
#include "chm/RecallTable.hpp"
namespace fs = std::filesystem;

#ifndef SRC_DIR
	constexpr auto SRC_DIR = "";
#endif

int main() {
	try {
		chm::RecallTable table(chm::RecallTableConfig(
			fs::path(SRC_DIR) / "data" / "test.bin",
			7, {3, 4, 6}, 3, false, 100, 1
		));
		table.run(std::cout);
		table.print(std::cout);

	} catch(const std::exception& e) {
		std::cerr << "[ERROR] " << e.what() << '\n';
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

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
			fs::path(SRC_DIR) / "data" / "angular-d25-20000.bin",
			200, {10, 50, 100, 500, 1000}, 16, false, 200, 1
		));
		table.run(std::cout);
		table.print(std::cout);

	} catch(const std::exception& e) {
		std::cerr << "[ERROR] " << e.what() << '\n';
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

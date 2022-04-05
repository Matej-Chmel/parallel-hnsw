#include <cstdlib>
#include <filesystem>
#include <iostream>
#include "chm/RecallTable.hpp"
namespace fs = std::filesystem;

#ifndef SRC_DIR
	constexpr auto SRC_DIR = "";
#endif

void runRecallTable(const chm::RecallTableConfig& cfg) {
	chm::RecallTable table(cfg);
	table.run(std::cout);
	table.print(std::cout);
}

int main() {
	try {
		chm::RecallTableConfig cfg(
			fs::path(SRC_DIR) / "data" / "angular-d25-20000.bin",
			200, {10, 50, 100, 500, 1000}, 16, 200
		);
		runRecallTable(cfg);
		runRecallTable(cfg.getParallel(1));
		runRecallTable(cfg.getParallel(2));
		runRecallTable(cfg.getParallel(3));
		runRecallTable(cfg.getParallel(4));

	} catch(const std::exception& e) {
		std::cerr << "[ERROR] " << e.what() << '\n';
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

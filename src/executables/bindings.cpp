#include <iostream>
#include <pybind11/chrono.h>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "chm/Benchmark.hpp"

namespace chm {
	namespace py = pybind11;

	PYBIND11_MODULE(parallel_hnsw, m) {
		m.def("getBestSIMDType", getBestSIMDType);
		m.doc() = "Python bindings for parallel-hnsw.";

		py::class_<BenchmarkStats>(m, "BenchmarkStats")
			.def_readonly("avg", &BenchmarkStats::avg)
			.def_readonly("max", &BenchmarkStats::max)
			.def_readonly("min", &BenchmarkStats::min);

		py::class_<QueryBenchmarkStats>(m, "QueryBenchmarkStats")
			.def_readonly("avg", &QueryBenchmarkStats::avg)
			.def_readonly("max", &QueryBenchmarkStats::max)
			.def_readonly("min", &QueryBenchmarkStats::min)
			.def_readonly("avgRecall", &QueryBenchmarkStats::avgRecall)
			.def_readonly("maxRecall", &QueryBenchmarkStats::maxRecall)
			.def_readonly("minRecall", &QueryBenchmarkStats::minRecall);

		py::enum_<SIMDType>(m, "SIMDType")
			.value("AVX", SIMDType::AVX)
			.value("AVX512", SIMDType::AVX512)
			.value("BEST", SIMDType::BEST)
			.value("NONE", SIMDType::NONE)
			.value("SSE", SIMDType::SSE);

		py::enum_<SpaceKind>(m, "Space")
			.value("ANGULAR", SpaceKind::ANGULAR)
			.value("EUCLIDEAN", SpaceKind::EUCLIDEAN)
			.value("INNER_PRODUCT", SpaceKind::INNER_PRODUCT);

		py::class_<Dataset, DatasetPtr>(m, "Dataset")
			.def(py::init<
				const size_t, const size_t, const uint, const SpaceKind,
				const SIMDType, const size_t, const size_t>(),
				py::arg("dim"), py::arg("k"), py::arg("seed"), py::arg("spaceKind"),
				py::arg("simdType"), py::arg("testCount"), py::arg("trainCount")
			)
			.def("__str__", &Dataset::getString)
			.def_readonly("dim", &Dataset::dim)
			.def_readonly("k", &Dataset::k)
			.def_readonly("space", &Dataset::spaceKind)
			.def_readonly("SIMD", &Dataset::simdType)
			.def_readonly("testCount", &Dataset::testCount)
			.def_readonly("trainCount", &Dataset::trainCount);

		py::class_<Benchmark>(m, "Benchmark")
			.def(py::init<
				DatasetPtr, const uint, const std::vector<uint>&, const uint,
				const uint, const bool, const size_t, const size_t>(),
				py::arg("dataset"), py::arg("efConstruction"), py::arg("efSearchValues"),
				py::arg("levelGenSeed"), py::arg("mMax"), py::arg("parallel"),
				py::arg("runsCount"), py::arg("workerCount") = 1
			)
			.def("__str__", &Benchmark::getString)
			.def("getBuildStats", &Benchmark::getBuildStats)
			.def("getParallel", &Benchmark::getParallel, py::arg("workerCount"))
			.def("getQueryStats", &Benchmark::getQueryStats)
			.def("print", [](const Benchmark& b) {
				b.print(std::cout);
			})
			.def("run", [](Benchmark& b) -> Benchmark& {
				(void)b.run(std::cout);
				return b;
			})
			.def_property_readonly("dataset", &Benchmark::getDataset)
			.def_readonly("parallel", &Benchmark::parallel)
			.def_readonly("runs", &Benchmark::runsCount)
			.def_readonly("workers", &Benchmark::workerCount);

		py::add_ostream_redirect(m, "ostream");
	}
}

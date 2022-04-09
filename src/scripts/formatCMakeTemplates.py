from pathlib import Path
from SIMDCapability import SIMDCapability

def getCMakeDefs(macros: str, target: str):
	N = "\n"
	return f"{N}target_compile_definitions({target} PRIVATE {macros})" if macros else ""

def formatCMakeTemplates(src: Path):
	simd = SIMDCapability()
	arch = simd.getMsvcArchFlag()
	archStr = "" if arch is None else arch
	macros = " ".join(simd.getMacros())

	with (src / "CMakeLists.txt").open("w", encoding="utf-8") as f:
		f.write((src / "templates" / "CMake.txt").read_text(encoding="utf-8"
			).replace("@ARCH@", f" {archStr}"
			).replace("@EXE_DEFS@", getCMakeDefs(macros, "benchmark")
			).replace("@LIB_DEFS@", getCMakeDefs(macros, "chmLib")
		))

def main():
	formatCMakeTemplates(Path(__file__).parent.parent)

if __name__ == "__main__":
	main()

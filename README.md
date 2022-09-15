# Paralelní aproximace KNN problému
Paralelní zpracování techniky Hierarchical Navigable Small Worlds ([HNSW](https://github.com/nmslib/hnswlib)). Součástí je i srovnání se sekvenční implementací.

Další implementace a srovnání s implementací z původního [článku](https://doi.org/10.1109/TPAMI.2018.2889473) je dostupná v repozitáři [approximate-knn](https://github.com/Matej-Chmel/approximate-knn).

## Systémové požadavky
- CMake, verze 3.6 nebo vyšší
- Python, verze 3.7 nebo vyšší

Pro Linux nainstalujte tyto balíčky:
- build-essential
- cmake
- python3.7-dev
- python3.7-venv

## Návod
Spusťte skript [run.py](run.py) Pythonu.

Skript vytvoří virtuální prostředí, stáhne potřebné softwarové balíčky, zkompiluje C++ implementaci a vytvoří k ní Python rozhraní. Poté spustí měření na velmi malých kolekcích, které by měly trvat nanejvýš dvě minuty. Tato měření jsou prováděna pouze dvakrát, takže mohou obsahovat odchylky. Po dokončení měření skript zobrazí grafy srovnání sekvenční a paralelní implementace. Následující graf se zobrazí po zavření předchozího grafu. Všechny grafy jsou poté uloženy ve složce `src/plots`, která bude skriptem vytvořena.

## Software třetích stran
Tento projekt používá část kódu z původní implementace HNSW [hnswlib](https://github.com/nmslib/hnswlib/tree/7cc0ecbd43723418f43b8e73a46debbbc3940346), [Licence](LICENSE_hnswlib).

- Převzata definice typů PORTABLE_ALIGN v souboru [DistanceFunction.hpp](src/chm/DistanceFunction.hpp).
- Převzaty metriky vzdáleností v souborech [euclideanDistance.hpp](src/chm/euclideanDistance.hpp) a [innerProduct.hpp](src/chm/innerProduct.hpp). Seznam změn:
	- Vlastní podmínky kompilace.
	- Volba využitého SIMD rozšíření zohledňuje preference uživatele.
	- Počet složek vektorů, které lze paralelně zpracovat, metrika nepočítá.
	- Každé funkci je přiřazen objekt, který ji zastupuje a ukládá název funkce.

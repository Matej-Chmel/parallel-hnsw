Paralelní aproximace KNN problému

Návod:
Spusťte skript "run.py" pomocí Pythonu verze 3.7 nebo vyšší.

Skript vytvoří virtuální prostředí, stáhne potřebné softwarové balíčky,
zkompiluje C++ implementaci a vytvoří k ní Python rozhraní.
Poté spustí měření na velmi malých kolekcích, které by měly trvat nanejvýš
dvě minuty. Tato měření jsou prováděna pouze dvakrát, takže mohou obsahovat
odchylky. Výsledky měření na větších datech, prováděná vícekrát, jsou
součástí dokumentu "CHM0065_PDS.pdf".

Po dokončení měření skript zobrazí grafy srovnání sekvenční a paralelní
implementace. Následující graf se zobrazí po zavření předchozího grafu.
Všechny grafy jsou poté uloženy ve složce "src/plots", která bude skriptem
vytvořena.

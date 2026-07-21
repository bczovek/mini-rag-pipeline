# Mini RAG pipeline

## Követelmények

- Python 3.11 vagy újabb

## A korpusz

A `corpus/corpus.json` egy manifest fájl, amely leírja, mely PDF-eket kell
feldolgozni, és hogyan:

- `footerLinesPatterns`: reguláris kifejezések listája, amelyekkel az egyes
  oldalak alján ismétlődő lábléc-sorok (pl. oldalszám, "Utolsó módosítás:")
  eltávolíthatók a szövegből.
- `chunkSize`: egy chunkba eső maximum tokenek száma (alapértelmezett: 512).
- `documents`: a feldolgozandó dokumentumok listája, soronként:
  - `inputFile`: a PDF fájl neve (a manifesttel azonos könyvtárban keresi).
  - `skipPages`: kihagyandó oldalszámok listája (pl. tartalomjegyzék).

A `corpus/` könyvtárban található a Magyar Telekom lakossági ÁSZF-jének és
az összes 3-as mellékletének (3a/3b/3c/3d) alap `corpus.json` manifestje és
a hozzá tartozó PDF-ek.

## Futtatás

A pipeline-t a `run.sh` szkripttel kell indítani, amely:

1. létrehoz egy lokális, a projekt könyvtárában lévő Python virtuális
   környezetet (`./.venv`), ha még nem létezik,
2. aktiválja azt,
3. telepíti a szükséges dependecy-ket a virtuális környezetbe
4. lefuttatja a pipeline-t a megadott argumentumokkal.

```bash
cd mini-rag
./run.sh corpus/corpus.json
```

Ez feldolgozza (PDF-parse + chunkolás + embedding) a `corpus.json`-ban
felsorolt összes dokumentumot, majd egy interaktív kérdés-válasz módba lép:

```
Enter a question to search the corpus, or type 'exit' to quit.
> Milyen díjak vonatkoznak a mobil előfizetésekre?
```

A kilépéshez írd be, hogy `exit` (vagy nyomj `Ctrl+D`-t).

### Parancssori paraméterek

| Paraméter | Kötelező | Alapérték | Leírás |
|---|---|---|---|
| `corpus` (pozicionális) | igen | – | A korpusz manifest JSON fájl elérési útja (pl. `corpus/corpus.json`). |
| `--top-k` | nem | `4` | Hány, a kérdéshez legközelebbi (legnagyobb cosine similarity) chunkot jelenítsen meg találatonként. Figyelmen kívül marad, ha `--min-similarity` meg van adva. |
| `--min-similarity` | nem | – | Ha meg van adva, a `--top-k` helyett minden olyan chunkot visszaad, amelynek a kérdéshez viszonyított cosine similarity-je legalább ennyi (egy 0 és 1 közötti szám), a leghasonlóbbtól kezdve.

Példák:

```bash
# Alapértelmezett top-4 találat
./run.sh corpus/corpus.json

# A legjobb 8 találat megjelenítése
./run.sh corpus/corpus.json --top-k 8

# Minden, legalább 0.5 koszinusz-hasonlóságú találat (cosine metrikára épülő funkció)
./run.sh corpus/corpus.json --min-similarity 0.5
```

## Megjegyzések

- A vektortár csak memóriában létezik: a program leállítása után a beágyazott
  adatok elvesznek, a következő futtatáskor a `corpus.json`-ban felsorolt
  dokumentumok újra feldolgozásra és beágyazásra kerülnek.
- A beágyazás ~6 percet vesz igénybe.

# Hybrydowy System Rekomendacji NLP-GNN (MIND)


Projekt implementuje **Hybrydowy System Rekomendacyjny**, który łączy:
1.  **NLP (Oparte na treści)**: Kodowanie tekstu tytułów wiadomości.
2.  **GNN (Kolaboracyjne)**: Oparta na grafach agregacja historii interakcji użytkownika

## 📊 Funkcjonalności
- **Konstrukcja Grafu Heterogenicznego**: Mapuje interakcje Użytkowników i Wiadomości z logów.
- **Trening**: Wykorzystuje agregację historii użytkownika dla wydajnego treningu na dużych zbiorach danych (~4 minuty na epokę).
- **Ewaluacja**: Oblicza **AUC**, **NDCG@10**, **MRR** oraz **HitRate@10**.
- **Eksploracyjna Analiza Danych (EDA)**: Skrypty do analizy rzadkości danych (sparsity), problemu zimnego startu (cold start) i balansu klas.

## 🚀 Konfiguracja

### Wymagania 
- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric
- Pandas, Scikit-learn, Matplotlib, Seaborn

### Instalacja
```bash
pip install torch torch_geometric pandas scikit-learn matplotlib seaborn tqdm
```
## 📂 Dane

Użyte dane to zbiór MIND-Small (Microsoft News Dataset).

1.  **Pobranie danych**: https://msnews.github.io/
2.  **Rozpakowanie** do głównego katalogu w repozytorium
    - Struktura powinna wyglądać tak:
      ```
      /MINDsmall_train/
      /MINDsmall_dev/
      /processed/
      /checkpoints/
      ```

## Użycie

### 1. Data Analysis (EDA)
Generowanie wykresów wizualizujących rzadkość danych, cold-start i rozkład treści:
```bash
python eda_script.py
```
Zapisuje wykresy do `plots/`.

### 2. Trening i Ewaluacja
Trenowanie modelu i ewaluacja na zbiorze walidacyjnym:
```bash
python train.py
```
- **Konfiguracja**: Hiperparametry można dostosować w pliku config.py.
- **Output**: Loss i metryki (AUC, NDCG, MRR) w konsoli. Model zapisany do `checkpoints/`.

## 📈 Wyniki (PoW)
Po jednej epoce na zbiorze MIND-Small:
- **AUC**: 0.53
- **NDCG@10**: 0.31
- **HitRate@10**: 60%

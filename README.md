# Diplomsko-delo

- Podatki: Adult income data, nad 50K dolarjev letno / pod 50K dolarjev letno
- experiments -> koda za logistično regresijo (osnovni model), ansambel modelov pred pozabljanjem podatkov in ansambel modelov s pozabljanjem
- models -> shranjeni modeli
- utils -> "helper" funkcije
- data -> osnovni podatki (adult income), razdeljeni podatki (split ensemble)

## Rezultati eksperimentov

### Logistična regresija - osnovni model

**Uspešnost na validacijski množici:**
- Accuracy: 0.8231
- Podrobno:
  ```
              precision    recall  f1-score   support
       <=50K       0.84      0.95      0.89      4503
        >50K       0.74      0.46      0.57      1530
  ```

**Uspešnost na testni množici:**
- Accuracy: 0.8199
- Podrobno:
  ```
              precision    recall  f1-score   support
       <=50K       0.84      0.94      0.89     11360
        >50K       0.71      0.45      0.55      3700
  ```

### Split Ensemble Model (10 modelov)

**Podatki o razdelitvi:**
- Skupno število vzorcev: 30162
- Vzorcev na model: 3016

**Natančnost posameznih modelov na validacijski množici:**
| Model | Natančnost|
|-------|-----------|
| 1     | 0.8162    |
| 2     | 0.8212    |
| 3     | 0.8493    |
| 4     | 0.8311    |
| 5     | 0.8046    |
| 6     | 0.8311    |
| 7     | 0.7881    |
| 8     | 0.7997    |
| 9     | 0.7930    |
| 10    | 0.8262    |

**Povprečna natančnost validacijske množice:** 0.8161

**Uspešnost ansambla na testni množici:**
- Accuracy: 0.8194
- Podrobno:
  ```
              precision    recall  f1-score   support
       <=50K       0.84      0.94      0.89     11360
        >50K       0.71      0.45      0.55      3700
  ```

### Po unlearning na modelu 3 (-20%)

**Podrobnosti eksperimenta:**
- Odstotek pozabljenih podatkov: 20%
- Število vzorcev pred: 2412 (Odstranjena je bila validacijska množica - 20%)
- Število odstranjenih vzorcev: 482
- Število vzorcev po: 1930

**Rezultati:**
- Natančnost validacijske množice: 0.8262
- Natančnost testne množice ("ansambelsko" glasovanje): 0.8193
- Podrobno:
  ```
              precision    recall  f1-score   support
       <=50K       0.84      0.94      0.89     11360
        >50K       0.71      0.45      0.55      3700
  ```


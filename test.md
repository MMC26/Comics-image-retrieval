# Visualització i Avaluació Qualitativa del Model

Aquest projecte utilitza diverses funcions de test qualitatiu per analitzar el rendiment del model en un entorn on no disposem de les etiquetes reals del conjunt de test. A continuació es descriuen les funcions principals utilitzades per visualitzar i entendre el comportament del model.

---

## `denormalize_imgnet`: Visualització correcta d’imatges

La normalització de les imatges segons els valors d'ImageNet afecta la seva visualització directa. Per aquest motiu, hem implementat la funció `denormalize_imgnet`, que desfà aquesta normalització i permet veure les imatges en el seu format natural. Aquesta funció és cridada en totes les visualitzacions qualitatives posteriors.

---

## `visualize_fixed_texts_from_loader`: Test qualitatiu amb textos fixos

El conjunt de test no inclou les etiquetes que indiquen quina imatge és la correcta per a cada text, ja que està pensat per a una competició. Per aquest motiu, hem creat la funció `visualize_fixed_texts_from_loader`, que:

- Selecciona sempre els mateixos textos (índexs globals fixes).
- Mostra les 10 imatges més similars per a cada text.
- Permet comparar visualment quin model genera millors resultats.

Aquesta funció és especialment útil per comparar diferents versions del model o hiperparàmetres mantenint els textos constants.

---

## `analyze_global_validation_similarities`: Posició de la imatge correcta

Per aprofundir en l’anàlisi qualitativa, hem creat la funció `analyze_global_validation_similarities`, que treballa sobre el conjunt de validació (on sí tenim l’etiqueta correcta). Aquesta funció:

- Calcula la similitud entre el text i totes les imatges globals.
- Ordena les imatges per puntuació de similitud.
- Mostra la posició real de la imatge correcta dins del rànquing.
- Dibuixa un gràfic on es veu si la resposta correcta queda davant o enrere.

Aquest mètode s’assembla a la mètrica **MRR (Mean Reciprocal Rank)**, i és útil per estudiar exemples concrets de forma visual i analítica.

---


## Execució dins de `main.py`

Totes aquestes funcions estan implementades al fitxer `test.py` i s’executen automàticament **després de l’entrenament i la validació** del model, dins del fitxer `main.py`. Aquesta integració permet obtenir de forma immediata:

- Una avaluació qualitativa visual.
- Comparatives entre models.
- Anàlisi detallat de posicionament d’imatges rellevants.

---



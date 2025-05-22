# Resultats de l’Experimentació

Aquest document recull el seguiment de les diferents proves realitzades durant el desenvolupament del sistema de recuperació d’imatges a partir de text, basat en un model multimodal amb BERT i ResNet50.

## 🧪 Enfoc inicial

Des del principi es va optar per utilitzar **hard negative mining** per construir les pèrdues de contrast. Aquesta decisió es va prendre amb la intenció de millorar la discriminació del model entre mostres similars però incorrectes. En les primeres fases es va observar que els resultats del test eren poc coherents: les imatges recuperades sovint no tenien cap relació amb els textos d’entrada.

## 🔹 Fase 1: Proves inicials amb subconjunt de dades (5.000 mostres)

Es va començar amb un model bàsic:
- Arquitectura: BERT + ResNet50
- `embedding_dim = 120`, `lr = 1e-4`, `epochs = 20`

➡️ Resultats:
- Overfitting molt evident: `train recall = 1`, però el `val recall=0.255` era molt baix.
- Enllaç: [W&B Run](https://wandb.ai/grup_07/comics-retrieval/runs/1404lx2u?nw=nwusermmcabestany26)

Es van fer proves variant els hiperparàmetres:

- `embedding_dim = 128`, `lr = 1e-5`  
  → Cap millora significativa, `train recall = 0.95` i `val recall=0.267`.
  [W&B Run](https://wandb.ai/grup_07/comics-retrieval/runs/g2wdqpf2?nw=nwusermmcabestany26)

- `embedding_dim = 512`, `lr = 1e-4`  
  → L’overfitting persistia, amb uns resultats: `train recall = 0.98` i `val recall=0.24`.
  [W&B Run](https://wandb.ai/grup_07/comics-retrieval/runs/j2g69v01?nw=nwusermmcabestany26)

🧠 **Hipòtesi:** massa poques dades i/o un model massa simple per capturar la semàntica multimodal.

## 🔹 Fase 2: Model amb totes les dades i regularització

Per millorar la generalització, es va afegir una capa lineal addicional i un **dropout de 0.3**. També es va entrenar amb **tot el conjunt de dades**. Amb la una configuració amb `embedding_dim=128`, `  learning_rate=1e-4` i `batch_size=32`.

➡️ Resultats:
- L’overfitting es va reduir lleugerament.
- Però el `val recall` continuava sense millorar, a més a més les imatges retornades seguien sent poc rellevants.
- Després de fer `15 epochs`, obtenim els següents resultats: `train recall = 0.609` i un `val recall = 0.248`
- Enllaços:
  - [W&B Run 1](https://wandb.ai/grup_07/comics-retrieval/runs/365t55pz)
  - [W&B Run 1 (cont.)](https://wandb.ai/grup_07/comics-retrieval/runs/syt3ybt2?nw=nwuserlaiaalexandrasjoberg)

## 🛠️ Altres proves: augment de dades

A partir d’aquest moment, es va començar a aplicar **data augmentation** sobre les imatges amb l’objectiu d’augmentar la robustesa del model i intentar millorar el rendiment en validació i test. Les transformacions aplicades incloïen:

- Retall aleatori amb redimensionat (RandomResizedCrop)
- Girs horitzontals aleatoris (RandomHorizontalFlip)
- Canvis lleus en la brillantor, contrast i saturació (ColorJitter)
- Normalització amb les estadístiques d’ImageNet

## 🧪 Nou Enfocament
Vist que els resultats seguien sent insatisfactoris, vam substituir `BERT` per `Sentence-BERT`, un model optimitzat per comparar embeddings en tasques de retrieval.

🔢 Noves mètriques per avaluar el sistema

Amb el canvi a Sentence-BERT, també hem refinat la nostra avaluació amb noves mètriques de retrieval més detallades:
- val_recall@1 i train_recall@1: indiquen la capacitat del model de recuperar la imatge correcta en primera posició dins dels conjunts de validació i entrenament, respectivament.
- val_global_recall@1 i val_global_recall@10: mesuren el recall sobre tot el conjunt de validació, utilitzant com a candidats totes les imatges del corpus. Això és molt més exigent i realista. 

- Enllaços:
  - [W&B Run 1](https://wandb.ai/grup_07/comics-retrieval/runs/o3mwmzsu?nw=nwusermmcabestany26)
  - [W&B Run 2](https://wandb.ai/grup_07/comics-retrieval/runs/97uujv00 )

🔎 Resultats qualitatius: 
Les imatges retornades tenen una millor correspondència semàntica amb el text:
  - Si el text descriu dues persones, les imatges també les mostren.
  - Si el text parla de colors o accions concretes, aquests detalls apareixen visualment.

➡️ Resultats:
- Un bon rendiment en entrenament: amb un `train_recall@1` de ~0.60
- Un rendiment més limitat en validació: el `val_recall@1` és de ~0.29, mostrant que la capacitat de generalització del model és força més baixa.
- Baix rendiment en context global: el `val_global_recall@10` és només ~0.015, la qual cosa indica que quan el model ha de recuperar entre totes les imatges del corpus, les seves prediccions cauen molt.

Sentence-BERT proporciona una base raonablement sòlida per a la tasca, amb associacions text-imatge que funcionen bé en contextos controlats (train/val), però no és suficientment robust per retrieval a escala gran.

### 🔹 Prova amb ResNet18
S’ha substituït ResNet50 per ResNet18 per reduir el temps d'entrenament i evitar overfitting.

- Enllaç: [W&B Run 1](https://wandb.ai/grup_07/comics-retrieval/runs/chk21jiq)
  
🔎 Resultats qualitatius: 
- En textos que descrivien entorns concrets (ex: desert, superheroi), cap de les imatges recuperades contenia aquests elements.
- Les representacions visuals no eren prou riques per captar els detalls de l’escena.

➡️ Resultats:
- El rendiment de ResNet18 és clarament inferior al Sentence-BERT sol, obté uns valors: `train_recall@1` de ~0.30, `val_recall@1`de ~0.21 i un `val_global_recall@10` de ~0.011.

 ResNet18 redueix massa la capacitat de representació → descartat.

 ### 🔹 Eliminació de la primera frase del text
 Moltes descripcions comencen amb frases genèriques (“This panel from a golden age...”) que no aporten informació visual útil.
(ACABAR)

 ### 🔹 Pooling combinat (avg + max)
 Capturar millor tant el context global com els detalls visuals destacats mitjançant la combinació de:
- avg_pool: patrons globals
- max_pool: trets prominents
  (ACABAR)

 ### 🔹 Fine-tuning parcial de ResNet (layer4 + fc)
(ACABAR)
 ## 🧪 Proves amb GTR
 
Per tal de continuar millorant la representació semàntica del text i fer un pas més enllà de Sentence-BERT, hem decidit incorporar un nou encoder: `GTR (Google's T5-based Text Retrieval model)`, GTR és una família de models dissenyada específicament per a tasques de retrieval.
 (ACABAR)

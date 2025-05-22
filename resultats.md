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
  - [W&B Run 2 (cont.)](https://wandb.ai/grup_07/comics-retrieval/runs/syt3ybt2?nw=nwuserlaiaalexandrasjoberg)

## 🛠️ Altres proves: augment de dades

A partir d’aquest moment, es va començar a aplicar **data augmentation** sobre les imatges amb l’objectiu d’augmentar la robustesa del model i intentar millorar el rendiment en validació i test. Les transformacions aplicades incloïen:

- Retall aleatori amb redimensionat (RandomResizedCrop)
- Girs horitzontals aleatoris (RandomHorizontalFlip)
- Canvis lleus en la brillantor, contrast i saturació (ColorJitter)
- Normalització amb les estadístiques d’ImageNet



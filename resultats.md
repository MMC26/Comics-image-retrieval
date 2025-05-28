# Resultats de l’Experimentació

Aquest document recull el seguiment de les diferents proves realitzades durant el desenvolupament del sistema de recuperació d’imatges a partir de text, basat en un model multimodal amb BERT i ResNet50.

## Enfoc inicial

Des del principi es va optar per utilitzar **hard negative mining** per construir les pèrdues de contrast. Aquesta decisió es va prendre amb la intenció de millorar la discriminació del model entre mostres similars però incorrectes. En les primeres fases es va observar que els resultats del test eren poc coherents: les imatges recuperades sovint no tenien cap relació amb els textos d’entrada.

## Fase 1: Proves inicials amb subconjunt de dades (5.000 mostres)

Es va començar amb un model bàsic:
- Arquitectura: BERT + ResNet50
- `embedding_dim = 120`, `lr = 1e-4`, `epochs = 20`

Resultats:
- Overfitting molt evident: `train recall = 1`, però el `val recall=0.255` era molt baix.
- Enllaç: [W&B Run](https://wandb.ai/grup_07/comics-retrieval/runs/1404lx2u?nw=nwusermmcabestany26)

Es van fer proves variant els hiperparàmetres:

- `embedding_dim = 128`, `lr = 1e-5`  
  → Cap millora significativa, `train recall = 0.95` i `val recall=0.267`.
  [W&B Run](https://wandb.ai/grup_07/comics-retrieval/runs/g2wdqpf2?nw=nwusermmcabestany26)

- `embedding_dim = 512`, `lr = 1e-4`  
  → L’overfitting persistia, amb uns resultats: `train recall = 0.98` i `val recall=0.24`.
  [W&B Run](https://wandb.ai/grup_07/comics-retrieval/runs/j2g69v01?nw=nwusermmcabestany26)

**Hipòtesi:** massa poques dades i/o un model massa simple per capturar la semàntica multimodal.

## Fase 2: Model amb totes les dades i regularització

Per millorar la generalització, es va afegir una capa lineal addicional i un **dropout de 0.3**. També es va entrenar amb **tot el conjunt de dades**. Amb la una configuració amb `embedding_dim=128`, `  learning_rate=1e-4` i `batch_size=32`.

Resultats:
- L’overfitting es va reduir lleugerament.
- Però el `val recall` continuava sense millorar, a més a més les imatges retornades seguien sent poc rellevants.
- Després de fer `15 epochs`, obtenim els següents resultats: `train recall = 0.609` i un `val recall = 0.248`
- Enllaços:
  - [W&B Run 1](https://wandb.ai/grup_07/comics-retrieval/runs/365t55pz)
  - [W&B Run 1 (cont.)](https://wandb.ai/grup_07/comics-retrieval/runs/syt3ybt2?nw=nwuserlaiaalexandrasjoberg)

## Altres proves: augment de dades

A partir d’aquest moment, es va començar a aplicar **data augmentation** sobre les imatges amb l’objectiu d’augmentar la robustesa del model i intentar millorar el rendiment en validació i test. Les transformacions aplicades incloïen:

- Retall aleatori amb redimensionat (RandomResizedCrop)
- Girs horitzontals aleatoris (RandomHorizontalFlip)
- Canvis lleus en la brillantor, contrast i saturació (ColorJitter)
- Normalització amb les estadístiques d’ImageNet

## Nou Enfocament: Sentence-BERT
Vist que els resultats seguien sent insatisfactoris, vam substituir `BERT` per `Sentence-BERT`, un model optimitzat per comparar embeddings en tasques de retrieval.
A més a més, ara solament s'agafen les primeres 100 paraules de la caption.

### Noves mètriques per avaluar el sistema

Amb el canvi a Sentence-BERT, també hem refinat la nostra avaluació amb noves mètriques de retrieval més detallades:
- val_recall@1 i train_recall@1: indiquen la capacitat del model de recuperar la imatge correcta en primera posició dins dels conjunts de validació i entrenament, respectivament.
- val_global_recall@1 i val_global_recall@10: mesuren el recall sobre tot el conjunt de validació, utilitzant com a candidats totes les imatges del corpus. Això és molt més exigent i realista. 

- Enllaços:
  - [W&B Run 1](https://wandb.ai/grup_07/comics-retrieval/runs/o3mwmzsu?nw=nwusermmcabestany26)
  - [W&B Run 2](https://wandb.ai/grup_07/comics-retrieval/runs/97uujv00 )

Resultats qualitatius: 
Les imatges retornades tenen una millor correspondència semàntica amb el text:
  - Si el text descriu dues persones, les imatges també les mostren.
  - Si el text parla de colors o accions concretes, aquests detalls apareixen visualment.

Resultats:
- Un bon rendiment en entrenament: amb un `train_recall@1` de ~0.60
- Un rendiment més limitat en validació: el `val_recall@1` és de ~0.29, mostrant que la capacitat de generalització del model és força més baixa.
- Baix rendiment en context global: el `val_global_recall@10` és només ~0.015, la qual cosa indica que quan el model ha de recuperar entre totes les imatges del corpus, les seves prediccions cauen molt.

Sentence-BERT proporciona una base raonablement sòlida per a la tasca, amb associacions text-imatge que funcionen bé en contextos controlats (train/val), però no és suficientment robust per retrieval a escala gran.

### Prova amb ResNet18
S’ha substituït ResNet50 per ResNet18 per reduir el temps d'entrenament i evitar overfitting.

- Enllaç: [W&B Run 1](https://wandb.ai/grup_07/comics-retrieval/runs/chk21jiq)
  
Resultats qualitatius: 
- En textos que descrivien entorns concrets (ex: desert, superheroi), cap de les imatges recuperades contenia aquests elements.
- Les representacions visuals no eren prou riques per captar els detalls de l’escena.

Resultats:
- El rendiment de ResNet18 és clarament inferior al Sentence-BERT sol, obté uns valors: `train_recall@1` de ~0.30, `val_recall@1`de ~0.21 i un `val_global_recall@10` de ~0.011.

 ResNet18 redueix massa la capacitat de representació → descartat.

 ### Eliminació de la primera frase del text
 Moltes descripcions comencen amb frases genèriques (“This panel from a golden age...”) que no aporten informació visual útil.

Per aquest motiu, vam decidir eliminar la primera frase dels textos abans d’extreure’n els embeddings.

- Enllaç: [W&B Run 1](https://wandb.ai/grup_07/comics-retrieval/runs/chk21jiq)

Resultats qualitatius:
Després de la modificació, vam observar que les imatges retornades tenien una millor correspondència amb els detalls significatius del text (com vestimenta, accions o personatges), i les similituds entre text i imatge eren més altes i coherents.

Resultats:
- El rendiment global augementa, obtenint uns valors de `val_global_recall@10` de 0.038 i `val_global_recall@1` de 0.
- Tot i que el Recall@1 no millora, el fet que el Recall@10 augmenti de manera clara indica que el model és més capaç de situar les imatges correctes dins els primers resultats, encara que no sempre en primera posició.

 ### Fine-tuning parcial de ResNet (layer4)
Per intentar millorar la qualitat dels embeddings visuals, vam aplicar fine-tuning parcial a la ResNet50, descongelant exclusivament la capa layer4, una de les més profundes del model. Aquesta capa és responsable de captar patrons semàntics complexos, i la nostra hipòtesi era que ajustar-la permetria adaptar millor les representacions visuals a la tasca de matching amb text.

- Enllaç: [W&B Run 1](https://wandb.ai/grup_07/comics-retrieval/runs/chk21jiqetrieval/runs/jha7f7xo)

Resultats qualitatius:
- Les imatges retornades tenen una gran correlació semàntica amb els textos. Els personatges, les escenes i els detalls concrets descrits són captats amb més precisió visual pel model.

Resultats: 
- Obtenim un `val_blobal_recall@10` de 0.011 i un `val_global_recall@1` de 0.
- Tot i que esperàvem una millora significativa, el Recall@10 es manté baix i fluctua molt al llarg de les èpoques.

Això suggereix que el fine-tuning parcial pot estar desajustant l’arquitectura: en no modificar les capes prèvies, els nous patrons apresos per layer4 podrien no estar ben harmonitzats amb les representacions congelades de les altres capes.

### Fine tuning + treure primera frase
Vam voler provar també com funcionaven aquestes dues millores juntes:

- Enllaç: [W&B Run 1](https://wandb.ai/grup_07/comics-retrieval/runs/jihx14u5)

Resultats qualitatius: 
- El model és capaç d’identificar elements específics com avions, colors o núvols.
- No obstant això, la correspondència general text-imatge és menys precisa que en altres configuracions.

Resultats:
- Obtenim un valor de `gval_global_recall@10` bastant alt, de 0.034, però no millora respecte la configuració de només eliminar la primera frase.
- A més, continuem observant fluctuacions importants entre èpoques.

 ## Proves amb GTR
 
Per tal de continuar millorant la representació semàntica del text i fer un pas més enllà de Sentence-BERT, hem decidit incorporar un nou encoder: `GTR (Google's T5-based Text Retrieval model)`, GTR és una família de models dissenyada específicament per a tasques de retrieval.

Vam aplicar les millors millores que haviem vist que funcionaven amb Sentence-BERT: treure la primera frase i fine-tuning.

### Fine tuning 

- Enllaç: [W&B Run 1](https://wandb.ai/grup_07/comics-retrieval/runs/vob8sf5k)
  
Resultats qualitatius: 
-Els resultats són pobres: el model no capta bé el contingut visual.
- Es produeixen similituds negatives, cosa que indica desalineació entre embeddings text-imatge.

Resultats:
Obtenim un `val_global_recall@10: 0.01908`, pitjor que qualsevol altra configuració anterior.

### Fine tunning + Treure primera frase

- Enllaç: [W&B Run 1](https://wandb.ai/grup_07/comics-retrieval/runs/areda52c)

Resultats qualitatius: 
- No observem una millora clara en la qualitat de les prediccions.
  
Resultats:  
- Obtenim un `val_global_recall@10: 0.04198` (el més alt de tots)
- Malgrat la bona mètrica, la qualitat dels resultats no acompanya, i les fluctuacions dificulten l’estabilitat del model.

### Treure primera frase
- Enllaç: [W&B Run 1](https://wandb.ai/grup_07/comics-retrieval/runs/4sjuwupe )
Resultats qualitatius: 
- El model mostra una millor comprensió semàntica.
- Les imatges retornades representen millor el contingut visual descrit (accions, objectes, vestuari, etc.).

Resultats:
- Obtenim un `val_global_recall@10: 0.026` i un `val_global_recall@1: 0.00382`
- Els resultats són estables entre èpoques i qualitativament coherents.

# Model Final Seleccionat
Després de comparar totes les configuracions, el model amb GTR i eliminació de la primera frase (sense fine-tuning) ha estat el que ha mostrat:

- Millor coherència qualitativa

- Estabilitat en el temps

- Suficient rendiment quantitatiu

Per aquests motius, hem escollit aquesta configuració com la millor per al nostre sistema de matching text-imatge.

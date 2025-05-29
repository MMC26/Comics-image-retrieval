[![Data de lliurament de la tasca](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kv-q1sko)  
[![Obre amb Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=19309682)

# Comic - Recuperació multimodal d'imatges de comics

Comic és un projecte d’aprenentatge profund centrat en la recuperació multimodal, amb l’objectiu d’emparellar els subtítols de les vinyetes dels còmics amb les imatges més rellevants entre diversos candidats. El projecte utilitza codificadors de text i imatge d’última generació (BERT/GTR per al text, ResNet per a les imatges) per aprendre representacions conjuntes, amb un fort enfocament en la **reproductibilitat** i el seguiment d’experiments mitjançant **Weights & Biases (wandb)**.

## Resum del projecte

Donat un subtítol d’un còmic i un conjunt d’imatges candidates, l’objectiu és recuperar la imatge correcta que millor coincideixi amb el subtítol. El sistema s’entrena i s’avalua amb el conjunt de dades **ComicsPAP**, utilitzant una **pèrdua triplet** per apropar en l’espai d’embedding les parelles rellevants text-imatge i allunyar les incorrectes. El projecte explora diferents arquitectures de model i estratègies d’ajust fi, i inclou rutines completes de validació i visualització.

## Estructura del codi

- `main.py`: Punt d’entrada. Estableix llavors aleatòries, prepara la configuració i llança les rutines d’entrenament, validació i test.
- `train.py`: Conté el bucle d’entrenament, funcions de validació i càlculs de mètriques de record.
- `test.py`: Proporciona codi per a l’anàlisi qualitativa i la visualització dels resultats de recuperació.
- `dataloaders.py`, `dataloaders/`: Classes de conjunt de dades, càrrega i preprocessament per a ComicsPAP.
- `utils/utils.py`: Funcions utilitàries, pèrdua personalitzada (CosineTripletLoss), construcció de models i carregadors (`make`).
- `environment.yml`: Especificació de l’entorn Conda per a una instal·lació reproduïble.
- `models/`: (No mostrat, però referenciat al codi: conté les arquitectures dels models.)
- Altres scripts i carpetes auxiliars per a experiments i documentació.

El codi és modular: pots canviar fàcilment les variants de dataset o els backbones dels models editant la configuració a `main.py`.

## Opcions d’execució a `main.py`

El fitxer `main.py` ofereix una configuració flexible mitjançant el diccionari `config`. Pots controlar els principals aspectes del pipeline modificant aquests paràmetres:

### Selecció de backbone del model:
- Canvia entre diferents codificadors de text (p. ex., Sentence-BERT, GTR) amb les opcions `use_sentence_bert` i `use_gtr`.
- Escull la dimensió d’embedding (`embedding_dim`) segons el backbone (p. ex., 768 per a GTR, 384 per a Sentence-BERT).
- Pots congelar o ajustar parts del model (`freeze_bert`, `freeze_resnet`, `fine_tune_resnet_tail`).

### Configuració de dades i entrenament:
- Defineix `batch_size`, `max_samples` (per a proves o execucions petites), `epochs` i `num_workers` per controlar la velocitat i escala de l’entrenament.
- Tria si vols eliminar la primera frase dels subtítols en el preprocessament (`remove_first_sentence`).

### Checkpoints i reproductibilitat:
- La ruta a les dades s'especifica en el fitxer `dataloaders.py` a la funció `ComicsDataset`, està especificada la ruta de la maquina virtual on s'han fet les proves. Si es vol canviar el directori haurà de contenir el dataset en file format `.parquet`
- Especifica on desar i carregar checkpoints del model (`checkpoint_path`).
- Totes les llavors aleatòries s’estableixen a l’inici per assegurar la reproductibilitat.

### Seguiment i monitoratge d’experiments:
- Totes les execucions s’enregistren a **Weights & Biases (wandb)**. Defineix `project_name` per organitzar els experiments.
- Les mètriques d’entrenament i validació, així com l’anàlisi qualitativa, es registren automàticament.

### Com adaptar l’execució:
- Per canviar la configuració del model, edita simplement el diccionari `config`.
- Per executar només inferència o avaluació, pots comentar el bloc d’entrenament i afegir el teu script usant les funcions utilitàries.
- Per a experiments avançats (p. ex., diferents datasets o funcions de pèrdua), amplia els mòduls corresponents (`dataloaders.py`, `utils/utils.py`, o afegeix nous models).

## Exemple de codi

La pipeline principal se centra en entrenar un model de recuperació sobre ComicsPAP usant pèrdua triplet. Tots els experiments i mètriques es registren amb wandb.

### Configuració de l'entorn:

Abans d’executar el codi, crea i activa l’entorn conda:

```sh
conda env create --file environment.yml
conda activate xnap-example
```

Si en es vol utilitzar conda llavors crear un entorn amb venv o pip i utilitzar els requirements

```sh
python -m venv env
env/Scripts/activate
pip install -r requitement.txt
```

### Executar la pipeline principal d'entrenament i avaluació:

```sh
python main.py
```

Això farà:
- Entrenar el model amb el conjunt d’entrenament de ComicsPAP.
- Avalua amb els conjunts de validació i test.
- Registra totes les mètriques i visualitzacions a wandb.
- Crear un checkpoint en format .pt per recuperar els pesos a l'entrenament o analitzar el model

### Característiques clau:
- Utilitza BERT o GTR com codificador de text i ResNet (opcionalment ajustat) com codificador d’imatge.
- Pèrdua triplet personalitzada per aprendre embeddings conjunts.
- Carregadors de dades compatibles amb dades en brut o preprocessades.
- Eines de visualització per a l’anàlisi qualitativa de la recuperació.

## Contributors

- Marsal Cuenca (marsal.cuenca@gmail.com)
- Laia A. Sjöberg Cerezo (laiaalexandra.sjoberg@autonoma.cat)
- Maria Muñoz Cabestany (mmcabestany26@gmail.com)
- Blanca Piñol Chacon (1666134@uab.cat)

## Course & Institution

Xarxes Neuronals i Aprenentatge Profund  
Grau de Matemàtica Computacional i Anàlisi de Dades, UAB, 2023



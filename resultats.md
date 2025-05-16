## Resultats amb totes les dades:

Abans de fer la cerca d'hiperparàmetres, hem fet una prova amb totes les dades: 
- **Batch size**: 32  
- **Èpoques**: 5  
- **BERT**: True  
- **RESNET**: True  
- **Learning rate**: 0.0001  
- **Embedding dim**: 512

Aquesta prova ens ha resultat molt lenta i, per aquest motiu, no l’hem pogut repetir. Tot i així, hem obtingut els gràfics de loss i accuracy del train i de la validació.

En el **train**, els resultats han tingut sentit, ja que la loss ha anat baixant i l’accuracy ha anat pujant fins al 72%. 
En canvi, en la **validació**, la loss ha anat pujant i baixant sense una tendència clara, i l’accuracy ha disminuït lleugerament fins al 52%. Això pot ser degut a que no hem pogut fer masses èpoques.

## Resultats de la cerca d’hiperparàmetres:

Per intentar millorar el model, hem fet una cerca d’hiperparàmetres provant diferents combinacions.
La cerca s’ha fet amb poques mostres (**`max_samples = 20`**) per evitar temps d’entrenament elevat. Alguns experiments han acabat amb error `KILLED`, però aquests són els que han funcionat:

Les combinacions provades han sigut:
| Exp | LR     | EMB | BERT | RESNET | BS | Train Loss | Train Acc | Val Loss | Val Acc |
|-----|--------|-----|------|--------|----|------------|-----------|----------|---------|
| 1   | 0.0001 | 128 | ✅   | ✅     | 16 | 0.0267     | 1.0000    | 0.1992   | 0.5500  |
| 2   | 0.0001 | 512 | ✅   | ✅     | 16 | 0.0504     | 0.9000    | 0.2172   | 0.3000  |
| 3   | 0.0005 | 128 | ✅   | ✅     | 16 | 0.0536     | 1.0000    | 0.1794   | 0.7000  |
| 4   | 0.0005 | 512 | ✅   | ✅     | 16 | —          | —         | —        | —       |

### Proposta raonada en base a l’anàlisi dels resultats obtinguts

Després de diverses iteracions experimentals i anàlisi qualitativa dels resultats, hem acabat definint una arquitectura 
final que millora significativament la relació entre text i imatge. Aquesta proposta es basa en un conjunt de millores 
implementades que han estat seleccionades segons el seu impacte en la qualitat dels resultats obtinguts.

#### Millores aplicades a l'arquitectura final

- **Ús de GTR (Google Text Retrieval)**  
  El model GTR està dissenyat específicament per a tasques de recuperació multimodal, on text i imatge comparteixen un 
  espai d’embeddings optimitzat conjuntament. Això ha permès millorar clarament la qualitat del matching entre panells i 
  descripcions, especialment en escenes complexes.

- **Limitació del text a les 100 primeres paraules, excloent la primera frase**  
  Hem observat que la informació més útil per identificar l’escena es troba habitualment a l’inici de la descripció, 
  exceptuant la primera frase, que sol ser bastant repetitiva i sol introduir informació contextual o històrica que no contribueix a la tasca de matching. 
  Per això, hem aplicat una estratègia que conserva el principi de la descripció (100 primeres paraules), però elimina 
  la primera frase, que acostuma a contenir informació irrellevant. Això ajuda a centrar el model en la part significativa 
  del text, reduint el soroll semàntic.
- **Normalització dels embeddings**  
  Tant els vectors de text com els d’imatge es normalitzen per estar en el mateix espai i fer que la comparació mitjançant 
  producte escalar o cosinus sigui coherent.

#### Propostes descartades o no implementades finalment

- **ResNet18 en comptes de ResNet50**  
  Tot i provar una arquitectura més lleugera per evitar l’overfitting, es va perdre capacitat de representació i es van 
  obtenir pitjors resultats tant qualitativament com quantitativament.

- **Models sense fine-tuning**  
  Deixar congelades totes les capes de la xarxa visual no ha permès adaptar-se al domini específic dels còmics, limitant 
  la qualitat dels embeddings visuals.

- **Sentence-BERT**  
  Malgrat millorar respecte a BERT-base, Sentence-BERT no ha assolit els resultats que proporciona GTR.

#### Conclusió

El millor model obtingut ha estat el que combina **GTR**, la **supressió de la primera frase del text** i la **limitació 
a les 100 paraules següents**. Aquesta configuració maximitza la qualitat dels embeddings multimodals, ofereix millors
resultats qualitatius en la recuperació de panells i presenta un comportament més coherent en presència de distractors similars. 

| Aspecte / Proposta                                             | Descripció i Raonament                                                                                              | Resultats obtinguts                                                        | Estat             |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | ----------------- |
| **Ús de GTR (Google Text Retrieval)**                          | Model específic per a recuperació multimodal, que optimitza conjuntament embeddings de text i imatge.               | Millora clara en qualitat del matching, especialment en escenes complexes. | Aplicat i efectiu |
| **Limitació del text a 100 paraules excloent la 1a frase**     | La primera frase sovint aporta info contextual no rellevant; eliminant-la es centra el model en la descripció útil. | Millora en la precisió i coherència de la recuperació.                     | Aplicat i efectiu |
| **Normalització dels embeddings**                              | S’assegura la comparació justa amb mètrica cosinus per vectors de text i imatge.                                    | Comparació més consistent i fiable.                                        | Aplicat i efectiu |
| **ResNet18 en comptes de ResNet50**                            | Intent per evitar overfitting usant arquitectura més lleugera.                                                      | Pitjor capacitat de representació i resultats inferiors.                   | Descartat         |
| **Models sense fine-tuning**                                   | Deixar congelades les capes visuals per estalviar entrenament.                                                      | No s’adapta bé al domini, pitjor qualitat d’embeddings visuals.            | Descartat         |
| **Sentence-BERT sense GTR**                                    | Millora sobre BERT-base, però sense cohesió multimodal tan bona com GTR.                                            | Menys efectiu en contextos amb distractors visuals similars.               | Descartat         |

Resultats amb totes les dades:

Abans de fer la cerca d'hiperparàmetres, hem fet una prova amb totes les dades, amb batch de 32 i 5 èpoques. Aquesta prova ens ha resultat molt lenta i, per aquest motiu, no l’hem pogut repetir. Tot i així, hem obtingut els gràfics de loss i accuracy del train i de la validació.

En el train, els resultats han tingut sentit, ja que la loss ha anat baixant i l’accuracy ha anat pujant fins al 72%. En canvi, en la validació, la loss ha anat pujant i baixant sense una tendència clara, i l’accuracy ha disminuït lleugerament fins al 52%. Això pot ser degut a que no hem pogut fer masses èpoques.

Resultats de la cerca d’hiperparàmetres:

Per intentar millorar el model, hem fet una cerca d’hiperparàmetres provant diferents combinacions. L'hem feta amb poques mostres, ja que fer-la amb totes les dades ens ha semblat inviable pel temps que requeria. 

Les combinacions provades han sigut:

** Ces donnees ne sont pas publiques, ne pas partager en dehors du corbeil lab **

#####

Personnes a contacter pour des questions :
Thibaud Godon, thibaud.godon.1@ulaval.ca : mise en ligne des donnees, script get_data.py, analyses Random Forest et RandomSCM
Jacques Corbeil, jacques.corbeil@fmed.ulaval.ca : coordinateur de l'etude, acquisition de la proteomique
Francis Briere, francis.briere.1@ulaval.ca : acquisition de la metabolomique

#####

metadata.csv :
100 patientes (seulement des femmes)
	50 avec covid long
	50 sans covid
informations cliniques
detail des symptomes

metabolomics.csv :
Francis Briere, 31/03/2021
normalisation qui utilise les qc
les 100 patients décrits dans metadata.csv et quelques autres (environ 50) dont le label n'est pas connu

proteomics.csv :
Jacques Corbeil, 09/04/2021

proteomics_cyt.csv :
Jacques Corbeil, 09/04/2021
comme proteomics.csv mais dans ce cas les proteines sont des cytokines

get_data.py
Contient du code utile pour extraire des matrices X et y pour faire du machine learning
Bonne pratique : ne pas modifier get_data.py, faire une copie pour une utilisation personnalisee

#####

Informations sur les resultats d'algorithmes de machine learning:
Random Forest et RandomSCM:
	bonne perfs de prediction sur la metabolomique : 
	               random_forest  random_scm
Matthews CC           0.8901      0.9316
accuracy              0.9441      0.9647
f1 score              0.9443      0.9649
roc_auc_score         0.9890      0.9970

	pas d'information utile trouvee dans la proteomique:
                   random_forest  random_scm
Matthews CC           0.1581      0.0842
accuracy              0.5647      0.5382
f1 score              0.5247      0.4861
roc_auc_score         0.5596      0.5322

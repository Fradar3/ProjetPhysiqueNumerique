# Projet : Modélisation de Phénomènes Physiques par Automates Cellulaires

**Cours :** PHY-3500 - Physique Numérique
**Équipe :** [Vos Noms]
**Date :** [Date]

---

## I. Rapport Écrit (Format Article Scientifique, Max 15 pages)

**(Objectif : Présenter une démarche de recherche rigoureuse, de la théorie à l'analyse numérique, ancrée dans des problèmes scientifiques pertinents.)**

**1. Page Titre**
   - Titre du projet
   - Noms des auteurs
   - Affiliation (PHY-3500, Université Laval)
   - Date

**2. Résumé (Abstract)**
   - Contexte : Les automates cellulaires (AC) comme outils de modélisation discrets pour systèmes complexes spatio-temporels.
   - Problème : Application des AC à la modélisation de [Application 1, e.g., la percolation dans les feux de forêt] et [Application 2, e.g., la transition de phase dans le modèle d'Ising 2D].
   - Méthodes : Présentation du cadre numérique des AC (grille 2D, états, voisinages, règles de transition synchrones). Description des modèles AC spécifiques pour chaque application (états, règles probabilistes/déterministes, paramètres clés). Implémentation en Python avec NumPy et Matplotlib.
   - Résultats Clés : Mise en évidence du seuil de percolation pour [Application 1]. Observation de la transition de phase et estimation de la température critique pour [Application 2].
   - Conclusion : Validation de l'utilité des AC pour capturer qualitativement et quantitativement des phénomènes physiques essentiels. Discussion des avantages et limites.

**3. Introduction**
   - Définition formelle des automates cellulaires (espace, temps, états discrets, localité, synchronie).
   - Bref historique : Mention de von Neumann, Ulam, Wolfram. *Mentionner Conway et le Jeu de la Vie comme exemple historique célèbre d'émergence, mais préciser que notre focus est sur des applications physiques directes.*
   - Motivation : Pourquoi utiliser les AC en physique numérique ? (Modélisation de systèmes étendus, comportement émergent à partir de règles simples, exploration de non-linéarités et transitions de phase, alternative à la résolution de PDEs dans certains cas).
   - Problématique et Objectifs : Démontrer comment les AC peuvent modéliser et permettre l'étude numérique de phénomènes physiques comme [Application 1 : e.g., la percolation] et [Application 2 : e.g., les transitions de phase magnétiques]. Objectifs : implémenter les modèles, réaliser des simulations, analyser quantitativement les résultats (seuils, paramètres d'ordre), visualiser les dynamiques spatiales.
   - Structure du rapport.

**4. Méthodes Numériques**
   - **4.1 Cadre Général des Automates Cellulaires 2D :**
      - Grille (lattice) : Définition (N x M), type (carrée).
      - États : Encodage numérique (entiers, booléens).
      - Voisinage : Définition (von Neumann, Moore - *justifier le choix, e.g., Moore pour isotropie*).
      - Conditions aux Limites : Types (périodiques, fixes/absorbantes - *justifier le choix en fonction du problème simulé*). Implémentation (e.g., modulo N pour périodique, padding pour fixe).
      - Règle de Transition : Fonction `Update(cellule, voisinage) -> nouvel_état`.
      - Dynamique Temporelle : Mise à jour synchrone (nécessité de deux grilles ou copie).
   - **4.2 Modèle Spécifique 1 : [e.g., Feu de Forêt]**
      - Description du phénomène physique (propagation, seuil).
      - États : 0 (Vide), 1 (Arbre), 2 (Feu).
      - Règles : Explicites (e.g., Feu->Vide; Arbre->Feu si voisin Feu; Vide->Arbre avec proba `p_growth`? Arbre sans feu reste Arbre). *Justifier les règles par rapport à la physique simplifiée du feu.*
      - Paramètres : Densité initiale d'arbres `p`.
   - **4.3 Modèle Spécifique 2 : [e.g., Modèle d'Ising 2D]**
      - Description du phénomène physique (ferromagnétisme, transition de phase).
      - États : +1 (spin up), -1 (spin down).
      - Hamiltonien : `H = -J Σ<i,j> s_i s_j`.
      - Règles (Algorithme de Metropolis) :
         - Choisir un spin aléatoire `s_i`.
         - Calculer le changement d'énergie `ΔE` si on le retourne (`s_i -> -s_i`).
         - Si `ΔE < 0`, retourner le spin.
         - Si `ΔE >= 0`, retourner le spin avec probabilité `exp(-ΔE / k_B T)`. (*Ici k_B peut être absorbé dans T*).
      - Paramètres : Constante de couplage `J` (souvent `J=1`), Température `T`.
   - **4.4 Implémentation :**
      - Langage : Python.
      - Bibliothèques : NumPy (opérations sur tableaux), Matplotlib (visualisation statique/animations), `random` (pour les aspects probabilistes).
      - Pseudo-code de la boucle principale de simulation.
      - Mention des aspects d'efficacité (vectorisation partielle si utilisée).

**5. Résultats**
   - Présentation structurée par application.
   - **5.1 Résultats pour [Application 1 : e.g., Feu de Forêt]**
      - Visualisations typiques (snapshots à différents temps, ou référence à une animation). Montrer la propagation pour `p` sous et au-dessus du seuil.
      - Analyse quantitative : Graphe de la fraction de la forêt brûlée vs. densité initiale `p`. Identification claire du seuil de percolation `p_c`.
      - Interprétation physique des observations (transition abrupte au seuil).
   - **5.2 Résultats pour [Application 2 : e.g., Modèle d'Ising 2D]**
      - Visualisations typiques : Snapshots de la configuration des spins à différentes températures (basse T: domaines larges, haute T: aléatoire, près de Tc: fluctuations importantes).
      - Analyse quantitative :
         - Graphe de la magnétisation moyenne `M = |<Σ s_i>| / N²` en fonction de la température `T` (après équilibration). Identification de la température critique `T_c`.
         - (Optionnel) Graphe de la susceptibilité magnétique ou de la chaleur spécifique pour mieux localiser `T_c`.
      - Interprétation physique (transition de phase ordre-désordre).

**6. Discussion et Conclusion**
   - Synthèse des résultats clés : confirmation de la capacité des AC à reproduire des phénomènes comme [la percolation] et [les transitions de phase].
   - Analyse critique :
      - Points forts des AC pour ces problèmes (capture de la spatialité, émergence, simplicité relative d'implémentation).
      - Limites observées (effets de taille finie, sensibilité aux conditions initiales/paramètres, artefacts de grille potentiels, temps de simulation pour l'équilibre dans Ising).
   - Lien avec la Physique Numérique : Les AC comme un paradigme de simulation complémentaire aux méthodes continues (EDP). Utilité pour l'exploration et la compréhension qualitative/quantitative de systèmes complexes.
   - Perspectives (si pertinent) : Raffinement des modèles, autres applications, comparaison avec d'autres méthodes.
   - Conclusion générale sur la pertinence et l'intérêt du sujet dans le cadre du cours.

**7. Bibliographie**
   - Liste des articles, livres, ressources web utilisés. Format cohérent.

**(Annexe : Code source principal (si demandé et pertinent, sinon juste soumis séparément))**

---

## II. Présentation Orale (15 minutes + 5 min Questions)

**(Objectif : Communiquer clairement la démarche, les résultats clés et la pertinence du projet à un public d'étudiants en physique numérique. Focus sur la visualisation et la synthèse.)**

**Diapo 1 : Titre (0.5 min)**
   - Comme rapport. Présentation rapide de l'équipe.

**Diapo 2 : Introduction (2 min)**
   - C'est quoi un AC ? (Définition visuelle simple : grille, états, voisins, règles).
   - Pourquoi en Physique Numérique ? (Modéliser des systèmes complexes spatialement étendus, voir l'émergence).
   - Notre Objectif : Utiliser les AC pour étudier [Phénomène 1] et [Phénomène 2].
   - Plan de la présentation.

**Diapo 3-4 : Méthode AC Générale (3 min)**
   - Schéma de la Grille 2D, États, Voisinage (Moore/Von Neumann - *justifier*), Conditions Limites (*justifier*).
   - Algorithme principal (boucle de temps, mise à jour synchrone). *Visuel !*
   - Outils : Python, NumPy, Matplotlib.

**Diapo 5 : Application 1 - [e.g., Feu de Forêt] : Modèle (1.5 min)**
   - Problème : Modéliser la propagation d'un feu et le seuil de percolation.
   - Modèle AC : États (visuel : couleur vide, vert, rouge), Règles (animation simple d'une cellule qui change d'état). Paramètre clé `p`.

**Diapo 6 : Application 1 - Résultats (2 min)**
   - **Animation clé** montrant la propagation sous et au-dessus du seuil.
   - **Graphe clé :** Fraction brûlée vs. `p`. Mettre en évidence le seuil `p_c`.
   - Interprétation : Transition abrupte = phénomène physique de percolation.

**Diapo 7 : Application 2 - [e.g., Modèle d'Ising 2D] : Modèle (1.5 min)**
   - Problème : Modéliser la transition ferromagnétique.
   - Modèle AC : États (visuel : 2 couleurs pour +/- spins). Règle de Metropolis (expliquer l'idée : accepter les changements favorables, accepter les défavorables avec proba `exp(-ΔE/T)`). Paramètre clé `T`.

**Diapo 8 : Application 2 - Résultats (2 min)**
   - **Visualisation clé :** Snapshots de configurations à basse T (ordonné), haute T (désordonné), près de T_c (fluctuations).
   - **Graphe clé :** Magnétisation `M` vs. `T`. Mettre en évidence `T_c`.
   - Interprétation : Transition de phase ordre-désordre.

**Diapo 9 : Discussion (2 min)**
   - Ce qu'on a vu : Les AC simples capturent bien [la percolation] et [les transitions de phase].
   - Avantages montrés ici : Visualisation directe, lien règles locales <-> comportement global.
   - Limites rencontrées : Temps de calcul (Ising), effets de taille finie.

**Diapo 10 : Conclusion (1 min)**
   - Les AC sont un outil puissant et flexible en physique numérique pour explorer des systèmes complexes spatiaux.
   - Notre travail a illustré ceci pour [Phénomène 1] et [Phénomène 2].
   - Remerciements (prof, coéquipiers).

**Diapo 11 : Questions ?**
   - Afficher un visuel intéressant (e.g., une belle config d'Ising ou un front de feu) pendant les questions.

**(Prévoir ~1 min de marge / transitions)**

---

Ce plan détaillé devrait vous fournir une base solide pour organiser votre travail et vos livrables. N'oubliez pas d'adapter les applications spécifiques ([...]) selon votre choix final !
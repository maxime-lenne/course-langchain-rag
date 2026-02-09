# RAG (Retrieval-Augmented Generation) - LangChain

![LangChain](img/langchain.jpeg)

Les systèmes **RAG (Retrieval-Augmented Generation)** dans LangChain permettent aux modèles de langage de s'appuyer sur des **connaissances externes** pour produire des réponses plus précises, actualisées et pertinentes.

Contrairement à un simple LLM qui génère une réponse uniquement à partir de ce qu'il a appris pendant son entraînement, un système RAG interroge une base de documents pour retrouver des morceaux d'information pertinents – appelés **chunks** – et les injecte dans le prompt du LLM.

![RAG](img/rag.jpeg)

**Que montre le schéma ci-dessus ?**

Le processus se divise en **deux grandes phases** : **préparation des documents** et **traitement des requêtes**.

**Préparation des documents (à gauche)**
- (1) Un fichier (document source) est divisé en **chunks**, c'est-à-dire en petits segments de texte.
- (2) Chaque chunk est passé dans un LLM Embedder, un encodeur qui transforme le texte en un vecteur numérique (**embeddings**).
- (3) Ces vecteurs sont ensuite stockés dans un Vector Store, une base de données spécialisée pour les recherches par **similarité sémantique**.

**Traitement des requêtes (à droite)**
- (a) Lorsqu'un utilisateur emet une requête, celle-ci est à son tour encodée via **le même LLM Embedder** pour obtenir son vecteur.
- (b) Ce vecteur est utilisé par le **Retriever**, qui compare la requête aux vecteurs des **chunks** pour trouver les plus similaires.
- (c) Les chunks retrouvés sont envoyés au LLM, qui les utilise comme contexte pour formuler une réponse.


En résumé, ce fonctionnement est illustré par la boucle :

> Requête → Encodage → Recherche dans la base vectorielle → Récupération des chunks → Passage au LLM → Réponse contextuelle

## 1. Chargement du modèle LLM local

Dans cette section, nous chargeons un modèle de langage local grâce à **Ollama**. Cela permet de travailler avec un **LLM directement sur notre machine**, sans connexion à une API externe.

Nous utilisons ici la classe `ChatOllama` de **LangChain**, qui nous permet d'interagir facilement avec un modèle comme **llama3** ainsi qu'un **modèle d'embeddings** déjà téléchargés via Ollama.

## 2. RAG standard

Le **RAG standard** consiste à :
- formuler une requête explicite
- interroger une base de documents vectorisée
- utiliser un modèle LLM pour générer une réponse à partir des résultats retrouvés.

Ce pipeline est **efficace pour des questions indépendantes, sans contexte conversationnel**.

### 2.1 Préparation des documents

Nous initialisons les chemins nécessaires à la préparation des documents d'entrée.

### 2.2 Initialisation du vector store

Nous vérifions ici si la base vectorielle existe déjà.
Si ce n'est pas le cas, le fichier source est chargé, découpé en morceaux, enrichi de métadonnées, puis indexé dans Chroma DB.

### 2.3 Initialisation du moteur de recherche vectorielle

Une fois la base vectorielle Chroma initialisée avec les embeddings, nous la transformons en **moteur de recherche (retriever)**.
Cela permet de retrouver les documents les plus proches sémantiquement d'une question ou d'une requête.

### 2.4 Exécution d'une requête de recherche

Dans cette étape, nous combinons la recherche vectorielle avec un LLM.
L'objectif est de fournir une réponse pertinente à une question, en s'appuyant uniquement sur les documents retrouvés dans la base vectorielle.
Le modèle est guidé par un prompt structuré qui inclut la requête initiale et les contenus des chunks pertinents.

### 🧩 Exercice

La société NovTech gère de nombreux documents internes :
- des rapports d'incidents (panne, erreur technique, post-mortem),
- des procédures opérationnelles (onboarding, accès système, déploiement…).

Actuellement, les équipes perdent du temps à chercher les bonnes informations à travers des fichiers éparpillés.

Votre objectif est de construire un assistant basé sur l'architecture RAG qui permettra :
- de retrouver rapidement les procédures en cas de besoin,
- de consulter les résolutions d'incidents similaires,
- de répondre à des questions en langage naturel en s'appuyant uniquement sur les documents internes.

Pour vous aider, vous pouvez suivre les étapes suivantes :
1. Chargement des documents
2. Découpage en chunks
3. Indexation vectorielle
4. Recherche contextuelle
5. Génération de réponse

ℹ️ Les documents de l'entreprise se trouve dans le dossier `data/novtech`.
💪🏻 **Bonus** : Rendre possible un filtrage par catégorie dans les recherches

## 3. RAG conversationnel

Dans un cadre d'**interaction continue**, les utilisateurs posent souvent des questions implicites ou référentielles (ex. "Et lui ?"). Le **RAG conversationnel** ajoute une étape clé : la **reformulation de la question en prenant en compte l'historique du dialogue**.

Cette version de RAG permet de maintenir la pertinence des recherches dans la base vectorielle tout en conservant la fluidité de la conversation, ce qui la rend adaptée aux assistants IA ou aux chatbots avancés.

**Exemple**

Historique de la conversation :
- Utilisateur : *Qui est le CEO de Tesla ?*
- IA : *Elon Musk est le CEO de Tesla*.
- Utilisateur : *Et de SpaceX ?*

➡️ La question "Et de SpaceX ?" est ambiguë seule. Le moteur de recherche (retriever) ne sait pas de quoi il s'agit exactement.

Avec une reformulation de la question de l'utilisateur cela donnerait : "Qui est le CEO de SpaceX ?"

➡️ Résultat : la requête est claire, et la recherche dans la base vectorielle peut retourner les bons documents.

**👍 LangChain facilite ce processus**

LangChain fournit une abstraction prête à l'emploi grâce à la classe `ConversationalRetrievalChain`.
Cette classe prend automatiquement en charge :
- la reformulation de la question via le LLM
- la recherche dans la base vectorielle
- la génération de la réponse finale à partir des documents récupérés et de l'historique

➡️ Elle encapsule ainsi toute la logique conversationnelle d'un RAG en une seule ligne.

### 🧩 Exercice

Repartez de l'exercice précédent (NovTech), et implémentez un assistant de conversation continue.

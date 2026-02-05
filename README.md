![LangChain](img/langchain.jpeg)

Ce projet a pour but de proposer une démonstration pédagogique de l'utilisation de **LangChain**
pour construire une application de type **RAG (Retrieval-Augmented Generation)**.
Il est destiné à un public souhaitant apprendre à combiner LLMs et recherche de documents
afin de créer des agents capables de répondre à partir de sources personnalisées.

## Objectifs pédagogiques

- Comprendre les bases du RAG (Retrieval-Augmented Generation)  
- Utiliser LangChain pour construire une chaîne de traitement simple  
- Interagir avec un LLM à partir de documents personnels  
- Explorer les étapes d'indexation, de recherche, et de génération de réponses  
- Fournir une base pour la création d'applications plus complexes  

## Prérequis

- Python ≥ 3.10  
- Environnement Jupyter (ou autre IDE compatible)  
- Clé API OpenAI (ou autre LLM compatible)  

## Installation

1. Clonez le dépôt :

   ```bash
   git clone https://github.com/maxime-lenne/course-langchain-rag     
   cd rag
   ```

2. Créez un environnement virtuel (optionnel mais recommandé) :

   ```bash
   python -m venv venv
   source venv/bin/activate  # ou .\venv\Scripts\activate sous Windows
   ```

3. Installez les dépendances :

   ```bash
   pip install -r requirements.txt
   ```

4. Lancez le notebook :

   ```bash
   jupyter notebook
   ```

## Structure du projet

- `langchain.ipynb` : notebook principal avec les démonstrations pédagogiques  
- `data/` : dossiers de documents à indexer  
- `img/` : illustrations ou schémas pour la présentation  
- `requirements.txt` : dépendances Python à installer  

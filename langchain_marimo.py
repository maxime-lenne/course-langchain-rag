import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(mo):
    mo.md(
        """
        ![LangChain](img/langchain.jpeg)

        # RAG (Retrieval-Augmented Generation) avec LangChain

        Les systèmes **RAG** permettent aux modèles de langage de s'appuyer sur des
        **connaissances externes** pour produire des réponses plus précises et pertinentes.

        Contrairement à un simple LLM, un système RAG interroge une base de documents
        pour retrouver des morceaux d'information pertinents — appelés **chunks** —
        et les injecte dans le prompt du LLM.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        ![RAG](img/rag.jpeg)

        ## Processus RAG

        **Préparation des documents :**
        1. Document → **Chunks** (découpage)
        2. Chunks → **Embeddings** (vectorisation)
        3. Embeddings → **Vector Store** (stockage)

        **Traitement des requêtes :**
        1. Requête → **Embedding**
        2. Recherche par **similarité** dans le Vector Store
        3. Chunks pertinents + Requête → **LLM** → Réponse
        """
    )
    return


@app.cell
def __(mo):
    mo.md("# 1. Chargement du modèle LLM et Embedder")
    return


@app.cell
def __():
    import os
    from dotenv import load_dotenv
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import TextLoader
    from langchain_community.vectorstores import Chroma
    from langchain.chains import ConversationalRetrievalChain

    load_dotenv(override=True)

    model = ChatOllama(model="llama3", temperature=0)
    embedder = OllamaEmbeddings(model="nomic-embed-text")
    return (
        os, load_dotenv, model, embedder, ChatOllama, OllamaEmbeddings,
        SystemMessage, HumanMessage, RecursiveCharacterTextSplitter,
        TextLoader, Chroma, ConversationalRetrievalChain
    )


@app.cell
def __(mo):
    mo.md(
        """
        # 2. RAG standard

        Le RAG standard consiste à :
        - Formuler une requête explicite
        - Interroger une base de documents vectorisée
        - Utiliser un LLM pour générer une réponse

        Ce pipeline est efficace pour des **questions indépendantes, sans contexte conversationnel**.
        """
    )
    return


@app.cell
def __(mo):
    mo.md("### 2.1 Préparation des documents")
    return


@app.cell
def __(os):
    # Chemins vers les fichiers
    current_dir = os.getcwd()
    file_name = "meeting_reports.txt"
    file_path = os.path.join(current_dir, "data", file_name)
    db_dir = os.path.join(current_dir, "data", "db")
    return current_dir, file_name, file_path, db_dir


@app.cell
def __(mo):
    mo.md("### 2.2 Initialisation du vector store")
    return


@app.cell
def __(os, mo, file_path, db_dir, embedder, TextLoader, RecursiveCharacterTextSplitter, Chroma):
    # Vérification si la base existe déjà
    if not os.path.exists(db_dir):
        mo.md("**Initialisation du vector store...**")

        # Chargement du document
        loader = TextLoader(file_path)
        loaded_document = loader.load()

        # Découpage en chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        chunks = text_splitter.split_documents(loaded_document)

        # Ajout de métadonnées
        for chunk in chunks:
            chunk.metadata["source"] = file_path
            chunk.metadata["category"] = "meeting"

        # Création de la base vectorielle
        db = Chroma.from_documents(chunks, embedder, persist_directory=db_dir)
        mo.md("**Vector store créé !**")
    else:
        mo.md("**Vector store existant chargé.**")
    return


@app.cell
def __(mo):
    mo.md("### 2.3 Initialisation du moteur de recherche")
    return


@app.cell
def __(db_dir, embedder, Chroma):
    # Chargement de la base vectorielle
    db = Chroma(persist_directory=db_dir, embedding_function=embedder)

    # Conversion en retriever
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    return db, retriever


@app.cell
def __(mo):
    mo.md("### 2.4 Interface de recherche")
    return


@app.cell
def __(mo):
    # Interface utilisateur pour la requête
    query_input = mo.ui.text(
        value="Quels sont les réunions concernant la société Neolink ?",
        placeholder="Entrez votre question...",
        label="Question"
    )
    query_input
    return (query_input,)


@app.cell
def __(mo, model, retriever, query_input, SystemMessage, HumanMessage):
    if query_input.value:
        # Recherche des chunks pertinents
        relevant_chunks = retriever.invoke(query_input.value)

        # Construction du message
        input_message = (
            "Voici des documents qui vont t'aider à répondre à la question : "
            + query_input.value
            + "\n\nDocuments pertinents : \n"
            + "\n\n".join([chunk.page_content for chunk in relevant_chunks])
            + "\n\nDonne une réponse basée uniquement sur les documents fournis."
        )

        messages = [
            SystemMessage(content="Tu es un assistant qui aide à retrouver des informations internes."),
            HumanMessage(content=input_message)
        ]

        result = model.invoke(messages)
        mo.md(f"""
        **Question :** {query_input.value}

        ---

        **Réponse :**

        {result.content}
        """)
    return


@app.cell
def __(mo):
    mo.md(
        """
        ## Exercice 1

        La société **NovTech** gère de nombreux documents internes :
        - Rapports d'incidents (`data/novtech/incidents/`)
        - Procédures opérationnelles (`data/novtech/procedures/`)

        Construisez un assistant RAG qui permet de :
        - Retrouver rapidement les procédures
        - Consulter les résolutions d'incidents similaires
        - Répondre en langage naturel

        **Bonus** : Permettre un filtrage par catégorie (incidents vs procédures)
        """
    )
    return


@app.cell
def __(mo):
    # Votre code ici
    mo.md("Complétez l'exercice ci-dessus")
    return


@app.cell
def __(mo):
    mo.md(
        """
        # 3. RAG conversationnel

        Le **RAG conversationnel** ajoute une étape clé : la **reformulation de la question**
        en prenant en compte l'historique du dialogue.

        **Exemple :**
        - Utilisateur : *Qui est le CEO de Tesla ?*
        - IA : *Elon Musk*
        - Utilisateur : *Et de SpaceX ?*

        → La question "Et de SpaceX ?" est reformulée en "Qui est le CEO de SpaceX ?"
        """
    )
    return


@app.cell
def __(mo):
    # État pour l'historique de conversation
    chat_history_state = mo.state([])
    return (chat_history_state,)


@app.cell
def __(mo):
    chat_query = mo.ui.text(
        placeholder="Posez une question...",
        label="Votre question"
    )
    chat_query
    return (chat_query,)


@app.cell
def __(mo, model, db, chat_query, ConversationalRetrievalChain):
    # Chaîne RAG conversationnelle
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=db.as_retriever()
    )

    # Historique simple (non persistant entre les cellules pour la démo)
    chat_history = []

    if chat_query.value:
        result_chat = qa_chain({"question": chat_query.value, "chat_history": chat_history})
        mo.md(f"""
        **Vous :** {chat_query.value}

        **Assistant :** {result_chat["answer"]}
        """)
    else:
        mo.md("*Posez une question pour commencer*")
    return qa_chain, chat_history


@app.cell
def __(mo):
    mo.md(
        """
        ## Exercice 2

        Repartez de l'exercice NovTech et implémentez un assistant de conversation continue
        avec `ConversationalRetrievalChain`.

        L'assistant doit pouvoir :
        - Se souvenir des questions précédentes
        - Reformuler automatiquement les questions implicites
        """
    )
    return


@app.cell
def __(mo):
    # Votre code ici
    mo.md("Complétez l'exercice ci-dessus")
    return


if __name__ == "__main__":
    app.run()

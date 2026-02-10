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
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables import RunnablePassthrough
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import TextLoader
    from langchain_chroma import Chroma  # ✅ LangChain v1 : pip install langchain-chroma

    load_dotenv(override=True)

    model = ChatOllama(model="llama3", temperature=0)
    embedder = OllamaEmbeddings(model="nomic-embed-text")
    return (
        os, load_dotenv, model, embedder,
        ChatOllama, OllamaEmbeddings,
        HumanMessage, AIMessage,
        StrOutputParser, ChatPromptTemplate, MessagesPlaceholder, RunnablePassthrough,
        RecursiveCharacterTextSplitter, TextLoader, Chroma,
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
    mo.md(
        """
        ### 2.4 Exécution d'une requête de recherche

        Le pipeline est composé de 4 étapes enchaînées avec l'opérateur `|` :
        1. **Récupération** : le retriever cherche les chunks pertinents et une fonction les formate
        2. **Prompt** : le contexte et la question sont injectés dans un template structuré
        3. **LLM** : le modèle génère une réponse à partir du prompt enrichi
        4. **Parser** : la sortie brute est convertie en chaîne de caractères simple
        """
    )
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
def __(mo, model, retriever, query_input, ChatPromptTemplate, RunnablePassthrough, StrOutputParser):
    if query_input.value:
        # Fonction de formatage des documents
        def _format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Template du prompt
        _prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Tu es un assistant qui aide à retrouver tout type d'informations interne à une entreprise. "
             "Réponds uniquement en te basant sur les documents fournis. "
             "Si l'information n'est pas dans les documents, dis-le clairement."),
            ("human", "Documents pertinents :\n\n{context}\n\nQuestion : {question}")
        ])

        # Chaîne RAG avec LCEL
        rag_chain = (
            {"context": retriever | _format_docs, "question": RunnablePassthrough()}
            | _prompt
            | model
            | StrOutputParser()
        )

        result = rag_chain.invoke(query_input.value)

        mo.md(f"""
        **Question :** {query_input.value}

        ---

        **Réponse :**

        {result}
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

        Le **RAG conversationnel** maintient un historique de la conversation pour
        permettre des questions de suivi implicites (ex. "Et lui ?").

        **Approche LCEL avec historique explicite :**
        On utilise un `MessagesPlaceholder` dans le prompt pour injecter l'historique
        (`chat_history`) à chaque tour. La liste est mise à jour manuellement
        après chaque échange (`HumanMessage` / `AIMessage`).

        > 💡 Pour une mémoire persistante multi-sessions, voir `rag.ipynb`.
        """
    )
    return


@app.cell
def __(mo):
    # État pour l'historique de conversation (persistant entre les re-exécutions Marimo)
    get_history, set_history = mo.state([])
    return get_history, set_history


@app.cell
def __(mo):
    chat_query = mo.ui.text(
        placeholder="Posez une question...",
        label="Votre question"
    )
    chat_query
    return (chat_query,)


@app.cell
def __(mo, model, retriever, chat_query, get_history, set_history,
       ChatPromptTemplate, MessagesPlaceholder, StrOutputParser,
       HumanMessage, AIMessage):
    if chat_query.value:
        # Prompt avec historique de conversation
        _prompt_conv = ChatPromptTemplate.from_messages([
            ("system",
             "Tu es un assistant qui aide à retrouver tout type d'informations interne à une entreprise. "
             "Réponds uniquement en te basant sur les documents fournis. "
             "Si l'information n'est pas dans les documents, dis-le clairement."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "Documents pertinents :\n\n{context}\n\nQuestion : {question}")
        ])

        # Chaîne RAG conversationnelle avec LCEL
        _chain = (
            {
                "context": lambda x: "\n\n".join(
                    doc.page_content for doc in retriever.invoke(x["question"])
                ),
                "question": lambda x: x["question"],
                "chat_history": lambda x: x["chat_history"]
            }
            | _prompt_conv
            | model
            | StrOutputParser()
        )

        _result = _chain.invoke({
            "question": chat_query.value,
            "chat_history": get_history()
        })

        # Mise à jour de l'historique
        _new_history = get_history() + [
            HumanMessage(content=chat_query.value),
            AIMessage(content=_result)
        ]
        set_history(_new_history)

        mo.md(f"""
        **Vous :** {chat_query.value}

        **Assistant :** {_result}
        """)
    else:
        mo.md("*Posez une question pour commencer*")
    return


@app.cell
def __(mo):
    mo.md(
        """
        ## Exercice 2

        Repartez de l'exercice NovTech et implémentez un assistant de conversation continue
        avec la chaîne LCEL + `MessagesPlaceholder`.

        L'assistant doit pouvoir :
        - Se souvenir des questions précédentes
        - Utiliser le contexte de l'historique pour des questions de suivi implicites
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

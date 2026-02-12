import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="PokéChati AI", page_icon="context")
st.title("🔴 PokéChati: Tu experto Pokémon")

# 1. Configuración de la API Key (Moved to sidebar)
with st.sidebar:
    st.header("Configuración")
    groq_api_key = st.text_input("Ingresa tu Clave API de Groq:", type="password")
    st.markdown("Obtén tu clave API de Groq aquí: [https://console.groq.com/keys](https://console.groq.com/keys)")

if groq_api_key:
    # 2. DEFINIR LA PERSONALIDAD (Now part of the prompt template)
    # The persona will be directly embedded into the system message of the ChatPromptTemplate

    # 3. Inicializar el modelo ChatGroq
    llm = ChatGroq(api_key=groq_api_key, model="llama-3.3-70b-versatile", temperature=0.7)

    # Initialize embeddings and vectorstore in session state
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None

    # 4. Document Uploader para contexto extra (RAG) (Moved to sidebar)
    with st.sidebar:
        uploaded_file = st.file_uploader("Sube un documento PDF para contexto adicional (RAG)", type="pdf")

    if uploaded_file is not None:
        # Save uploaded file temporarily to process
        with open("temp_doc.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyMuPDFLoader("temp_doc.pdf")
        doc = loader.load()

        # Ensure we concatenate page content correctly
        full_text = "\n".join([p.page_content for p in doc])

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(full_text)

        if chunks:
            with st.spinner("Creando base de datos vectorial..."):
                st.session_state.vectorstore = FAISS.from_texts(chunks, embedding=st.session_state.embeddings)
            st.success(f"Documento procesado. {len(chunks)} fragmentos cargados para el contexto RAG.")
        else:
            st.warning("No se pudo extraer texto del documento. Intenta con otro archivo.")

    # 5. Memoria del chat (para RAG, manejamos el historial explícitamente)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 6. Mostrar historial
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 7. Prompt para RAG with ChatGroq personality
    system_persona = (
        "Eres un experto mundial en Pokémon. Tu nombre es PokéChati. "
        "Respondes de forma entusiasta, usas emojis de Pokémon y conoces "
        "todos los detalles sobre tipos, debilidades, stats base y lore de los juegos. "
        "Si alguien te pregunta algo que no sea de Pokémon, intenta llevar la conversación "
        "de vuelta al mundo Pokémon de forma divertida."
    )
    rag_prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_persona + "\nUtiliza la siguiente información de contexto para responder la pregunta: {context}"),
        ("user", "{question}")
    ])

    # Create the RAG chain
    rag_chain = rag_prompt_template | llm | StrOutputParser()

    # 8. Entrada de usuario
    if prompt := st.chat_input("¿Qué Pokémon quieres investigar?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            context = ""
            if st.session_state.vectorstore is not None:
                retriever = st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
                docs = retriever.invoke(prompt)
                context = "\n\n".join([doc.page_content for doc in docs])
                st.write("Contexto recuperado de documentos.")
                # st.write(f"DEBUG Context: {context[:200]}...") # For debugging

            try:
                # Generate content using the RAG chain
                response_stream = rag_chain.stream({"context": context, "question": prompt})
                for chunk in response_stream:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

            except Exception as e:
                if "429" in str(e) or "rate limit" in str(e).lower():
                    st.error("💤 Los servidores están cansados. Espera un momento (Límite 429 / Rate Limit Exceeded).")
                else:
                    st.error(f"Error: {e}")

            st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.warning("Por favor, introduce tu Clave API de Groq para comenzar.")

    

    

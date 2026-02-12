import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="PokéChati AI", page_icon="context")
st.title("🔴 PokéChati: Tu experto Pokémon")

# 1. Configuración de la API Key
google_api_key = st.text_input("Ingresa tu Clave API de Google (para Gemini):", type="password")

if google_api_key:
    genai.configure(api_key=google_api_key)

    # 2. DEFINIR LA PERSONALIDAD (Instrucción de Sistema)
    instruccion_pokexperto = (
        "Eres un experto mundial en Pokémon. Tu nombre es PokéChati. "
        "Respondes de forma entusiasta, usas emojis de Pokémon y conoces "
        "todos los detalles sobre tipos, debilidades, stats base y lore de los juegos. "
        "Si alguien te pregunta algo que no sea de Pokémon, intenta llevar la conversación "
        "de vuelta al mundo Pokémon de forma divertida." #
    )

    # 3. Inicializar el modelo con la instrucción
    model = genai.GenerativeModel(
        model_name="models/gemini-2.0-flash",
        system_instruction=instruccion_pokexperto
    )

    # Initialize embeddings and vectorstore in session state
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None

    # 4. Document Uploader para contexto extra (RAG)
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

    # 7. Prompt para RAG
    rag_prompt_template = ChatPromptTemplate.from_messages([
        ("system", instruccion_pokexperto + "\nUtiliza la siguiente información de contexto para responder la pregunta: {context}"),
        ("user", "{question}")
    ])

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

            # Construct the RAG prompt
            formatted_prompt = rag_prompt_template.format(context=context, question=prompt)

            try:
                # Generate content using the formatted prompt
                response_stream = model.generate_content(formatted_prompt, stream=True)
                for chunk in response_stream:
                    full_response += chunk.text
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

            except Exception as e:
                if "429" in str(e):
                    st.error("💤 Los servidores están cansados. Espera un momento (Límite 429).")
                else:
                    st.error(f"Error: {e}")

            st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.warning("Por favor, introduce tu Clave API de Google para comenzar.")

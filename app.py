import streamlit as st
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_community.vectorstores import Chroma      # ❌ removed
from langchain_community.vectorstores import FAISS         # ✅ use FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

import csv, os
from dotenv import load_dotenv

def log_interaction(session_id, question, answer, latency):
    with open("interactions.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([session_id, question, answer, latency])

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# Streamlit App
st.title("Conversational RAG with PDF Upload and Chat History")
st.write("Upload a PDF and chat with its content!")

# --- Sidebar ---
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")
session_id = st.sidebar.text_input("Session ID", value="default_session")

# Display chat history in sidebar
if "store" in st.session_state and session_id in st.session_state.store:
    st.sidebar.subheader("Chat History")
    for i, msg in enumerate(st.session_state.store[session_id].messages, 1):
        role = "🧑 User" if msg.type == "human" else "🤖 Assistant"
        st.sidebar.markdown(f"**{role}:** {msg.content}")

# Check if API key is provided
if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-it")

    if "store" not in st.session_state:
        st.session_state.store = {}

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=False)

    if uploaded_file:
        documents = []
        tempPdf = f"temp_{uploaded_file.name}"

        with open(tempPdf, "wb") as file:
            if hasattr(uploaded_file, "getvalue"):
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name
            else:
                file.write(uploaded_file)
                file_name = "uploaded_file.pdf"

        loader = PyPDFLoader(tempPdf)
        docs = loader.load()
        documents.extend(docs)

        # ✅ Use FAISS instead of Chroma
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)

        if not splits:
            st.error("No text could be extracted from the PDF. Please upload a valid PDF.")
        else:
            vectorstore = FAISS.from_documents(splits, embeddings)
            retriever = vectorstore.as_retriever()

            os.remove(tempPdf)  # Clean up temporary file

            contextualize_q_system_prompt = (
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )

            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

            history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your question:")
        if user_input:
            import time
            start_time = time.perf_counter()

            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )

            elapsed = time.perf_counter() - start_time
            st.success(f"Assistant: {response['answer']}")
            st.sidebar.write(f"Response time: {elapsed:.2f} sec")

            feedback = st.radio(
                "Was this answer helpful?",
                ["👍 Yes", "👎 No"],
                index=None,
                key=f"feedback_{session_id}_{user_input}"
            )

            if feedback:
                if feedback == "👎 No":
                    with open("feedback_log.txt", "a", encoding="utf-8") as f:
                        f.write(f"{session_id},{user_input},{response['answer']},{feedback},{elapsed:.2f}\n")
                    log_interaction(session_id, user_input, response["answer"], elapsed)
                else:
                    st.info("🙏 Thank you for your feedback!")
else:
    st.warning("Please enter your Groq API Key in the sidebar.")


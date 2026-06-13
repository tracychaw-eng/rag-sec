import streamlit as st
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from pinecone import Pinecone
import os

# Page config
st.set_page_config(page_title="SEC Filing Chat", page_icon="📄")
st.title("📄 SEC 10-K Filing Chat")
st.caption("Ask questions about MSFT, NVDA, and JPM 10-K filings")

# Initialize once per session
@st.cache_resource
def get_chat_engine():
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.llm = OpenAI(model="gpt-4o-mini")

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    pinecone_index = pc.Index("sec-filings")
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

    return index.as_chat_engine(
        chat_mode="condense_plus_context",
        memory=memory,
        similarity_top_k=6,
        system_prompt=(
        "You are a financial analyst assistant with access to SEC 10-K filings "
        "for Microsoft (MSFT), NVIDIA (NVDA), and JPMorgan Chase (JPM). "
        "Answer questions based on the filing content. "
        "If the context doesn't contain enough information, say so clearly. "
        "Be concise and cite which company's filing you're referencing."
        ),
    )

chat_engine = get_chat_engine()

# Session state for chat history (display only)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about SEC filings..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Searching filings..."):
            response = chat_engine.chat(prompt)
            st.markdown(str(response))

    st.session_state.messages.append({"role": "assistant", "content": str(response)})
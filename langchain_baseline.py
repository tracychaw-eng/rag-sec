from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import os

# Same embedding model the LlamaIndex pipeline used
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Step 1: Connect to EXISTING Pinecone index — no re-indexing
vectorstore = PineconeVectorStore(
    index_name="sec-filings",
    embedding=embeddings,
    pinecone_api_key=os.environ["PINECONE_API_KEY"],
)

# Step 2: Create a retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},  # same top_k as main_pinecone.py
)

# Step 3: Build the RAG chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Prompt template — you control this entirely
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context from SEC 10-K filings.
If the context doesn't contain enough information, say so clearly.

Context:
{context}

Question: {question}

Answer:""")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Format retrieved docs into a single string
def format_docs(docs):
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', doc.metadata.get('file_name', 'unknown'))}]\n{doc.page_content}"
        for doc in docs
    )

# The chain: question → retrieve → format → prompt → LLM → parse
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Step 4: Run the same 5 questions
questions = [
    "What are the main risk factors?",
    "What cybersecurity risks does the company face?",
    "How does competition affect the business?",
    "What regulatory risks are mentioned?",
    "What macroeconomic risks could impact performance?",
]

for i, q in enumerate(questions, 1):
    print(f"\nQ{i}: {q}")
    print("-" * 50)

    # Get retrieved docs for scoring
    docs = retriever.invoke(q)
    for j, doc in enumerate(docs):
        source = doc.metadata.get("source", doc.metadata.get("file_name", "unknown"))
        print(f"  Chunk {j+1} | Source: {source}")

    # Get answer
    answer = chain.invoke(q)
    print(f"\nAnswer: {answer}")

# Step 5: Also capture retrieval scores
print("\n" + "=" * 60)
print("SCORE COMPARISON — LangChain on same Pinecone index")
print("=" * 60)

for i, q in enumerate(questions, 1):
    results = vectorstore.similarity_search_with_score(q, k=3)
    top_score = results[0][1] if results else 0
    sources = set()
    for doc, score in results:
        src = doc.metadata.get("source", doc.metadata.get("file_name", "unknown"))
        sources.add(src)
    print(f"Q{i}   Top Score: {top_score:.3f}    Sources: {', '.join(sources)}")

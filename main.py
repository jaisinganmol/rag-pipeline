"""
RAG Pipeline - Step by Step Implementation with Prompt Printing
=============================================================
This script demonstrates a Retrieval-Augmented Generation (RAG) pipeline
using LangChain, showing each step of the process, including:
1. Document loading and chunking.
2. Vector store population.
3. Retriever setup.
4. Chain construction using LCEL.
5. Printing the final prompt (Context + Question) sent to the LLM.

REQUIREMENTS:
1. An '.env' file with OPENAI_API_KEY defined.
2. A document file at 'documents/2024_state_of_the_union.txt'.
3. Dependencies: langchain, pydantic, openai, python-dotenv, chromadb.
"""

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter


def print_step(step_num, title):
    """Print formatted step header"""
    print("\n" + "=" * 60)
    print(f"STEP {step_num}: {title}")
    print("=" * 60)


def format_docs(docs):
    """Convert list of Documents to single string for the LLM context"""
    return "\n\n".join(doc.page_content for doc in docs)


def print_prompt_box(title, content, max_lines=None):
    """Print content (like a prompt or response) in a nice box format"""
    print("\n" + "┌" + "─" * 78 + "┐")
    print("│ " + title.ljust(76) + " │")
    print("├" + "─" * 78 + "┤")

    lines = content.split('\n')
    # Limit lines for long inputs like the context string
    if max_lines and len(lines) > max_lines:
        for line in lines[:max_lines]:
            # Handle lines longer than the box width
            while len(line) > 76:
                print("│ " + line[:76] + " │")
                line = line[76:]
            print("│ " + line.ljust(76) + " │")
        print("│ " + f"... ({len(lines) - max_lines} more lines, truncated view)".ljust(76) + " │")
    else:
        for line in lines:
            # Handle lines longer than the box width
            while len(line) > 76:
                print("│ " + line[:76] + " │")
                line = line[76:]
            print("│ " + line.ljust(76) + " │")

    print("└" + "─" * 78 + "┘")


def main():
    # --- RAG PIPELINE STEPS ---

    # STEP 1: Load Environment Variables
    print_step(1, "Loading Environment Variables")
    # This expects an OPENAI_API_KEY in a .env file
    load_dotenv('.env')
    print("API keys loaded (requires OPENAI_API_KEY)")

    # STEP 2: Initialize Embeddings & Vector Store
    print_step(2, "Initialize Embeddings & Vector Store")
    # Use a robust embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    # ChromaDB will store the document chunks locally
    vector_store = Chroma(
        collection_name="rag_demo_collection",
        embedding_function=embeddings
    )
    print("Vector store (Chroma) initialized with OpenAIEmbeddings")

    # STEP 3: Read & Split Document
    print_step(3, "Read & Split Document")

    document_path = "documents/2024_state_of_the_union.txt"
    try:
        # Load the content of the document
        with open(document_path) as f:
            state_of_the_union = f.read()
        print(f"Document loaded: {len(state_of_the_union)} characters")
    except FileNotFoundError:
        print(f"ERROR: Could not find {document_path}")
        print("Please create a 'documents' folder and add your source file.")
        return

    # Initialize a text splitter for creating chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    texts = text_splitter.create_documents([state_of_the_union])
    print(f"Split into {len(texts)} chunks (size=1000, overlap=200)")

    # STEP 4: Add Chunks to Vector Store
    print_step(4, "Add Chunks to Vector Store")
    # This step generates embeddings for each chunk and stores them
    ids = vector_store.add_documents(texts)
    print(f"Added {len(ids)} document chunks to vector store")

    # STEP 5: Test Similarity Search (Sanity Check)
    print_step(5, "Test Similarity Search (Retriever Test)")
    query = "Who invaded Ukraine?"
    print(f"Query: '{query}'")
    print("\nSearching vector store for most similar chunks...")

    results = vector_store.similarity_search(query, k=2)
    print(f"\nFound {len(results)} most similar chunks:\n")

    for i, res in enumerate(results, 1):
        print(f"Chunk {i} (first 200 chars):")
        print(f"{res.page_content[:200]}...")
        print()

    # STEP 6: Create Retriever
    print_step(6, "Create Retriever")
    # The retriever is the component responsible for fetching relevant docs
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    print("Retriever created (will return top 4 chunks)")

    # STEP 7: Initialize LLM
    print_step(7, "Initialize LLM")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # Low temperature for factual RAG
    print("LLM initialized (gpt-4o-mini)")

    # STEP 8: Create Prompt Template
    print_step(8, "Create Prompt Template")
    prompt_template = """Use the context provided to answer the user's question below. If you do not know the answer based on the context provided, tell the user that you do not know the answer to their question based on the context provided and that you are sorry.

    context: {context}

    question: {query}

    answer: """

    custom_rag_prompt = PromptTemplate.from_template(prompt_template)
    print("Prompt template created")
    print_prompt_box("PROMPT TEMPLATE", prompt_template)

    # STEP 9: Build RAG Chain (using LangChain Expression Language - LCEL)
    print_step(9, "Build RAG Chain (LCEL)")
    rag_chain = (
        # 1. Input: The user's query
        # 2. Context: Retrieved chunks are fetched and formatted
        # 3. Query: The user's query is passed through
            {"context": retriever | format_docs, "query": RunnablePassthrough()}
            # 4. Prompt: The complete prompt is built
            | custom_rag_prompt
            # 5. LLM: The prompt is sent to the LLM
            | llm
            # 6. Output: The response is parsed to a string
            | StrOutputParser()
    )
    print("RAG chain assembled: {retriever | format_docs} -> Prompt -> LLM -> Parser")

    # --- QUERY TESTS ---

    # STEP 10: Query with Answer (Hit)
    print_step(10, "Query RAG Chain (Question WITH Answer - HIT)")
    question1 = "Who invaded Ukraine and what are the consequences?"
    print(f"Question: '{question1}'\n")

    # 10a. Manually trace the retrieval step
    print("10a. Retrieving relevant chunks from vector store...")
    retrieved_docs = retriever.invoke(question1)
    print(f"     Retrieved {len(retrieved_docs)} chunks\n")

    # 10b. Manually trace the context formatting step
    print("10b. Formatting chunks into context string...")
    context1 = format_docs(retrieved_docs)
    print(f"     Context length: {len(context1)} characters")
    print(f"     Number of chunks combined: {len(retrieved_docs)}\n")

    # 10c. Manually trace the prompt construction step
    print("10c. Building the complete prompt with context...")
    filled_prompt1 = custom_rag_prompt.format(context=context1, query=question1)
    print_prompt_box("COMPLETE PROMPT SENT TO LLM (Truncated View)", filled_prompt1, max_lines=30)

    print("\n10d. Sending prompt to LLM (executing rag_chain.invoke)...")
    answer1 = rag_chain.invoke(question1)

    print_prompt_box("LLM RESPONSE", answer1)

    # STEP 11: Query without Answer (Miss/Rejection)
    print_step(11, "Query RAG Chain (Question WITHOUT Answer - MISS)")
    question2 = "What did the President say about space tourism?"
    print(f"Question: '{question2}'\n")

    # 11a. Manually trace the retrieval step
    print("11a. Retrieving relevant chunks from vector store...")
    retrieved_docs2 = retriever.invoke(question2)
    print(f"     Retrieved {len(retrieved_docs2)} chunks")
    print("     (These are the MOST SIMILAR chunks, even if they don't answer the question)\n")

    # 11b. Show which chunks were retrieved for the miss
    print("11b. Chunks retrieved:")
    for i, doc in enumerate(retrieved_docs2, 1):
        print(f"\n     Chunk {i} (Source Text Start):")
        print(f"     {doc.page_content[:200].replace('\n', ' ')}...")  # Print snippet

    # 11c. Manually trace the context formatting step
    print("\n11c. Formatting chunks into context string...")
    context2 = format_docs(retrieved_docs2)
    print(f"     Context length: {len(context2)} characters\n")

    # 11d. Manually trace the prompt construction step
    print("11d. Building the complete prompt with context...")
    filled_prompt2 = custom_rag_prompt.format(context=context2, query=question2)
    print_prompt_box("COMPLETE PROMPT SENT TO LLM (Truncated View)", filled_prompt2, max_lines=30)

    print("\n11e. Sending prompt to LLM (executing rag_chain.invoke)...")
    answer2 = rag_chain.invoke(question2)

    print_prompt_box("LLM RESPONSE (Should be a rejection)", answer2)

    # --- SUMMARY ---
    print("\n" + "=" * 60)
    print("SUMMARY: The RAG Flow")
    print("=" * 60)
    print("""
    THE FLOW:
    1. User Query (Text)
    2. Embeddings (Query is converted to a vector)
    3. Retriever (Vector Search finds top K document chunks)
    4. Context Format (Chunks combined into a string)
    5. Prompt Template (Context and Query injected into the template)
    6. LLM (The complete, single prompt is sent to the model)
    7. Answer (The model returns the final answer or rejection)
    """)

    # Clean up the vector store collection for re-runs (optional)
    vector_store.delete_collection()
    print("\nClean-up: Vector store collection deleted for next run.")
    print("=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
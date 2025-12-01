"""
RAG Pipeline with Conversational Memory
This version remembers previous conversations and can handle follow-up questions
"""
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from typing import List
from langchain_core.documents import Document


def print_step(step_num: int, title: str):
    """Print formatted step header"""
    print("\n" + "=" * 60)
    print(f"STEP {step_num}: {title}")
    print("=" * 60)


def print_prompt_box(title: str, content: str, max_lines: int = None):
    """Print content in a clean box format"""
    print("\n" + "┌" + "─" * 78 + "┐")
    print("│ " + title.ljust(76) + " │")
    print("├" + "─" * 78 + "┤")

    lines = content.split('\n')
    # Limit lines for display clarity
    if max_lines and len(lines) > max_lines:
        for line in lines[:max_lines]:
            # Wrap long lines
            while len(line) > 76:
                print("│ " + line[:76] + " │")
                line = line[76:]
            print("│ " + line.ljust(76) + " │")
        print("│ " + f"... ({len(lines) - max_lines} more lines)".ljust(76) + " │")
    else:
        for line in lines:
            # Wrap long lines
            while len(line) > 76:
                print("│ " + line[:76] + " │")
                line = line[76:]
            print("│ " + line.ljust(76) + " │")

    print("└" + "─" * 78 + "┘")


def print_memory_contents(memory: ConversationBufferMemory):
    """Print what's stored in memory"""
    print("\nCURRENT MEMORY:")
    chat_history = memory.load_memory_variables({})
    if chat_history.get('chat_history'):
        for msg in chat_history['chat_history']:
            role = "User" if msg.type == "human" else "Assistant"
            print(f"  {role}: {msg.content[:100].replace('\n', ' ')}...")
    else:
        print("  (Empty)")


def main():
    # STEP 1: Load Environment Variables
    print_step(1, "Loading Environment Variables")
    # NOTE: Requires an .env file with OPENAI_API_KEY set.
    load_dotenv('.env')
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found in environment variables.")
        return
    print("Status: API keys loaded")

    # STEP 2: Initialize Embeddings & Vector Store
    print_step(2, "Initialize Embeddings & Vector Store")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    persist_directory = "./chroma_db"

    # Initialize the Chroma instance. This loads or creates the client/directory.
    vector_store = Chroma(
        collection_name="test_collection",
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    print("Status: Vector store initialized")

    # STEP 3: Read & Split Document
    print_step(3, "Read & Split Document")

    # NOTE: Ensure you have a 'documents' folder with '2024_state_of_the_union.txt'
    document_path = "documents/2024_state_of_the_union.txt"
    try:
        with open(document_path, encoding='utf-8') as f:
            state_of_the_union = f.read()
        print(f"Status: Document loaded ({len(state_of_the_union)} characters)")
    except FileNotFoundError:
        print(f"ERROR: Could not find {document_path}. Please create the file.")
        return

    # Use CharacterTextSplitter for chunking
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    texts = text_splitter.create_documents([state_of_the_union])
    print(f"Status: Split into {len(texts)} chunks")

    # STEP 4: Add Chunks to Vector Store (Robust Chroma Handling)
    print_step(4, "Add Chunks to Vector Store")

    # Robustly ensure a clean collection state before adding documents.
    try:
        # Attempt to delete the collection for a clean run
        vector_store.delete_collection()
        print("Status: Existing collection deleted for fresh run.")
        # After deletion, re-initialize to ensure the collection exists and is ready
        vector_store = Chroma(
            collection_name="test_collection",
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
    except Exception as e:
        # This handles the common ValueError when the collection is not initialized.
        print(f"Warning: Could not delete existing collection, proceeding. Error: {e}")

    # Add the new documents to the (now clean or newly created) collection
    ids = vector_store.add_documents(texts)
    print(f"Status: Added {len(ids)} document chunks to vector store")

    # STEP 5: Test Similarity Search
    print_step(5, "Test Similarity Search")
    query = "Who invaded Ukraine?"
    print(f"Query: '{query}'")

    results = vector_store.similarity_search(query, k=2)
    print(f"\nFound {len(results)} most similar chunks.")

    for i, res in enumerate(results, 1):
        print(f"Chunk {i} (first 200 chars): {res.page_content[:200].replace('\n', ' ')}...")

    # STEP 6: Create Retriever
    print_step(6, "Create Retriever")
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    print("Status: Retriever created (returns top 4 chunks)")

    # STEP 7: Initialize LLM
    print_step(7, "Initialize LLM")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    print("Status: LLM initialized (gpt-4o-mini)")

    # STEP 8: Create Memory
    print_step(8, "Initialize Conversational Memory")
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    print("Status: Memory initialized (ConversationBufferMemory)")

    # STEP 9: Build Conversational RAG Chain
    print_step(9, "Build Conversational RAG Chain")

    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False
    )

    print("Status: Conversational RAG chain assembled")

    # STEP 10: First Query
    print_step(10, "Query 1 - Initial Question")
    question1 = "Who invaded Ukraine?"
    print(f"Question: '{question1}'")

    print_memory_contents(memory)

    result1 = conversational_chain({"question": question1})
    answer1 = result1['answer']

    print_prompt_box("ASSISTANT RESPONSE", answer1)
    print_memory_contents(memory)

    # STEP 11: Follow-up Query (Tests Memory!)
    print_step(11, "Query 2 - Follow-up Question (Tests Memory)")
    question2 = "Why is this wrong?"
    print(f"Question: '{question2}' (Tests context from Q1)")

    print_memory_contents(memory)

    result2 = conversational_chain({"question": question2})
    answer2 = result2['answer']

    print_prompt_box("ASSISTANT RESPONSE", answer2)
    print_memory_contents(memory)

    # STEP 12: Another Follow-up
    print_step(12, "Query 3 - Another Follow-up (Recalling history)")
    question3 = "What did you tell me in your first answer?"
    print(f"Question: '{question3}'")

    print_memory_contents(memory)

    result3 = conversational_chain({"question": question3})
    answer3 = result3['answer']

    print_prompt_box("ASSISTANT RESPONSE", answer3)

    # STEP 13: Unrelated Question
    print_step(13, "Query 4 - Unrelated Question (Tests LLM General Knowledge)")
    question4 = "What is 2+2?"
    print(f"Question: '{question4}'")

    result4 = conversational_chain({"question": question4})
    answer4 = result4['answer']

    print_prompt_box("ASSISTANT RESPONSE", answer4)

    # SUMMARY
    print("\n" + "=" * 60)
    print("SUMMARY: Conversational RAG Performance")
    print("=" * 60)
    print("""
KEY TAKEAWAYS:

1. CONTEXT AWARENESS:
   - The **ConversationalRetrievalChain** uses **ConversationBufferMemory** to remember the chat history.
   - This allows the model to correctly answer follow-up questions like "Why is this wrong?" by understanding that "this" refers to the invasion discussed in the previous turn.

2. FLOW:
   - User Question + Chat History -> Chain -> Retriever -> LLM -> Answer -> **Update Memory**. 

3. USAGE:
   - Essential for any multi-turn conversational application, such as customer support or interactive assistants.
""")

    print("\n" + "=" * 60)
    print("FINAL MEMORY STATE:")
    print("=" * 60)
    print_memory_contents(memory)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
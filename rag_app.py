import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def main():
    print("Initializing Simple RAG Application...")
    
    # 1. Load the document
    file_path = "sample.txt"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return
    
    print("Loading document...")
    loader = TextLoader(file_path)
    docs = loader.load()

    # 2. Split the document into chunks
    print("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # 3. Create embeddings and store in FAISS vector store
    print("Creating embeddings (this may take a moment to download the model the first time)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever()

    # 4. Set up the local Ollama LLM
    # Note: Make sure Ollama is running and you have pulled the model (e.g., 'ollama run llama3')
    model_name = "llama3.2:1b" # Change this if you are using a different model like 'mistral'
    print(f"Connecting to local Ollama instance (model: {model_name})...")
    
    try:
        llm = OllamaLLM(model=model_name)
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Please ensure Ollama is installed, running, and the model is downloaded.")
        return

    # 5. Create the retrieval chain
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("\n" + "="*50)
    print("RAG System Ready! Type 'exit' or 'quit' to stop.")
    print("="*50 + "\n")

    # 6. Interactive chat loop
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting...")
            break
            
        if not user_input.strip():
            continue

        try:
            print("Thinking...")
            response = rag_chain.invoke({"input": user_input})
            print(f"\nAI: {response['answer']}\n")
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Is Ollama running locally?\n")

if __name__ == "__main__":
    main()

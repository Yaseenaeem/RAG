# Retrieval-Augmented Generation (RAG)
# Pipeline:
# PDF -> Chunks -> Embeddings -> FAISS -> Retrieval -> Gemma -> Answer
# -------------------------------------------------------------------------------------------------------------------------------

#Importing Necessary Libraries
from sentence_transformers import CrossEncoder # Re-ranking the chunks before being sent to gemma and after FAISS
from sentence_transformers import SentenceTransformer  #To transform sentences from English to Embeddings of
#384-Dimension Vectors in float32 Understand by FAISS
from langchain_community.document_loaders import PyPDFLoader #Langchain is like a toolkit used by Python
#for reading whole Pdf automatically and later create automatic chunks using text splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import pickle
import os
import ollama #Ollama is a software that have pre trained Ai models and we using Gemma 4b here
from rank_bm25 import BM25Okapi

# Program does not need to rebuilt everything in database everytime, they are pre stored and will load 
# same from where program ended last time like a Library reopens next day.
# Python program look if vector.index already there, if yes it simply load already saved files and answer:
if os.path.exists("Vector.index") and os.path.exists("chunks.pkl"):
    print("Existing vector database found!")

     # Load the saved FAISS index
    index = faiss.read_index("Vector.index")
    print("FAISS index loaded successfully!")

    # Load the saved chunks
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    print("Chunks loaded successfully!")

    # Rebuild BM25 index from loaded chunks
    texts = [doc.page_content for doc in chunks]
    tokenized_chunks = [
        text.lower().split()
        for text in texts
    ]
    bm25 = BM25Okapi(tokenized_chunks)

    print("BM25 index loaded!")

    # Load the embedding model
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Embedding model loaded!")

    print("Loading reranker...")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("Reranker loaded!")
else:
    print("No existing database found. Creating a new one...")

# If vector.index is not there or path not found then pythons is to build everything again.
# PDF Reading Automatic using pyPDF and Langchain
    print("\nLoading PDFs...")
    documents = []
    pdf_files = [          # Load All Files and read from them
    "Docs/Employee-Handbook.pdf",
    "Docs/Acceptable-Usage-Policy.pdf",
    "Docs/Business Expenses Policy.pdf",
    "Docs/HR-Manual.pdf",
    "Docs/IT-Security-Policy.pdf",
    "Docs/Remote-Work-Policy.pdf" ]
    
    for pdf in pdf_files:
        print(f"Loading {pdf}...")
        loader = PyPDFLoader(pdf)
        documents.extend(loader.load())

    print("All PDFs loaded successfully!")
    print("Total pages:", len(documents))


    # Creating Chunks of 500 Characters
    print("\nCreating chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(documents)

    # Check what information LangChain stores with each chunk
    # print("\nMetadata of first chunk:")
    # print(chunks[0].metadata)
    print("Chunks created successfully!")
    print("Total chunks:", len(chunks))


    #print("\nFirst Chunk:")
    #print(chunks[0].page_content)

    #print("\nSecond Chunk:")
    #print(chunks[1].page_content)

    #print("\nLast Chunk:")
    #print(chunks[-1].page_content)

    # Loading Embedding Model -> Translator to convert English Text to coordinates in 384 Dimensional Vectors 
    # FAISS don't Understand English but embeddings in float32 using sentence transformer
    print("\nLoading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Embedding model loaded!")

    print("Loading reranker...")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("Reranker loaded!")

    # Creating Embeddings for every chunk -> Assigning Coordinates of 384 Dimensional Vector
    print("\nCreating embeddings...")
    texts = [doc.page_content for doc in chunks] 

#    ---------------- BM25 INDEX ----------------
# FAISS is good at understanding meaning (semantic search)
# but sometimes misses exact keywords.

# Example:
# User asks: "unused vacation days"

# FAISS may understand: vacation policy

# BM25 directly looks for exact words like: unused, vacation, days
# So BM25 acts like a second librarian who remembers
# which pages contain specific keywords.

# BM25 requires tokenized text (list of words), therefore every chunk is converted into:
# "Employees should request approval..."
# -> ["employees", "should", "request", "approval", ...]

# These tokenized chunks are stored in the BM25 index
# and later used for keyword-based retrieval.
    tokenized_chunks = [  # BM25 needs tokenized text
        text.lower().split()
        for text in texts
    ]   

    bm25 = BM25Okapi(tokenized_chunks)
    print("BM25 index created!")
    embeddings = model.encode(texts)
    embeddings = np.array(embeddings).astype("float32")
    print("Embeddings created successfully!")
    # print("Total embeddings:", len(embeddings))


    # The Librarian (FAISS) recieves all 160(No. of chunks) coordinate card and then remembers where every chunk is
    # placed inside database

    # Faiss -> Librarian
    # Chunks -> Books
    # Similar Embeddings (Nearby Coordinates) -> placed nearby in database ->
    # Similar Books placed in same bookshelf in a library
    # Faiss organizes the library (Database) embeddings (Books/ chunks) placed in order
    print("\nCreating FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print("FAISS index created successfully!")
    print("Number of vectors stored:", index.ntotal)
    # 160 Chunks = 160 Vectors

    # -------- RAG MEMORY --------------
    # Save the FAISS index to disk
    # So Everytime when program starts (Library opens), it's pre organized and FAISS don't need to spend time again
    # on organizing the chunks or create new embeddings everytime
    # Vector.index is like a notebook in which FAISS have written the map of library so when he returns to Lib
    # Everytime, he knows which embedding is placed where in database.
    # Vector.index is simply the saved map of your vector database. It contains the numerical vectors and the 
    # structure FAISS needs so you don't have to rebuild it every time.
    faiss.write_index(index, "Vector.index")
    print("FAISS index saved successfully!")


    # Save all chunks to disk
    # After having the notebook to know the map of library the content of books (chunks) also need to be saved to disc
    # so FAISS don't spend time loading chunks everytime (memorizing content).
    # -> Actual books(chunks) stored safely in a warehouse
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print("Chunks saved successfully!")

# The next thing is to walk into the library and ask:
# "What is the company's dress code?"

# Question will follow the exact same journey:

# Converted into an embedding.
# Given to the librarian.
# The librarian finds the nearest books.
# The books - chunks are handed to Gemma 3.
# Gemma reads the retrieved chunks and generates a natural answer.
# That will complete the fully automated RAG pipeline.


# ---------------- Conversation Memory ----------------
# Stores full conversation for the final LLM prompt
chat_history = []

# Stores only previous user questions for query rewriting
search_history = []

while True:
    question = input("\nAsk your question (type 'exit' to quit): ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    print("\nCreating search query...")

    recent_history = ""

    for q in search_history[-3:]:
        recent_history += f"User: {q}\n" 
        # TO make FAISS understand the context of follow up questions

    rewrite_prompt = f"""
    You rewrite search queries for document retrieval.

    Your job is ONLY to rewrite the user's latest question into a standalone search query.

    Rules:

    - Use previous conversation ONLY if the current question depends on it.
    - If the current question already contains enough context, leave it unchanged.
    - Keep the query short.
    - Preserve important keywords.
    - Never answer the question.
    - Return ONLY the rewritten search query.

    Examples

    Previous Conversation:
    User: What is the vacation policy?

    Current Question:
    How much notice is required?

    Search Query:
    vacation policy notice requirements

    Previous Conversation:
    User: What is the dress code?

    Current Question:
    Are trousers allowed?

    Search Query:
    dress code trousers policy

    Previous Conversation:
    User: What is the dress code?

    Current Question:
    What about shoes?

    Search Query:
    dress code shoes policy

    Now rewrite.

    Previous Conversation:
    {recent_history}

    Current Question:
    {question}

    Search Query:
    """
    # Think of it like handing a note to the librarian. The note says:
    # You are not answering. You are only rewriting the search query. Use conversation context if needed. 
    # Return only the rewritten query.

    rewrite_response = ollama.chat(  # Ask Gemma to rewrite
        model="gemma3:4b",
        messages=[
            {
                "role": "user",
                "content": rewrite_prompt
            }
        ]
    )

    search_query = rewrite_response["message"]["content"].strip()  # Extract rewritten query
    print("\nRewritten Search Query:")
    print(search_query)   # Debug output so we can see what the librarian is thinking.

    # BM25 keyword search
    # BM25 performs keyword search.
    # It checks which chunks contain the exact words from the search query.
    # Example: "vacation policy" -> ["vacation", "policy"]
    # BM25 scores every chunk and returns the top keyword matches.
    query_tokens = search_query.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_indices = np.argsort(bm25_scores)[::-1][:5]

    print("\nTop BM25 Results:")
    for rank, idx in enumerate(bm25_indices, start=1):
        source = chunks[idx].metadata["source"]
        page = chunks[idx].metadata["page_label"]

        print(
            f"{rank}. Score: {bm25_scores[idx]:.2f} | "
            f"{os.path.basename(source)} | "
            f"Page {page}" )

    print("\nConverting search query into embedding...")

    question_embedding = model.encode([search_query]).astype("float32") # Embed the rewritten query only once.
    print("Question converted successfully!")
    # Above explanation!
    # User:
    # What about trousers?
    # Search Query:
    # What does the company's dress code policy say about trousers?
    # (assuming Gemma rewrites well)
    # This gives FAISS a much better search phrase.



# Example Question Says:
# "What is the dress code?"

# The librarian doesn't understand English.
# So he first receives the coordinate card of your question.
# He then compares it with the coordinate cards of all 160 books.

# Imagine:
# Book 1 → Distance = 2.31
# Book 2 → Distance = 0.41   
# Book 3 → Distance = 1.92
# Book 160 → Distance = 3.05

# The librarian finds the closest matching chunks (books).
# k decides! k=1 -> 1 best matching chunk.
# k=3 -> 3 best matching chunks print and passed to gemma
    # print("\nSearching the vector database...")

    distances, indices = index.search(question_embedding, k=5)
# indices stores the positions of the retrieved chunks.
# Example: indices = [[45, 46, 47, 12, 91]]
    # print("\nRetrieved Distances:")
 # looking if all chunks faiss send to gemma are relevant or not by observing distances between chunks
    for rank, (distance, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        source = chunks[idx].metadata["source"]
        page = chunks[idx].metadata["page_label"]

        # print(
            # f"{rank}. Distance: {distance:.4f} | "
            # f"{os.path.basename(source)} | Page {page}" )
    print("Search completed!")

    # Combine FAISS and BM25 results
    candidate_indices = list(indices[0]) + list(bm25_indices)

    # Hybrid Search: FAISS finds chunks by meanin, BM25 finds chunks by exact keywords.
    # We merge both result sets into one candidate pool and Duplicate pages are removed before reranking.
    # The reranker then decides which chunks are truly most relevant to the user's question.
    unique_indices = []
    seen_pages = set()

    for idx in candidate_indices:

        source = chunks[idx].metadata["source"]
        page = chunks[idx].metadata["page_label"]

        page_id = (source, page)

        if page_id not in seen_pages:
            seen_pages.add(page_id)
            unique_indices.append(idx)

    # Build all question-chunk pairs
    pairs = []

    for i in unique_indices:
        pairs.append([search_query, chunks[i].page_content]) # Cross Encoder only recieves clean pages

    # Run the reranker ONCE
    scores = reranker.predict(pairs)

    # Sort from highest score to lowest
    reranked = sorted(
        zip(scores, unique_indices),  # The score belongs now to unique indices
        reverse=True
)
    # debug
    # print(reranked)

    # Top Ranked Results out of 10 retrieved chunks by FAISS sent to re-ranker -> Gemma(LLM) for response generation
    print("\nTop Reranked Results:")

    for rank, (score, idx) in enumerate(reranked[:5], start=1):
        source = chunks[idx].metadata["source"]
        page = chunks[idx].metadata["page_label"]

        print(
            f"{rank}. Score: {score:.2f} | "
            f"{os.path.basename(source)} | "
            f"Page {page}"  )

# Chunk Retrieval
# At this moment, Gemma hasn't spoken yet.
    seen_chunks = set()
    retrieved_chunks = ""
    sources = []  # Storing sources list in retrieved chunks

    for count, (score, i) in enumerate(reranked[:5], start=1):
        chunk_text = chunks[i].page_content
        if chunk_text in seen_chunks:
            continue
    # reranked instead of indices[0], (score, i) because each item contains both the reranker score and the chunk index
    # [:5] so Gemma only receives the top 5 reranked chunks

        seen_chunks.add(chunk_text)  # Avoiding sending same chunks multiple times to gemma, 
        # improves efficiency of program
        source = chunks[i].metadata["source"]
        page = chunks[i].metadata["page_label"]
        sources.append((source, page))
        retrieved_chunks += f"Context {count}\n\n"
        retrieved_chunks += chunk_text
        retrieved_chunks += "\n\n"
    print("\nRetrieved Chunks:\n")
    print(retrieved_chunks)


# FAISS already found the correct page.
# So we only send that one relevant section.
# That exact text gets inserted into the prompt.

    recent_history = "\n".join(chat_history[-3:])

        # Now Gemma sees: 
        # User: What is vacation policy?
        # User: What about unused days?
        # User: How much notice is required?
        # Much clearer.

    prompt = f"""
    You are an enterprise document assistant.

    Answer the user's question using ONLY the provided context.

    Rules:
    - Do not use outside knowledge.
    - Do not use information from previous conversation unless it is explicitly present in the provided context.
    - If the answer is not available in the context, say:
      "I couldn't find that information in the provided documents."
    - Combine information from multiple contexts when necessary.
    - Give concise, clear answers.
    - Do not mention information that is unrelated to the user's current question.

    Context:
    {retrieved_chunks}

    Question:
    {question}

    Answer:
    """
# Send the prompt to Gemma 3 running locally through Ollama
    response = ollama.chat(
        model="gemma3:4b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "num_predict": 200  # librarian doesn't need to write a 3-page report for: "What is vacation policy?"
                            # This reduces generation time.
        }
    )
    
  # Print Gemma's answer
    answer = response["message"]["content"]
    print("\nAnswer:")
    print(answer)

    unique_sources = list(dict.fromkeys(sources)) # duplicate citations (references) are removed,
    # maintaining original order

    print("\nSources:") # Adding sources (citations) in recieved LLM's response
    for source, page in unique_sources:
        filename = os.path.basename(source)
        print(f"- {filename} (Page {page})")

    # Save the conversation into memory
    chat_history.append(
        f"User: {question}\nAssistant: {answer}"
    )
    # Keep only the last 3 questions
    search_history.append(question)
    search_history = search_history[-3:]

# The completed prompt becomes something like:
# You are a helpful assistant.
# Answer the user's question ONLY using the information below.
# Context:
# Dress code
# Our company's official dress code is Business Casual...
# Question:
# What is the dress code?
# This is exactly what Gemma receives.
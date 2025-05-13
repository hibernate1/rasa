# from langchain_chroma import Chroma
# from langchain.embeddings import OpenAIEmbeddings

# # Initialize embeddings and Chroma DB
# embeddings = OpenAIEmbeddings()  # Or your chosen embeddings
# db = Chroma(persist_directory="./db", embedding_function=embeddings)

# # Add some documents to the DB (basic example)
# documents = ["Document 1 text", "Document 2 text", "Document 3 text"]
# db.add_documents(documents)

# # Run a search query
# query = "Document 1"
# results = db.similarity_search(query, k=1)
# print(results)
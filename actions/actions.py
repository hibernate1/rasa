from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import os

class ActionQueryPdfKnowledge(Action):

    def name(self):
        return "action_query_pdf_knowledge"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: dict):

        query = tracker.latest_message.get("text")  # Get the query from the user message

        # Load and chunk the PDF document
        try:
            loader = PyPDFLoader("sample.pdf")  # Adjust the path if necessary
            pages = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)  # Split text into chunks
            docs = text_splitter.split_documents(pages)
        except Exception as e:
            dispatcher.utter_message(text="Error loading or processing the PDF document.")
            return []

        # Use HuggingFace Embeddings for document embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Initialize ChromaDB for document storage
        db_dir = "./chroma_storage"  # Directory for Chroma storage
        try:
            db = Chroma.from_documents(docs, embeddings, persist_directory=db_dir)  # Use Chroma for document search
        except Exception as e:
            dispatcher.utter_message(text="Error setting up ChromaDB.")
            return []

        # Search for the answer based on the user query
        try:
            results = db.similarity_search(query, k=1)  # Get the top result
            if results:
                answer = results[0].page_content  # Extract the content from the top result
            else:
                answer = "Sorry, I couldn't find anything relevant in the document."
        except Exception as e:
            answer = "Error performing the document search."

        # Respond with the answer
        dispatcher.utter_message(text=answer)
        return []

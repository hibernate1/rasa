from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from langchain.vectorstores import Chroma

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import os
from rasa_sdk import Action
import re
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from typing import Any, Dict, List
from transformers import pipeline

from typing import Any, Dict, List
from rasa_sdk.forms import FormValidationAction
import requests

from typing import Any, Text, Dict, List
from transformers import pipeline

from typing import Any, Text, Dict, List
from transformers import pipeline
from rasa_sdk import Action
from rasa_sdk.events import EventType
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.interfaces import Tracker 
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import os

from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import EventType
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import chromadb
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from chromadb.errors import InvalidCollectionException

# Initialize Chroma client and embedding model
client = chromadb.Client()
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
collection_name = "pdf_collection"

# Try to get or create the collection
try:
    collection = client.get_collection(collection_name)
except InvalidCollectionException:
    # If collection doesn't exist, create it
    collection = client.create_collection(collection_name)
    print(f"Collection {collection_name} created.")

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text")
    return text

# Function to add PDF to Chroma collection
def add_pdf_to_chroma(pdf_path, collection):
    pdf_text = extract_text_from_pdf(pdf_path)
    if pdf_text:
        doc_id = pdf_path.split("/")[-1]  # Use the PDF file name as the ID
        embedding = embedding_model.encode(pdf_text).tolist()  # Encode the PDF text into an embedding
        collection.add(
            documents=[pdf_text],
            metadatas=[{"source": pdf_path}],
            ids=[doc_id],
            embeddings=[embedding]
        )
        print(f"Added {pdf_path} to Chroma.")
    else:
        print(f"No text found in {pdf_path}.")

# Function to query the collection
def query_collection(query_text, collection, top_k=3):
    embedding = embedding_model.encode(query_text).tolist()  # Generate embedding for the query
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k  # Limit to top_k results
    )
    return results

# Action class for querying PDF knowledge
class ActionQueryPdfKnowledge(Action):
    
    def name(self) -> str:
        return "action_query_pdf_knowledge"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: dict) -> list:
        
        query = tracker.latest_message.get("text")  # Get the query from the user message

        if not query:
            dispatcher.utter_message(text="Please provide a query.")
            return []

        # Process PDF (you can call this before querying if needed)
        pdf_file_path = "sample.pdf"  # Adjust the path as necessary
        add_pdf_to_chroma(pdf_file_path, collection)

        # Query the collection
        try:
            query_results = query_collection(query, collection)

            # Check if any results are returned
            if query_results and query_results['documents']:
                answer = query_results['documents'][0]  # Get the content from the top result
                # Ensure the response is a single string
                answer = str(answer)
                dispatcher.utter_message(text=answer)
            else:
                dispatcher.utter_message(text="Sorry, I couldn't find anything relevant in the document.")
        
        except Exception as e:
            dispatcher.utter_message(text=f"Error performing the document search: {str(e)}")

        return []



class ActionSubmitComplaint(Action):
    def name(self) -> str:
        return "action_submit_complaint"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[str, Any]) -> List[Dict[str, Any]]:

        system = tracker.get_slot("platform")
        complaint_details = tracker.get_slot("complaint_details")

        summary = f"Here's a summary of your complaint:\n- System: {system}\n - Details: {complaint_details}"
        dispatcher.utter_message(text=summary)

        # Replace this with your actual API endpoint
        url = "https://your-api.com/submit-complaint"
        data = {
            "system": system,
            "complaint_detailsme": complaint_details
        }

        try:
            response = requests.post(url, json=data)
            if response.status_code == 200:
                dispatcher.utter_message(text="Complaint submitted successfully.")
            else:
                dispatcher.utter_message(text="There was an issue submitting your complaint.")
        except Exception as e:
            dispatcher.utter_message(text=f"Failed to submit complaint: {e}")

        return []
    
class ActionSolveMath(Action):
    def name(self) -> str:
        return "action_solve_math"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[str, Any]) -> List[Dict[str, Any]]:

        user_query = tracker.latest_message.get('text')

        # Match everything that looks like a valid math expression
        expression_match = re.search(r'(\d+[\d\+\-\*\/\(\)\s\.]*)', user_query)
        expression = expression_match.group(1).strip() if expression_match else None

        if expression:
            try:
                # Evaluate the expression safely
                result = eval(expression, {"__builtins__": None}, {})
                dispatcher.utter_message(text=f"The answer is: {result}")
            except Exception as e:
                dispatcher.utter_message(text="Sorry, I couldn't evaluate that. Please provide a valid math expression.")
        else:
            dispatcher.utter_message(text="Please provide a valid math expression.")

        return []
# class ActionFallbackLLM(Action):
#     def name(self) -> Text:
#         return "action_fallback_llm"

#     def __init__(self):
#         self.generator = pipeline("text-generation", model="gpt2")

#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#         last_msg = tracker.latest_message.get("text")
#         response = self.generator(last_msg, max_length=100, do_sample=True)[0]["generated_text"]
#         dispatcher.utter_message(text=response)
#         return []
    

# Load the model once globally
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import wikipediaapi
from transformers import pipeline


 

from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import wikipediaapi
from transformers import pipeline
import re
import logging

# Enable logging
logger = logging.getLogger(__name__)

# Load the question-answering pipeline (once at module level)
qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2"
)

#action_fallback_llm
class ActionFallbackLLM(Action):

    def name(self) -> Text:
        return "action_fallback_llm"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Extract user question (assuming it's stored in a slot or latest message)
        user_question = tracker.latest_message.get('text')  # or use a slot

        if not user_question:
            dispatcher.utter_message(text="I didn't catch your question. Can you rephrase?")
            return []

        # Step 1: Try to answer via Wikipedia QA
        answer = self.answer_from_wikipedia(user_question)

        # Step 2: If Wikipedia fails, use LLM
        if not answer:
            answer = "I didn't catch your question. Could you please rephrase it?"

        dispatcher.utter_message(text=answer)
        return []

    def answer_from_wikipedia(self, question: str) -> str:
        # Naive topic extraction (improve with NLP/slots if needed)
        topic = self.extract_topic(question)

        # Use Wikipedia API with a valid user agent
        #  wiki = wikipediaapi.Wikipedia('en', user_agent='MyBot1.0')
        wiki = wikipediaapi.Wikipedia(language='en',user_agent='server-api-agent') 

        page = wiki.page(topic)

        if not page.exists():
            return ""

        context = page.summary[:1000]  # Limit context size

        try:
            result = qa_pipeline(question=question, context=context)
            answer = result.get("answer", "").strip()
            if answer.lower() not in ["", "unknown"]:
                return answer
        except Exception:
            pass
        return ""

    def extract_topic(self, question: str) -> str:
        import re
        keywords = re.findall(r"\b[A-Z][a-z]*\b", question)
        return keywords[-1] if keywords else question.split()[-1]




    
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

class ActionGreetUser(Action):
    def name(self):
        return "action_greet_user"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker, domain: dict):
        dispatcher.utter_message("Hello from Rasa action!")
        return []

import requests
import logging
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class ActionSubmitComplaint(Action):
    def name(self):
        return "action_submit_complaint"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        system = tracker.get_slot("system_name")
        details = tracker.get_slot("complaint_details")

        # Step 1: Call login API
        login_url = "http://13.234.10.90:8080/api/v1/auth/login"
        login_payload = {
            "email": "admin@admin.com",
            "password": "1234"
        }

        try:
            login_response = requests.post(login_url, json=login_payload)
            login_response.raise_for_status()
            token = login_response.json().get("token")

            if not token:
                logger.error("Login failed: No token received.")
                dispatcher.utter_message(text="Login failed: No token received.")
                return []

        except requests.RequestException as e:
            logger.error(f"Login error: {str(e)}")
            dispatcher.utter_message(text=f"Login error: {str(e)}")
            return []

        # Step 2: Submit complaint using token
        submit_url = "http://13.234.10.90:8080/api/v1/ticket/create"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        complaint_payload = {
            "name": "ovi",
            "title": system + " issue",
            "email": "buetovi@gmail.com",
            "detail": [
                {
                    "content": [
                        {
                            "type": "text",
                            "text": details,
                            "styles": {}
                        }
                    ],
                    "children": []
                }
            ],
            "priority": "medium",
            "type": "Bug",
            "createdBy": {
                "id": "7da7b41c-240c-4aa8-94dd-52c9e3981280",
                "name": "admin",
                "email": "admin@admin.com"
            }
        }

        try:
            submit_response = requests.post(submit_url, json=complaint_payload, headers=headers)
            submit_response.raise_for_status()
            logger.info(f"Complaint submitted successfully for {system}.")
            dispatcher.utter_message(text=f"Complaint submitted successfully for {system}.")

        except requests.RequestException as e:
            logger.error(f"Failed to submit complaint: {str(e)}")
            dispatcher.utter_message(text=f"Failed to submit complaint: {str(e)}")
            return [SlotSet("system_name", None), SlotSet("complaint_details", None)]

        # Reset the slots
        return [SlotSet("system_name", None), SlotSet("complaint_details", None)]

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import os
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModel
import torch
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter

class HuggingFaceEmbeddings:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed_query(self, query: str):
        """Generate embedding for a query string."""
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state.mean(dim=1).squeeze()
        return embeddings.numpy()

    def embed_documents(self, documents: list):
        """Generate embeddings for a list of documents."""
        embeddings = []
        for doc in documents:
            inputs = self.tokenizer(doc.page_content, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                embeddings.append(self.model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy())
        return embeddings

class ActionQueryOrScrapeRobi(Action):
    def name(self):
        return "action_query_or_scrape_robi"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: dict):

        persist_dir = "./chroma"
        query = tracker.latest_message.get("text")

        # Use Hugging Face for embeddings (free alternative)
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        # Create the HuggingFace embedding object
        embedding = HuggingFaceEmbeddings(model_name=model_name)

        # If ChromaDB doesn't exist yet, scrape and store
        if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
            dispatcher.utter_message("Fetching Robi info from Wikipedia. Please wait...")

            # Scrape
            url = "https://en.wikipedia.org/wiki/Robi_(company)"
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")
            content = soup.find("div", {"class": "mw-parser-output"})
            paragraphs = content.find_all("p")
            text = "\n".join([p.get_text() for p in paragraphs if p.get_text(strip=True)])

            # Chunk and embed
            splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs = splitter.create_documents([text])

            # Use Hugging Face Embeddings
            vectorstore = Chroma.from_documents(
                docs, embedding.embed_documents, persist_directory=persist_dir
            )
            vectorstore.persist()
            db = vectorstore
        else:
            db = Chroma(persist_directory=persist_dir, embedding_function=embedding.embed_query)

        # Query
        results = db.similarity_search(query, k=2)
        answer = "\n".join([res.page_content for res in results])

        dispatcher.utter_message(text=answer or "Sorry, I couldn't find relevant information.")
        return []

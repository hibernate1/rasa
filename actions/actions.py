from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

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


class ActionSubmitComplaint(Action):
    def name(self) -> str:
        return "action_submit_complaint"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[str, Any]) -> List[Dict[str, Any]]:

        system = tracker.get_slot("system")
        name = tracker.get_slot("name")
        phone = tracker.get_slot("phone")
        complaint_details = tracker.get_slot("complaint_details")

        summary = f"Here's a summary of your complaint:\n- System: {system}\n- Name: {name}\n- Phone: {phone}\n- Details: {complaint_details}"
        dispatcher.utter_message(text=summary)

        # Replace this with your actual API endpoint
        url = "https://your-api.com/submit-complaint"
        data = {
            "system": system,
            "name": name,
            "phone": phone,
            "details": complaint_details
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

from typing import Dict, Any
import shutil
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')


class AIVoiceAssistant:
    def __init__(self):
        self._api_key = api_key
        self._persist_directory = "chroma_db"
        
        self._cleanup_existing_db()
        
        self._embeddings = OpenAIEmbeddings(openai_api_key=self._api_key)
        
        self._llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=self._api_key
        )
        
        self._memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self._vector_store = None
        
        if self._create_kb():
            self._create_chat_chain()
        else:
            raise Exception("Failed to initialize knowledge base")

    def _cleanup_existing_db(self) -> None:
        """Remove existing Chroma database if it exists"""
        try:
            if os.path.exists(self._persist_directory):
                shutil.rmtree(self._persist_directory)
                print("Cleaned up existing database")
        except Exception as e:
            print(f"Error cleaning up database: {e}")

    def _create_kb(self) -> bool:
        """Create knowledge base and return success status"""
        try:
            loader = PyPDFLoader(r"D:\zML\audio_conv_rag\rag\Harsh_Tyagi_1.4.pdf")  
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)

            self._vector_store = Chroma.from_documents(
                documents=splits,
                embedding=self._embeddings,
                persist_directory=self._persist_directory
            )
            
            self._vector_store.persist()
            print("Knowledgebase created successfully!")
            return True
        except Exception as e:
            print(f"Error while creating knowledgebase: {e}")
            return False

    def _create_chat_chain(self) -> None:
        """Create the conversational chain"""
        if not self._vector_store:
            raise Exception("Vector store not initialized")

        prompt_template = """
        You are a helpful AI Assistant designed to assist customers with queries based on the provided information from the document.
        Your goal is to respond concisely and accurately. If the information is not available, simply say you don't know.
 
        
        Context: {context}
        Chat History: {chat_history}
        Question: {question}
        
        Response:"""

        PROMPT = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=prompt_template
        )

        self._chain = ConversationalRetrievalChain.from_llm(
            llm=self._llm,
            retriever=self._vector_store.as_retriever(),
            memory=self._memory,
            combine_docs_chain_kwargs={"prompt": PROMPT}
        )

    def interact_with_llm(self, customer_query: str) -> str:
        """Interact with the language model"""
        try:
            if not hasattr(self, '_chain'):
                return "System is not properly initialized. Please restart."
                
            result = self._chain({"question": customer_query})
            return result["answer"]
        except Exception as e:
            print(f"Error during interaction: {e}")
            return "I apologize, but I'm having trouble processing your request."
            
    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, '_vector_store') and self._vector_store:
            try:
                self._vector_store.persist()
            except Exception as e:
                print(f"Error during cleanup: {e}")

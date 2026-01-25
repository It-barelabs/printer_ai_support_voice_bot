from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import List, TypedDict
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from uuid import UUID
import uuid
import os

load_dotenv()


# Define state for application with messages for memory
class State(MessagesState):
    question: str
    context: List[Document]
    answer: str


class TechnicianRAGAgent:
    """
    A RAG-based question-answering agent with session-based memory.
    Each session is identified by a UUID and maintains its own conversation history.
    """
    
    def __init__(self, persist_dir: str = "notebooks/chroma_store", collection_name: str = "HP_Color_LaserJet"):
        """
        Initialize the TechnicianRAGAgent.
        
        Args:
            persist_dir: Directory path for the Chroma vector store
            collection_name: Name of the collection to use for RAG
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        
        # Get Google Cloud project ID from environment variable
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT') or os.getenv('VERTEX_AI_PROJECT') or os.getenv('GCP_PROJECT')
        if not project_id:
            raise ValueError(
                "Google Cloud project ID not found. Please set one of these environment variables: "
                "GOOGLE_CLOUD_PROJECT, VERTEX_AI_PROJECT, or GCP_PROJECT"
            )
        
        location = os.getenv('VERTEX_AI_LOCATION', 'global')
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0.5,
            project=project_id,
            location=location,
            vertexai=True,
            max_retries=3
        )
        
        # Initialize embeddings (reused when switching collections)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            project=project_id,
            vertexai=True
        )
        
        # Initialize vector store with specified collection
        self.vector_store = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )
        
        # Define prompt with conversation history
        self.prompt = ChatPromptTemplate([
            ("system", """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know.
Use the conversation history to provide contextual answers."""),
            ("human", "Context: {context}\n\nQuestion: {question}")
        ])
        
        # Initialize memory checkpointe
        self.memory = InMemorySaver()
        self.config = {
                        "configurable": {
                            # "thread_id": self.thread_id
                        }
        }
        # Build the graph
        self.graph = self._build_graph()
        
        # Track active sessions and their collections
        self.active_sessions = set()
        self.session_collections = {}  # Maps session_id -> collection_name
    
    def _build_graph(self):
        """Build the LangGraph workflow with retrieve and generate steps."""
        graph_builder = StateGraph(State)
        
        # Add nodes
        graph_builder.add_node("retrieve", self._retrieve)
        graph_builder.add_node("generate", self._generate)
        
        # Add edges
        graph_builder.add_edge(START, "retrieve")
        graph_builder.add_edge("retrieve", "generate")
        
        # Compile with memory checkpointer
        return graph_builder.compile(checkpointer=self.memory)
    
    def _retrieve(self, state: State):
        """Retrieve relevant documents from vector store."""
        print(f"SEARCHING DOCUMENTS...\n{'='*20}")
        retrieved_docs = self.vector_store.similarity_search(state["question"])
        if retrieved_docs:
            print(f"searched...{retrieved_docs[0].page_content[:100]}\n...\n{'='*20}")
        return {"context": retrieved_docs}
    
    def _generate(self, state: State):
        """Generate answer using LLM with context and conversation history."""
        print(f"GENERATING ANSWER...\n{'='*20}")
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        
        # Prepare messages with history
        messages = self.prompt.invoke({
            "question": state["question"],
            "context": docs_content
        })
        
        response = self.llm.invoke(messages)
        
        # Return answer and update message history
        return {
            "answer": response.content,
            "messages": [
                HumanMessage(content=state["question"]),
                AIMessage(content=response.content)
            ]
        }
    
    def switch_collection(self, collection_name: str):
        """
        Switch to a different Chroma collection.
        
        Args:
            collection_name: Name of the collection to switch to
        """
        if collection_name == self.collection_name:
            print(f"Already using collection: {collection_name}")
            return
        
        print(f"Switching collection from '{self.collection_name}' to '{collection_name}'")
        
        # Update collection name
        self.collection_name = collection_name
        
        # Reinitialize vector store with new collection
        self.vector_store = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )
        
        print(f"Now using collection: {self.collection_name}")
    
    def invoke(self, question: str, session_id: str | UUID, collection_name: str | None = None) -> dict:
        """
        Process a question within a session.
        
        Args:
            question: The user's question
            session_id: UUID string or UUID object identifying the session
            collection_name: Optional collection name to use (for new sessions or to switch)
            
        Returns:
            dict: Response containing the answer and full state
        """
        # Convert session_id to string if it's a UUID object
        if isinstance(session_id, UUID):
            session_id = str(session_id)
        
        # Validate UUID format
        try:
            UUID(session_id)
        except ValueError:
            raise ValueError(f"Invalid UUID format: {session_id}")
        
        # Check if this is a new session
        is_new_session = session_id not in self.active_sessions
        
        if is_new_session:
            print(f"NEW SESSION: {session_id}")
            
            # Use specified collection or default
            target_collection = collection_name if collection_name else self.collection_name
            
            # Switch to the collection for this session
            if target_collection != self.collection_name:
                self.switch_collection(target_collection)
            
            # Track session and its collection
            self.session_collections[session_id] = target_collection
            self.reset_memory(session_id)
            self.active_sessions.add(session_id)
        else:
            # For existing session, optionally switch collection
            if collection_name and collection_name != self.session_collections.get(session_id):
                print(f"Switching collection for existing session {session_id}")
                self.switch_collection(collection_name)
                self.session_collections[session_id] = collection_name
            else:
                # Switch back to session's collection if needed
                session_collection = self.session_collections.get(session_id, self.collection_name)
                if session_collection != self.collection_name:
                    self.switch_collection(session_collection)
        
        print(f"SESSION: {session_id} | COLLECTION: {self.collection_name}")
        
        # Invoke graph with session-specific thread
        config = {"configurable": {"thread_id": session_id}}
        response = self.graph.invoke(
            {"question": question},
            config=config
        )
        
        return response
    
    def reset_memory(self, session_id: str | UUID):
        """
        Reset/clear memory for a specific session.
        
        Args:
            session_id: UUID string or UUID object identifying the session
        """
        if isinstance(session_id, UUID):
            session_id = str(session_id)
        
        # Remove from active sessions if present
        self.active_sessions.discard(session_id)
        
        # Remove collection mapping
        self.session_collections.pop(session_id, None)
        
        print(f"MEMORY RESET FOR SESSION: {session_id}")
    
    def get_session_history(self, session_id: str | UUID) -> list:
        """
        Get the conversation history for a specific session.
        
        Args:
            session_id: UUID string or UUID object identifying the session
            
        Returns:
            list: List of messages in the session
        """
        if isinstance(session_id, UUID):
            session_id = str(session_id)
        
        config = {"configurable": {"thread_id": session_id}}
        
        # Get the current state for this session
        try:
            state = self.graph.get_state(config)
            return state.values.get("messages", [])
        except Exception as e:
            print(f"Error retrieving history: {e}")
            return []

    def list_collections(self) -> list:
        """
        List available Chroma collection names for the current persistent directory.

        Returns:
            list: List of collection names (strings). Empty list on error.
        """
        try:
            # Access underlying chromadb client through the LangChain wrapper
            client = getattr(self.vector_store, "_client", None)
            if client is None:
                return []
            collections = client.list_collections()
            return [getattr(c, "name", None) for c in collections if getattr(c, "name", None)]
        except Exception as e:
            print(f"Error listing collections: {e}")
            return []


# Example usage
if __name__ == "__main__":
    # Create agent instance
    agent = TechnicianRAGAgent()
    
    # Create a session ID
    session_1 = str(uuid.uuid4())
    
    # Ask questions in the same session
    response1 = agent.invoke(
            question="I have paper jammed in my printer",
            session_id=session_1,
            collection_name="Canon_MF753Cdw"
        )
    print(f"\nAnswer 1: {response1['answer']}\n")
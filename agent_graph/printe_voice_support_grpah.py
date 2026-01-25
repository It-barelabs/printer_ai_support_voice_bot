from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import StateGraph, START, END, MessagesState, add_messages
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import Annotated, List, TypedDict
from langchain_chroma import Chroma
from pydantic import BaseModel, Field
from enum import Enum
from dotenv import load_dotenv
from uuid import UUID
import os
import re
from loguru import logger

load_dotenv()


# ============================================================================
# State Schema Definitions
# ============================================================================

class DialoguePhase(str, Enum):
    """Tracks the macro-state of the conversation."""
    GATHERING = "gathering"
    RETRIEVING = "retrieving"
    SOLVING = "solving"
    HANDOFF = "handoff"


class PrinterSlots(BaseModel):
    """Stores extracted entities from user input."""
    model_name: str | None = Field(default=None, description="Printer model name (e.g., 'HP OfficeJet', 'Canon TS9120')")
    error_code: str | None = Field(default=None, description="Error code if applicable (e.g., 'Error 50', '0x8004005')")
    symptom_description: str | None = Field(default=None, description="Description of the problem (e.g., 'paper jam', 'won't print')")
    os_environment: str | None = Field(default=None, description="Operating system (e.g., 'Windows 11', 'macOS', 'Linux')")


class SupportAgentState(MessagesState):
    """Main state schema for the support agent."""
    # Slot Register
    printer_slots: PrinterSlots
    
    # Runtime Configuration
    collection_name: str  # ChromaDB collection name (provided at initialization)
    missing_info: List[str]  # Priority queue of missing slots
    question_attempts: dict[str, int]  # Track how many times each slot has been asked
    
    # Dialogue Phase
    dialogue_phase: DialoguePhase
    
    # Retrieval Results
    retrieved_docs: List[Document]
    
    # Final Answer
    answer: str


# ============================================================================
# Voice Support Agent Class
# ============================================================================

class VoiceSupportAgent:
    """
    A voice-enabled printer support agent with iterative slot filling.
    Gathers information question-by-question, then retrieves solutions from ChromaDB.
    """
    
    def __init__(
        self,
        persist_dir: str = "chroma_store",
        collection_name: str = "HP_Color_LaserJet"
    ):
        """
        Initialize the VoiceSupportAgent.
        
        Args:
            persist_dir: Directory path for the Chroma vector store
            collection_name: Name of the ChromaDB collection to use
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
        
        # Initialize LLM for structured output and generation
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
        
        # Initialize memory checkpointer
        self.memory = InMemorySaver()
        
        # Build the graph
        self.graph = self._build_graph()
        
        # Track active sessions
        self.active_sessions = set()
    
    def _build_graph(self):
        """Build the LangGraph workflow with iterative slot filling."""
        graph_builder = StateGraph(SupportAgentState)
        
        # Add nodes
        graph_builder.add_node("analyze_input", self._analyze_input)
        graph_builder.add_node("supervise_slots", self._supervise_slots)
        graph_builder.add_node("generate_question", self._generate_question)
        graph_builder.add_node("retrieve_manual", self._retrieve_manual)
        graph_builder.add_node("synthesize_solution", self._synthesize_solution)
        
        # Add edges
        graph_builder.add_edge(START, "analyze_input")
        graph_builder.add_edge("analyze_input", "supervise_slots")
        
        # Conditional edge from supervisor
        graph_builder.add_conditional_edges(
            "supervise_slots",
            self._route_after_supervisor,
            {
                "ask_question": "generate_question",
                "retrieve": "retrieve_manual"
            }
        )
        
        graph_builder.add_edge("generate_question", END)
        graph_builder.add_edge("retrieve_manual", "synthesize_solution")
        graph_builder.add_edge("synthesize_solution", END)
        
        # Compile with memory checkpointer
        return graph_builder.compile(checkpointer=self.memory)
    
    # ========================================================================
    # Node Implementations
    # ========================================================================
    
    def _analyze_input(self, state: SupportAgentState):
        """Extract entities from the latest user message."""
        print(f"ANALYZING INPUT...\n{'='*20}")
        
        # Get the latest user message
        messages = state.get("messages", [])
        if not messages:
            # No messages yet, return current state
            return {}
        
        # Find the latest HumanMessage
        latest_message = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                latest_message = msg
                break
        
        if not latest_message:
            return {}
        
        user_text = latest_message.content.lower()
        
        # Check if user indicates they don't know (don't extract in this case)
        dont_know_phrases = [
            "i don't know", "i don't have", "i'm not sure", "i'm unsure",
            "don't know", "not sure", "unsure", "no idea", "can't tell",
            "i don't remember", "i forgot", "not available", "unavailable"
        ]
        user_doesnt_know = any(phrase in user_text for phrase in dont_know_phrases)
        
        # Use LLM with structured output to extract slots
        extraction_prompt = ChatPromptTemplate([
            ("system", """You are a precise data extraction engine. Analyze the user's transcript.
Your goal is to extract technical entities into the defined JSON schema.
If a user corrects a previous statement (e.g., 'I meant Windows, not Mac'), prioritize the latest information.
If the user says they don't know or are unsure, return null for that field.
Ignore polite conversation ('Hello', 'Thanks').
Extract only what is explicitly stated or clearly implied."""),
            ("human", "User transcript: {user_input}\n\nExtract printer information from this message.")
        ])
        
        # Get existing slots for context
        existing_slots = state.get("printer_slots", PrinterSlots())
        question_attempts = state.get("question_attempts", {})
        
        # Create structured output chain
        structured_llm = self.llm.with_structured_output(PrinterSlots)
        
        # Prepare context about existing slots
        context = f"Previously extracted: Model={existing_slots.model_name}, Error={existing_slots.error_code}, Symptom={existing_slots.symptom_description}, OS={existing_slots.os_environment}"
        
        messages_prompt = extraction_prompt.invoke({
            "user_input": f"{context}\n\nLatest message: {latest_message.content}"
        })
        
        # Extract slots (this will merge/overwrite as needed)
        extracted_slots = structured_llm.invoke(messages_prompt)
        
        # Merge with existing slots (new values overwrite old ones)
        merged_slots = PrinterSlots(
            model_name=extracted_slots.model_name or existing_slots.model_name,
            error_code=extracted_slots.error_code or existing_slots.error_code,
            symptom_description=extracted_slots.symptom_description or existing_slots.symptom_description,
            os_environment=extracted_slots.os_environment or existing_slots.os_environment
        )
        
        # If user said they don't know, increment attempt counter for the slot we're currently asking about
        if user_doesnt_know:
            missing = state.get("missing_info", [])
            if missing:
                current_slot = missing[0]
                question_attempts = question_attempts.copy()
                question_attempts[current_slot] = question_attempts.get(current_slot, 0) + 1
                print(f"User doesn't know {current_slot}. Attempt count: {question_attempts[current_slot]}")
            else:
                question_attempts = question_attempts.copy()
        
        print(f"Extracted slots: Model={merged_slots.model_name}, Symptom={merged_slots.symptom_description}")
        
        return {
            "printer_slots": merged_slots,
            "question_attempts": question_attempts,
            "dialogue_phase": DialoguePhase.GATHERING
        }
    
    def _supervise_slots(self, state: SupportAgentState):
        """Check slot completeness and determine next action."""
        print(f"SUPERVISING SLOTS...\n{'='*20}")
        
        slots = state.get("printer_slots", PrinterSlots())
        question_attempts = state.get("question_attempts", {})
        missing = []
        
        # Algorithm 1: Slot Prioritization Logic
        # 1. Critical Dependency: Collection Selection (model_name)
        # Skip if asked twice already
        if not slots.model_name:
            attempts = question_attempts.get("model_name", 0)
            if attempts < 2:
                missing.append("model_name")
            else:
                print(f"Skipping model_name (asked {attempts} times)")
        
        # 2. Critical Dependency: Search Query (symptom)
        # This is the most critical - we need at least this to proceed
        if not slots.symptom_description:
            attempts = question_attempts.get("symptom_description", 0)
            if attempts < 2:
                missing.append("symptom_description")
            else:
                print(f"Skipping symptom_description (asked {attempts} times) - proceeding anyway")
        
        # 3. Contextual Refinement: Error Code (conditional)
        symptom = (slots.symptom_description or "").lower()
        if ("error" in symptom or "code" in symptom or "0x" in symptom) and not slots.error_code:
            attempts = question_attempts.get("error_code", 0)
            if attempts < 2:
                missing.append("error_code")
            else:
                print(f"Skipping error_code (asked {attempts} times)")
        
        # 4. Environment Refinement: OS (conditional - for connectivity issues)
        connectivity_keywords = ["connect", "network", "wireless", "wifi", "driver", "install", "setup"]
        if any(keyword in symptom for keyword in connectivity_keywords) and not slots.os_environment:
            attempts = question_attempts.get("os_environment", 0)
            if attempts < 2:
                missing.append("os_environment")
            else:
                print(f"Skipping os_environment (asked {attempts} times)")
        
        print(f"Missing slots: {missing}")
        
        # If we have symptom_description, we can proceed even if other slots are missing
        can_proceed = bool(slots.symptom_description)
        
        return {
            "missing_info": missing,
            "dialogue_phase": DialoguePhase.RETRIEVING if can_proceed and not missing else DialoguePhase.GATHERING
        }
    
    def _route_after_supervisor(self, state: SupportAgentState) -> str:
        """Route based on slot completeness and whether we can proceed."""
        missing = state.get("missing_info", [])
        slots = state.get("printer_slots", PrinterSlots())
        
        # If we have symptom_description (critical for search), proceed even if other info is missing
        has_symptom = bool(slots.symptom_description)
        
        # If no missing slots, or we have symptom and can proceed
        if not missing or (has_symptom and not missing):
            return "retrieve"
        else:
            return "ask_question"
    
    def _generate_question(self, state: SupportAgentState):
        """Generate a single focused question for the top missing slot."""
        print(f"GENERATING QUESTION...\n{'='*20}")
        
        missing = state.get("missing_info", [])
        if not missing:
            return {"answer": "I have all the information I need."}
        
        top_missing = missing[0]  # Get highest priority missing slot
        slots = state.get("printer_slots", PrinterSlots())
        question_attempts = state.get("question_attempts", {})
        attempts = question_attempts.get(top_missing, 0)
        
        # Map slot names to user-friendly questions
        question_map = {
            "model_name": "What is the model name of your printer?",
            "symptom_description": "What problem are you experiencing with your printer?",
            "error_code": "What error code or error message is displayed?",
            "os_environment": "What operating system are you using?"
        }
        
        # If this is the second attempt, make the question more flexible
        if attempts == 1:
            question_map = {
                "model_name": "Do you know the brand or model of your printer? If not, that's okay, I can try to help anyway.",
                "symptom_description": "Can you describe what's happening with your printer? Even a brief description helps.",
                "error_code": "Is there any error message or code on the screen? If not, that's fine.",
                "os_environment": "What operating system are you using? If you're not sure, that's okay."
            }
        
        # Generate contextual question using LLM
        attempt_context = "This is the second time asking this question. Make it clear that it's okay if they don't know." if attempts == 1 else ""
        
        question_prompt = ChatPromptTemplate([
            ("system", """You are a helpful technical support assistant.
Generate a single, concise, friendly question to ask the user.
The question should be optimized for voice interaction - short and clear.
Do not ask multiple questions at once.
{attempt_context}"""),
            ("human", """The user needs to provide: {missing_slot}
Context so far: Model={model}, Symptom={symptom}, Error={error}, OS={os}
{attempt_context}
Generate a natural, conversational question to ask for this information.""")
        ])
        
        messages = question_prompt.invoke({
            "missing_slot": top_missing,
            "model": slots.model_name or "unknown",
            "symptom": slots.symptom_description or "unknown",
            "error": slots.error_code or "none",
            "os": slots.os_environment or "unknown",
            "attempt_context": attempt_context
        })
        
        response = self.llm.invoke(messages)
        question_text = response.content.strip()
        
        # Fallback to simple question if LLM fails
        if not question_text or len(question_text) < 10:
            question_text = question_map.get(top_missing, f"Can you tell me about {top_missing}?")
        
        # Increment attempt counter
        updated_attempts = question_attempts.copy()
        updated_attempts[top_missing] = attempts + 1
        
        print(f"Generated question (attempt {updated_attempts[top_missing]}): {question_text}")
        
        return {
            "answer": question_text,
            "messages": [AIMessage(content=question_text)],
            "question_attempts": updated_attempts
        }
    
    def _retrieve_manual(self, state: SupportAgentState):
        """Dynamically connect to ChromaDB collection and retrieve relevant docs."""
        print(f"RETRIEVING FROM MANUAL...\n{'='*20}")
        
        collection_name = state.get("collection_name")
        slots = state.get("printer_slots", PrinterSlots())
        
        if not collection_name:
            # If no collection name, try to use a default or proceed with what we have
            logger.warning("No collection name provided, using default")
            collection_name = self.collection_name
        
        # Build search query from slots
        query_parts = []
        if slots.symptom_description:
            query_parts.append(slots.symptom_description)
        if slots.error_code:
            query_parts.append(slots.error_code)
        if slots.model_name:
            # Include model name in query if available for better context
            query_parts.append(slots.model_name)
        
        if not query_parts:
            # If we have no query, create a generic one
            query_parts = ["printer problem", "troubleshooting"]
            logger.warning("No specific query available, using generic search")
        
        search_query = " ".join(query_parts)
        print(f"Searching collection '{collection_name}' with query: {search_query}")
        
        # Dynamically instantiate ChromaDB connection to specific collection
        vector_store = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
        
        # Perform similarity search
        retrieved_docs = vector_store.similarity_search(search_query, k=5)
        
        if retrieved_docs:
            print(f"Retrieved {len(retrieved_docs)} documents")
            print(f"First doc preview: {retrieved_docs[0].page_content[:100]}...")
        else:
            print("No documents retrieved")
        
        return {
            "retrieved_docs": retrieved_docs,
            "dialogue_phase": DialoguePhase.SOLVING
        }
    
    def _synthesize_solution(self, state: SupportAgentState):
        """Generate voice-optimized step-by-step solution."""
        print(f"SYNTHESIZING SOLUTION...\n{'='*20}")
        
        retrieved_docs = state.get("retrieved_docs", [])
        slots = state.get("printer_slots", PrinterSlots())
        
        if not retrieved_docs:
            return {
                "answer": "I couldn't find specific information about this issue in the manual. Please contact technical support.",
                "messages": [AIMessage(content="I couldn't find specific information about this issue in the manual. Please contact technical support.")]
            }
        
        # Combine retrieved documents
        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
        
        # Build context message acknowledging missing information if needed
        missing_info_note = ""
        if not slots.model_name:
            missing_info_note = "Note: The printer model is not known, so provide general troubleshooting steps."
        if not slots.error_code and slots.symptom_description and ("error" in slots.symptom_description.lower() or "code" in slots.symptom_description.lower()):
            missing_info_note += " Note: No specific error code was provided."
        
        # Generate solution with voice-optimized prompt
        solution_prompt = ChatPromptTemplate([
            ("system", """You are a voice support assistant. You have retrieved the technical manual.
Your goal is to explain the solution to the user.

Rules:
1. Be extremely concise. Do not read the whole page.
2. Present the solution as a numbered list of steps (1, 2, 3...).
3. Do not use Markdown or special characters (no *, #, _, `).
4. Remove references to images or figures (the user cannot see them).
5. Use simple, clear language suitable for voice output.
6. Keep each step short and actionable.
7. Limit to the most important 5-7 steps."""),
            ("human", """Printer Model: {model}
Problem: {symptom}
Error Code: {error}
{missing_note}

Retrieved Manual Content:
{context}

Generate a concise, step-by-step solution optimized for voice output.""")
        ])
        
        messages = solution_prompt.invoke({
            "model": slots.model_name or "printer",
            "symptom": slots.symptom_description or "issue",
            "error": slots.error_code or "none",
            "context": docs_content,
            "missing_note": missing_info_note if missing_info_note else ""
        })
        
        response = self.llm.invoke(messages)
        solution_text = response.content.strip()
        
        # Parse and clean the output for voice
        cleaned_solution = self._clean_voice_output(solution_text)
        
        print(f"Generated solution: {cleaned_solution[:100]}...")
        
        return {
            "answer": cleaned_solution,
            "messages": [AIMessage(content=cleaned_solution)]
        }
    
    def _clean_voice_output(self, text: str) -> str:
        """Remove Markdown and format for TTS."""
        # Remove Markdown syntax
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Italic
        text = re.sub(r'#+\s*', '', text)  # Headers
        text = re.sub(r'`([^`]+)`', r'\1', text)  # Code
        text = re.sub(r'_{2,}', '', text)  # Underline
        text = re.sub(r'~~([^~]+)~~', r'\1', text)  # Strikethrough
        
        # Remove visual references
        text = re.sub(r'\(see [Ff]igure [\d.]+\)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\(see [Ff]ig\.? [\d.]+\)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # Links
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)  # Images
        
        # Normalize list formatting
        text = re.sub(r'^[-*+]\s+', 'Step ', text, flags=re.MULTILINE)
        
        # Clean up extra whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    # ========================================================================
    # Public API
    # ========================================================================
    
    def invoke(
        self,
        user_input: str,
        session_id: str | UUID,
        collection_name: str | None = None
    ) -> dict:
        """
        Process a user input within a session.
        
        Args:
            user_input: The user's message (text from STT)
            session_id: UUID string or UUID object identifying the session
            collection_name: Optional collection name (uses instance default if not provided)
            
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
        
        # Determine collection to use
        target_collection = collection_name if collection_name else self.collection_name
        
        # Check if this is a new session
        is_new_session = session_id not in self.active_sessions
        
        if is_new_session:
            print(f"NEW SESSION: {session_id}")
            self.active_sessions.add(session_id)
        
        print(f"SESSION: {session_id} | COLLECTION: {target_collection}")
        
        # Get current state or initialize
        config = {"configurable": {"thread_id": session_id}}
        
        try:
            current_state = self.graph.get_state(config)
            existing_state = current_state.values if current_state.values else {}
        except:
            existing_state = {}
        
        # Initialize state if new session, otherwise merge with existing
        if is_new_session or not existing_state:
            initial_slots = PrinterSlots()
            existing_collection = target_collection
            question_attempts = {}
        else:
            initial_slots = existing_state.get("printer_slots", PrinterSlots())
            existing_collection = existing_state.get("collection_name", target_collection)
            question_attempts = existing_state.get("question_attempts", {})
        
        # Prepare state for invocation (only new message, state will be merged)
        state_input = {
            "messages": [HumanMessage(content=user_input)],
            "printer_slots": initial_slots,
            "collection_name": existing_collection,  # Use existing collection if available
            "missing_info": existing_state.get("missing_info", []),
            "question_attempts": question_attempts,
            "dialogue_phase": existing_state.get("dialogue_phase", DialoguePhase.GATHERING),
            "retrieved_docs": existing_state.get("retrieved_docs", []),
            "answer": existing_state.get("answer", "")
        }
        
        # Invoke graph
        response = self.graph.invoke(state_input, config=config)
        
        return response
    
    def reset_memory(self, session_id: str | UUID):
        """
        Reset/clear memory for a specific session.
        
        Args:
            session_id: UUID string or UUID object identifying the session
        """
        if isinstance(session_id, UUID):
            session_id = str(session_id)
        
        self.active_sessions.discard(session_id)
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
        
        try:
            state = self.graph.get_state(config)
            return state.values.get("messages", [])
        except Exception as e:
            print(f"Error retrieving history: {e}")
            return []


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import uuid
    
    # Create agent instance with collection name
    agent = VoiceSupportAgent(
        persist_dir="agent_graph/chroma_store",
        collection_name="Canon_MF753Cdw"
    )
    
    # Create a session ID
    session_1 = str(uuid.uuid4())
    # Interactive chat through keyboard interface in the terminal
    print("\n=== Keyboard Chat with VoiceSupportAgent ===")
    print("Type your message and press Enter. Type 'exit' or 'quit' to end the chat.\n")

    collection = "Canon_MF753Cdw"
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            print("Exiting chat.")
            break
        response = agent.invoke(
            user_input=user_input,
            session_id=session_1,
            collection_name=collection
        )
        print(f"Bot: {response.get('answer', 'No answer')}\n")


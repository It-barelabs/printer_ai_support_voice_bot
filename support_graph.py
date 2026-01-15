from typing import TypedDict, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv(override=True)

class EssayState(TypedDict):
    # Chat History
    messages: List[str]
    # Collected Data
    topic: str
    tone: str
    audience: str
    word_count: int
    # Deliverables
    thesis_statement: str
    outline: str
    final_draft: str
    # Control Flags
    current_stage: str  # "intake", "approval", "thesis", "outline", "writing"


# Initialize a fast model (good for real-time voice)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Pydantic model for structured extraction
class ExtractedInfo(BaseModel):
    """Structured information extracted from user messages"""
    topic: Optional[str] = Field(None, description="A concise summary of the essay topic (paraphrased, not exact quote)")
    word_count: Optional[int] = Field(None, description="The desired word count if mentioned")
    tone: Optional[str] = Field(None, description="A concise summary of the desired tone (paraphrased, not exact quote)")
    audience: Optional[str] = Field(None, description="A concise summary of the target audience (paraphrased, not exact quote)")

# Create structured output LLM
structured_llm = llm.with_structured_output(ExtractedInfo)

def intake_node(state: EssayState):
    """
    Analyzes user input to extract info, then asks the next question.
    """
    
    # Get current state values
    current_topic = state.get("topic", "")
    current_tone = state.get("tone", "")
    current_audience = state.get("audience", "")
    current_word_count = state.get("word_count", 0)
    messages = state.get("messages", [])
    
    # 1. Extract information from user's last message if it exists
    updated_topic = current_topic
    updated_tone = current_tone
    updated_audience = current_audience
    updated_word_count = current_word_count
    
    # If there's a user message, extract information from it
    if messages:
        last_user_message = messages[-1] if isinstance(messages[-1], str) else messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
        
        # Use structured LLM to extract information from the user's message
        extraction_prompt = f"""Analyze the following user message and extract information about their essay requirements.

Current known information:
- Topic: {current_topic if current_topic else "Not provided"}
- Word Count: {current_word_count if current_word_count > 0 else "Not provided"}
- Tone: {current_tone if current_tone else "Not provided"}
- Audience: {current_audience if current_audience else "Not provided"}

User's message: "{last_user_message}"

IMPORTANT: Extract and SUMMARIZE any new information provided. Do NOT use the exact text from the user's message.
Instead, create a concise, clear summary that captures the essence of what they said.
- For topic: Summarize the main subject/theme in 1-2 sentences
- For tone: Summarize the desired writing style (e.g., "formal and academic", "casual and conversational")
- For audience: Summarize the target readers (e.g., "college students", "general public", "academic researchers")

Only include fields that were mentioned or updated in the user's message. If a field was not mentioned, leave it as null."""
        
        try:
            # Use structured output - returns ExtractedInfo directly
            extracted = structured_llm.invoke([SystemMessage(content=extraction_prompt)])
            
            # Update state fields with extracted information
            if extracted.topic:
                updated_topic = extracted.topic
            if extracted.word_count is not None and extracted.word_count > 0:
                updated_word_count = extracted.word_count
            if extracted.tone:
                updated_tone = extracted.tone
            if extracted.audience:
                updated_audience = extracted.audience
        except Exception as e:
            # If extraction fails, continue with current values
            print(f"Warning: Could not extract structured info: {e}")
    
    # 2. System Prompt: Instructions for Voice Interaction
    system_prompt = """
    You are an interactive essay writing assistant. Your current goal is to collect 
    the following information from the user: Topic, Word Count, Tone, and Audience.
    
    Current gathered info:
    - Topic: {topic}
    - Word Count: {word_count}
    - Tone: {tone}
    - Audience: {audience}
    
    INSTRUCTIONS:
    1. If this is the first message (no user messages yet), greet the user and ask for the FIRST missing piece of information.
    2. If the user has sent a message, acknowledge it and check what is still missing.
    3. Ask for ONE missing piece of information.
    4. IMPORTANT: Since this is a voice interface, keep your question SHORT (under 15 words).
    5. If all info is collected (topic, word_count > 0, tone, and audience are all provided), say "INTAKE_COMPLETE".
    """
    
    # Format the prompt with updated state
    word_count_display = updated_word_count if updated_word_count > 0 else "Unknown"
    topic_display = updated_topic if updated_topic else "Unknown"
    tone_display = updated_tone if updated_tone else "Unknown"
    audience_display = updated_audience if updated_audience else "Unknown"
    
    formatted_prompt = system_prompt.format(
        topic=topic_display,
        word_count=word_count_display,
        tone=tone_display,
        audience=audience_display
    )
    
    # Convert string messages to HumanMessage objects if needed
    message_objects = [SystemMessage(content=formatted_prompt)]
    
    # If there are no messages, this is the first turn - the LLM should start the conversation
    # Otherwise, add all previous messages
    if messages:
        for msg in messages:
            if isinstance(msg, str):
                message_objects.append(HumanMessage(content=msg))
            else:
                message_objects.append(msg)
    
    # 3. Call the LLM to generate response
    response = llm.invoke(message_objects)
    content = response.content
    
    # 4. Check if all information is collected
    all_collected = (
        updated_topic and 
        updated_word_count > 0 and 
        updated_tone and 
        updated_audience
    ) or "INTAKE_COMPLETE" in content
    
    # 5. Prepare return state with all updates
    # Preserve all existing state fields and update only what changed
    return_state = {
        "topic": updated_topic,
        "tone": updated_tone,
        "audience": updated_audience,
        "word_count": updated_word_count,
        "messages": messages + [content],
        "thesis_statement": state.get("thesis_statement", ""),
        "outline": state.get("outline", ""),
        "final_draft": state.get("final_draft", "")
    }
    
    if all_collected:
        # Move to approval stage instead of directly to thesis
        return_state["current_stage"] = "approval"
    else:
        return_state["current_stage"] = state.get("current_stage", "intake")
    
    return return_state



# Create the graph builder
workflow = StateGraph(EssayState)

# Add our node
workflow.add_node("intake_agent", intake_node)

# Define the Conditional Logic (The Router)
def route_step(state: EssayState):
    # If the intake agent changed the stage to 'thesis', move there
    if state.get("current_stage") == "thesis":
        return "thesis_agent" 
    return END # Or loop back to wait for user input

# Add edges
workflow.set_entry_point("intake_agent")

# In a real app, you would connect this to the Thesis Node
# workflow.add_node("thesis_agent", thesis_node)
# workflow.add_conditional_edges("intake_agent", route_step)

# For now, add edge to END so graph can terminate properly
# The conditional logic is handled in the interactive loop
workflow.add_edge("intake_agent", END)

# Compile the graph
app = workflow.compile()

# Interactive conversation loop
def run_interactive_intake():
    """
    Run the intake process interactively, asking the user questions one at a time.
    """
    print("="*60)
    print("Essay Writing Assistant - Intake Phase")
    print("="*60)
    print("I'll help you create an essay. Let me collect some information.\n")
    
    # Initialize state with required fields
    current_state = {
        "messages": [],
        "topic": "",
        "tone": "",
        "audience": "",
        "word_count": 0,
        "thesis_statement": "",
        "outline": "",
        "final_draft": "",
        "current_stage": "intake"
    }
    
    # First turn: Agent starts the conversation
    result = app.invoke(current_state)
    current_state.update(result)
    
    # Extract and print the agent's first message
    if result.get("messages"):
        latest_message = result["messages"][-1]
        print(f"Assistant: {latest_message}\n")
    
    # Continue conversation loop until intake is complete
    while current_state.get("current_stage") == "intake":
        # Get user input
        user_input = input("You: ").strip()
        
        if user_input.lower() in ["quit", "exit", "stop"]:
            print("Exiting...")
            break
        
        # Add user message to state - preserve all existing state
        current_state["messages"] = current_state.get("messages", []) + [user_input]
        
        # Invoke the graph again with full current state
        # LangGraph will merge the returned state with the input state
        result = app.invoke(current_state)
        
        # Update current_state with all returned fields (merge)
        for key, value in result.items():
            current_state[key] = value
        
        # Check if intake is complete and move to approval
        if current_state.get("current_stage") == "approval":
            break
        
        # Print agent's response
        if result.get("messages"):
            latest_message = result["messages"][-1]
            print(f"Assistant: {latest_message}\n")
    
    # Approval stage: Show summary and get user approval
    while current_state.get("current_stage") == "approval":
        print("\n" + "="*60)
        print("SUMMARY OF COLLECTED INFORMATION")
        print("="*60)
        print(f"\nTopic: {current_state.get('topic', 'N/A')}")
        print(f"Word Count: {current_state.get('word_count', 0)}")
        print(f"Tone: {current_state.get('tone', 'N/A')}")
        print(f"Audience: {current_state.get('audience', 'N/A')}")
        print("="*60)
        
        # Get user approval
        approval = input("\nDo you approve this information? (yes/no): ").strip().lower()
        
        if approval in ["yes", "y", "approve", "ok"]:
            print("\n✓ Information approved! Moving to thesis generation...")
            current_state["current_stage"] = "thesis"
            break
        elif approval in ["no", "n", "reject"]:
            print("\nLet's revise the information. What would you like to change?")
            revision_input = input("You: ").strip()
            
            if revision_input.lower() in ["quit", "exit", "stop"]:
                print("Exiting...")
                break
            
            # Add revision to messages and go back to intake
            current_state["messages"] = current_state.get("messages", []) + [revision_input]
            current_state["current_stage"] = "intake"
            
            # Process the revision - continue intake loop
            while current_state.get("current_stage") == "intake":
                # Invoke the graph to process the revision
                result = app.invoke(current_state)
                for key, value in result.items():
                    current_state[key] = value
                
                # If moved to approval, break to show updated summary
                if current_state.get("current_stage") == "approval":
                    break
                
                # Print agent's response and get more input if needed
                if result.get("messages"):
                    latest_message = result["messages"][-1]
                    print(f"Assistant: {latest_message}\n")
                
                # Get additional user input if still in intake
                if current_state.get("current_stage") == "intake":
                    user_input = input("You: ").strip()
                    if user_input.lower() in ["quit", "exit", "stop"]:
                        print("Exiting...")
                        return current_state
                    current_state["messages"] = current_state.get("messages", []) + [user_input]
        else:
            print("Please answer 'yes' or 'no'.")
    
    return current_state

if __name__ == "__main__":
    final_state = run_interactive_intake()
    print("\nFinal State:", final_state)

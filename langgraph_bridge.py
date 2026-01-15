import asyncio
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    InputAudioRawFrame,
    OutputAudioRawFrame,
    TextFrame,
    LLMMessagesFrame,
    Frame,
    EndFrame,
    StartFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    LLMRunFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    LLMContextFrame
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContextFrame
from langchain_core.messages import HumanMessage, AIMessage
import logging
from loguru import logger

logging.getLogger("pipecat.transports.smallwebrtc").setLevel(logging.WARNING)


class LangGraphProcessor(FrameProcessor):
    def __init__(self, graph, thread_id: str = "default_thread"):
        super().__init__()
        self.graph = graph
        # LangGraph config for persistent state
        self.config = {"configurable": {"thread_id": thread_id}}

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """
        Standard Pipecat method to intercept frames in the pipeline.
        """

        await super().process_frame(frame, direction)

        # 1. Ignore raw audio frames - LangGraph doesn't need them
        if isinstance(frame, (InputAudioRawFrame, OutputAudioRawFrame)):
            await self.push_frame(frame, direction)
            return
        # 1. Detect when the user has finished speaking
        if isinstance(frame, LLMMessagesFrame):
            # Pipecat's ContextAggregator sends this frame when VAD detects silence
            messages = frame.messages
            if not messages:
                return
            
            # Get the last user message from the Pipecat context
            last_msg = messages[-1]
            if last_msg["role"] != "user":
                return

            user_input = last_msg["content"]
            print(f"🎤 User said: {user_input}")

            # 2. Invoke LangGraph
            # stream_mode="messages" is crucial for low-latency voice
            async for chunk in self.graph.astream(
                {"messages": [HumanMessage(content=user_input)]},
                self.config,
                stream_mode="messages"
            ):
                # 3. Convert LangGraph chunks -> Pipecat TextFrames
                if isinstance(chunk, tuple):
                    msg_chunk, _ = chunk
                    # Ensure we only speak the AI's content
                    if isinstance(msg_chunk, AIMessage) and msg_chunk.content:
                        await self.push_frame(TextFrame(text=msg_chunk.content))
            
        # Pass other system frames (EndFrame, etc.) through untouched
        elif isinstance(frame, (EndFrame, StartFrame)):
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)


class LangGraphBridge(FrameProcessor):
    def __init__(self, app, thread_id: str, initial_state: dict = None):
        super().__init__()
        self.app = app
        self.thread_id = thread_id
        self.config = {"configurable": {"thread_id": thread_id}}
        # Initialize current_state with the provided initial_state or default
        self.current_state = initial_state or {
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
        # Print initial state structure
        print("\n" + "="*80)
        print("📊 LangGraph Bridge Initial State Structure:")
        print("="*80)
        for key, value in self.current_state.items():
            if isinstance(value, list):
                print(f"  {key}: list (length: {len(value)})")
            elif isinstance(value, dict):
                print(f"  {key}: dict (keys: {list(value.keys())})")
            else:
                print(f"  {key}: {type(value).__name__} = {value}")
        print("="*80 + "\n")

    def _extract_text_from_message(self, message):
        """Extract text content from various message formats."""
        if isinstance(message, str):
            return message
        
        if isinstance(message, dict):
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                # Extract text from list of content parts
                texts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            texts.append(part.get("text", ""))
                        elif part.get("text"):
                            texts.append(part["text"])
                    elif isinstance(part, str):
                        texts.append(part)
                return " ".join(texts) if texts else None
        
        # Handle Content objects (Google format)
        if hasattr(message, "parts"):
            parts = getattr(message, "parts", [])
            texts = []
            for part in parts:
                if hasattr(part, "text") and part.text:
                    texts.append(part.text)
            return " ".join(texts) if texts else None
        
        # Handle objects with content attribute
        if hasattr(message, "content"):
            content = getattr(message, "content")
            if isinstance(content, str):
                return content
        
        return None

    def _convert_context_to_langgraph_state(self, context):
        """
        Convert context to LangGraph state format.
        
        Returns:
            Tuple of (state dict, has_user_message)
        """
        state = self.current_state.copy()
        has_user_message = False
        
        # Get messages from context
        context_messages = []
        if hasattr(context, "messages"):
            context_messages = context.messages
        elif hasattr(context, "_messages"):
            context_messages = context._messages
        
        if not context_messages:
            return state, False
        
        # Get the last message
        last_msg = context_messages[-1] if context_messages else None
        if not last_msg:
            return state, False
        
        # Extract role and content
        role = None
        content = None
        
        if isinstance(last_msg, dict):
            role = last_msg.get("role")
            content = self._extract_text_from_message(last_msg)
        elif hasattr(last_msg, "role"):
            role = getattr(last_msg, "role")
            content = self._extract_text_from_message(last_msg)
        
        # Only process user messages
        if role == "user" and content:
            has_user_message = True
            # Add user message to state
            current_messages = state.get("messages", [])
            if isinstance(current_messages, list):
                state["messages"] = current_messages + [content]
            else:
                state["messages"] = [content]
        
        return state, has_user_message

    async def _process_context(self, context):
        """
        Process a context and generate a response.
        Only the last message from the final result dict is sent to TTS.
        
        Args:
            context: The LLM context containing conversation history
        """
        await self.push_frame(LLMFullResponseStartFrame())
        
        try:
            # Convert context to LangGraph state
            state, has_user_message = self._convert_context_to_langgraph_state(context)
            
            logger.info(f"🔍 Converted context: has_user_message={has_user_message}, state keys: {list(state.keys())}")
            
            if not has_user_message:
                logger.debug("No user message in context, skipping")
                await self.push_frame(LLMFullResponseEndFrame())
                return
            
            logger.info(f"🎤 Processing context with state: {state.get('messages', [])}")
            
            # Update current state
            self.current_state = state
            
            # Get final result from LangGraph (no streaming, wait for complete response)
            logger.info(f"🚀 Calling LangGraph (waiting for final result)")
            
            try:
                if hasattr(self.app, 'ainvoke'):
                    final_result = await self.app.ainvoke(state, self.config)
                else:
                    import asyncio
                    final_result = await asyncio.to_thread(self.app.invoke, state, self.config)
                
                # Print the returned state structure
                print("\n" + "="*80)
                print("📊 LangGraph Bridge Final State Structure (returned from graph):")
                print("="*80)
                for key, value in final_result.items():
                    if isinstance(value, list):
                        print(f"  {key}: list (length: {len(value)})")
                        if value and len(value) <= 5:
                            print(f"    Contents: {value}")
                    elif isinstance(value, dict):
                        print(f"  {key}: dict (keys: {list(value.keys())})")
                    else:
                        value_str = str(value)
                        if len(value_str) > 100:
                            value_str = value_str[:100] + "..."
                        print(f"  {key}: {type(value).__name__} = {value_str}")
                print("="*80 + "\n")
                
                # Extract only the last message from the messages list
                bot_response_text = None
                if "messages" in final_result and final_result["messages"]:
                    messages = final_result["messages"]
                    # Get the last message (should be the bot's response)
                    last_msg = messages[-1]
                    
                    # Extract text from the last message
                    if isinstance(last_msg, AIMessage) and last_msg.content:
                        bot_response_text = last_msg.content
                    elif isinstance(last_msg, str):
                        bot_response_text = last_msg
                    elif hasattr(last_msg, "content"):
                        bot_response_text = str(last_msg.content)
                    elif isinstance(last_msg, dict):
                        bot_response_text = last_msg.get("content", "")
                    
                    if bot_response_text:
                        bot_response_text = str(bot_response_text).strip()
                
                # Send only the last message to TTS
                if bot_response_text:
                    logger.info(f"📤 Sending last message to TTS: {bot_response_text[:100]}...")
                    await self.push_frame(LLMTextFrame(bot_response_text))
                else:
                    logger.warning("⚠️ No bot response found in final result messages")
                
                # Update state with final result
                self.current_state.update(final_result)
                
            except Exception as e:
                logger.error(f"❌ Error calling LangGraph: {e}", exc_info=True)
                import traceback
                logger.error(traceback.format_exc())
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}", exc_info=True)
            import traceback
            logger.error(traceback.format_exc())
        finally:
            await self.push_frame(LLMFullResponseEndFrame())

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """
        Process incoming frames from the pipeline.
        
        Handles:
        - LLMContextFrame: Context from aggregator
        - OpenAILLMContextFrame: OpenAI-style context
        - TranscriptionFrame: Direct transcription
        - LLMRunFrame: Initial run trigger
        """
        # Always push frame downstream first (matching LLMService behavior)
        await super().process_frame(frame, direction)
        
        context = None
        
        # Handle context frames from aggregator - match GoogleLLMService exactly
        if isinstance(frame, OpenAILLMContextFrame):
            context = frame.context
        elif isinstance(frame, LLMContextFrame):
            context = frame.context
        else:
            # Push other frames through
            await self.push_frame(frame, direction)
        
        if context:
            await self._process_context(context)
    
    async def stop(self, frame):
        """Clean up on stop."""
        await super().stop(frame)
    
    async def cancel(self, frame):
        """Clean up on cancel."""
        await super().cancel(frame)
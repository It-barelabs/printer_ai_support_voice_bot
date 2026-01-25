import asyncio
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    Frame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    LLMContextFrame
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContextFrame
from langchain_core.messages import AIMessage
import logging
from loguru import logger

logging.getLogger("pipecat.transports.smallwebrtc").setLevel(logging.WARNING)


class LangGraphBridge(FrameProcessor):
    def __init__(
        self, 
        app, 
        thread_id: str, 
        initial_state: dict = None, 
        transport=None, 
        webrtc_connection=None,
        collection_name: str = None
    ):
        super().__init__()
        # Handle both class and instance - if class, raise helpful error
        if isinstance(app, type):
            raise ValueError(
                "LangGraphBridge requires an instance of VoiceSupportAgent, not the class. "
                "Please instantiate it first: VoiceSupportAgent(persist_dir='...', collection_name='...')"
            )
        self.app = app  # This should be a VoiceSupportAgent instance
        self.thread_id = thread_id
        self.config = {"configurable": {"thread_id": thread_id}}
        self.transport = transport
        self.webrtc_connection = webrtc_connection
        self.collection_name = collection_name
        
        # Initialize current_state with the provided initial_state or default for VoiceSupportAgent
        self.current_state = initial_state or {
            "messages": [],
            "printer_slots": None,
            "collection_name": collection_name or "",
            "missing_info": [],
            "dialogue_phase": "gathering",
            "retrieved_docs": [],
            "answer": ""
        }

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

    def _extract_user_input_from_context(self, context):
        """
        Extract user input text from context for VoiceSupportAgent.
        
        Returns:
            Tuple of (user_input string, has_user_message bool)
        """
        has_user_message = False
        user_input = None
        
        # Get messages from context
        context_messages = []
        if hasattr(context, "messages"):
            context_messages = context.messages
        elif hasattr(context, "_messages"):
            context_messages = context._messages
        
        if not context_messages:
            return None, False
        
        # Get the last message
        last_msg = context_messages[-1] if context_messages else None
        if not last_msg:
            return None, False
        
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
            user_input = content
        
        return user_input, has_user_message

    def _format_final_result_for_display(self, final_result):
        """Format final_result dict as a readable string for display."""
        lines = []
        lines.append("Final State Structure:")
        lines.append("-" * 80)
        for key, value in final_result.items():
            if isinstance(value, list):
                lines.append(f"  {key}: list (length: {len(value)})")
                if value and len(value) <= 3:
                    for i, item in enumerate(value):
                        if isinstance(item, str):
                            item_str = item[:50] + "..." if len(item) > 50 else item
                        else:
                            item_str = str(item)[:50] + "..." if len(str(item)) > 50 else str(item)
                        lines.append(f"    [{i}]: {item_str}")
            elif isinstance(value, dict):
                lines.append(f"  {key}: dict (keys: {list(value.keys())})")
            else:
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:100] + "..."
                lines.append(f"  {key}: {value_str}")
        lines.append("-" * 80)
        return "\n".join(lines)
    
    def _format_final_result_as_json(self, final_result):
        """Format final_result as JSON string for webpage."""
        import json
        # Convert to serializable format
        serializable_result = {}
        for key, value in final_result.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                serializable_result[key] = value
            elif isinstance(value, list):
                # Convert list items to strings if needed
                serializable_result[key] = [
                    str(item) if not isinstance(item, (str, int, float, bool, type(None))) else item
                    for item in value[:10]  # Limit to first 10 items
                ]
            elif isinstance(value, dict):
                serializable_result[key] = {k: str(v) for k, v in value.items()}
            elif hasattr(value, 'model_dump'):  # Pydantic model
                serializable_result[key] = value.model_dump()
            elif hasattr(value, 'dict'):  # Pydantic model (older API)
                serializable_result[key] = value.dict()
            else:
                serializable_result[key] = str(value)
        
        try:
            return json.dumps(serializable_result, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error serializing final_result: {e}")
            return str(serializable_result)
    
    async def _send_final_result_to_webpage(self, final_result):
        """Send final_result to webpage via data channel.
        This method is non-blocking and won't break the pipeline if it fails.
        """
        try:
            if not self.webrtc_connection:
                return
            
            import json
            final_result_json = self._format_final_result_as_json(final_result)
            final_result_str = self._format_final_result_for_display(final_result)
            final_result_dict = json.loads(final_result_json)
            
            message = {
                "label": "langgraph-bridge",
                "type": "final-result",
                "data": {
                    "final_result": final_result_dict,
                    "formatted": final_result_str
                }
            }
            
            if hasattr(self.webrtc_connection, 'send_app_message'):
                self.webrtc_connection.send_app_message(message)
                return
            
            # Fallback: try transport._webrtc_connection
            if hasattr(self.transport, '_webrtc_connection') and self.transport._webrtc_connection:
                if hasattr(self.transport._webrtc_connection, 'send_app_message'):
                    self.transport._webrtc_connection.send_app_message(message)
                    return
            
        except Exception as e:
            logger.debug(f"Error sending final_result to webpage (non-fatal): {e}")

    async def _process_context(self, context):
        """
        Process a context and generate a response using VoiceSupportAgent.
        The answer field from the response is sent to TTS.
        
        Args:
            context: The LLM context containing conversation history
        """
        await self.push_frame(LLMFullResponseStartFrame())
        
        try:
            # Extract user input from context
            user_input, has_user_message = self._extract_user_input_from_context(context)
            
            if not has_user_message or not user_input:
                await self.push_frame(LLMFullResponseEndFrame())
                return
            
            # Call VoiceSupportAgent.invoke() method
            try:
                # Check if app has invoke method (VoiceSupportAgent) or is a graph (direct LangGraph)
                if hasattr(self.app, 'invoke'):
                    # VoiceSupportAgent interface
                    if hasattr(self.app, 'ainvoke'):
                        # Async version if available
                        final_result = await self.app.ainvoke(
                            user_input=user_input,
                            session_id=self.thread_id,
                            collection_name=self.collection_name
                        )
                    else:
                        # Sync version - run in thread
                        final_result = await asyncio.to_thread(
                            self.app.invoke,
                            user_input=user_input,
                            session_id=self.thread_id,
                            collection_name=self.collection_name
                        )
                else:
                    # Direct LangGraph app (fallback for compatibility)
                    # Convert context to state format
                    state = self.current_state.copy()
                    state["messages"] = state.get("messages", []) + [user_input]
                    
                    if hasattr(self.app, 'ainvoke'):
                        final_result = await self.app.ainvoke(state, self.config)
                    else:
                        final_result = await asyncio.to_thread(self.app.invoke, state, self.config)
                
                # Extract bot response text
                bot_response_text = None
                
                # First, try to get answer field (VoiceSupportAgent returns this)
                if "answer" in final_result and final_result["answer"]:
                    bot_response_text = str(final_result["answer"]).strip()
                
                # Fallback: Extract from messages list
                if not bot_response_text and "messages" in final_result and final_result["messages"]:
                    messages = final_result["messages"]
                    # Find the last AIMessage
                    for msg in reversed(messages):
                        if isinstance(msg, AIMessage) and msg.content:
                            bot_response_text = str(msg.content).strip()
                            break
                        elif isinstance(msg, str):
                            bot_response_text = str(msg).strip()
                            break
                        elif hasattr(msg, "content"):
                            bot_response_text = str(msg.content).strip()
                            break
                        elif isinstance(msg, dict):
                            bot_response_text = str(msg.get("content", "")).strip()
                            if bot_response_text:
                                break
                
                # Send the response to TTS
                if bot_response_text:
                    await self.push_frame(LLMTextFrame(bot_response_text))
                else:
                    logger.warning("No bot response text found in final_result")
                
                # Update state with final result
                self.current_state.update(final_result)
                
                # Send final_result to webpage via data channel (non-blocking)
                await self._send_final_result_to_webpage(final_result)
                
            except Exception as e:
                logger.error(f"Error calling VoiceSupportAgent: {e}", exc_info=True)
            
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
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
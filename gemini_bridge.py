"""
Standalone Gemini Vertex AI Bridge for Pipecat.

This bridge connects to Google Gemini models through Vertex AI without
depending on pipecat's GoogleVertexLLMService or GoogleLLMService.

Usage Example:
    ```python
    from gemini_bridge import GeminiVertexBridge
    
    # Initialize the bridge
    bridge = GeminiVertexBridge(
        project_id="your-project-id",
        location="us-east4",
        model="gemini-2.5-flash",
        system_instruction="You are a helpful assistant.",
        temperature=0.7,
        max_tokens=1024,
    )
    
    # Add to your Pipecat pipeline
    pipeline = Pipeline([
        # ... other processors ...
        bridge,
        # ... other processors ...
    ])
    ```

Authentication:
    The bridge supports three authentication methods:
    1. Environment variable: Set GOOGLE_APPLICATION_CREDENTIALS to path of service account JSON
    2. credentials_path: Pass path to service account JSON file
    3. credentials: Pass JSON string of service account credentials
"""

import json
import os
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

# Suppress gRPC fork warnings
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"

from loguru import logger
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    InterimTranscriptionFrame,
    LLMContextFrame,
    LLMRunFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    StartFrame,
    EndFrame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContextFrame

try:
    from google.auth import default
    from google.auth.exceptions import GoogleAuthError
    from google.auth.transport.requests import Request
    from google.genai import Client
    from google.genai.types import (
        Content,
        Part,
        GenerateContentConfig,
        HttpOptions,
    )
    from google.oauth2 import service_account
except ImportError as e:
    logger.error(f"Missing required Google AI dependencies: {e}")
    logger.error("Install with: pip install google-genai google-auth google-auth-oauthlib")
    raise


@dataclass
class GeminiVertexConfig:
    """Configuration for Gemini Vertex AI bridge."""
    
    project_id: str
    location: str = "us-east4"
    model: str = "gemini-2.5-flash"
    credentials: Optional[str] = None
    credentials_path: Optional[str] = None
    system_instruction: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = 4096
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    http_options: Optional[HttpOptions] = None


class GeminiVertexBridge(FrameProcessor):
    """
    Standalone bridge to Google Gemini via Vertex AI.
    
    This bridge processes frames from the Pipecat pipeline and generates
    responses using Gemini models through Vertex AI, without depending on
    pipecat's Google LLM services.
    """
    
    def __init__(
        self,
        project_id: str,
        location: str = "us-east4",
        model: str = "gemini-2.5-flash",
        credentials: Optional[str] = None,
        credentials_path: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = 4096,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        http_options: Optional[HttpOptions] = None,
    ):
        """
        Initialize the Gemini Vertex AI bridge.
        
        Args:
            project_id: Google Cloud project ID (required)
            location: GCP region for Vertex AI endpoint (default: "us-east4")
            model: Model identifier (default: "gemini-2.5-flash")
            credentials: JSON string of service account credentials
            credentials_path: Path to service account JSON file
            system_instruction: System instruction/prompt for the model
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            http_options: HTTP options for the client
        """
        super().__init__()
        
        self._project_id = project_id
        self._location = location
        self._model = model
        self._system_instruction = system_instruction
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._top_p = top_p
        self._top_k = top_k
        
        # Get credentials
        self._credentials = self._get_credentials(credentials, credentials_path)
        
        # Create client
        self._client = Client(
            vertexai=True,
            credentials=self._credentials,
            project=self._project_id,
            location=self._location,
            http_options=http_options,
        )
        
        # Note: We don't maintain conversation history here - the context aggregator does that
        
        logger.info(
            f"✅ GeminiVertexBridge initialized: model={model}, "
            f"project={project_id}, location={location}"
        )
    
    @staticmethod
    def _get_credentials(
        credentials: Optional[str],
        credentials_path: Optional[str]
    ):
        """
        Retrieve Google service account credentials.
        
        Supports:
        1. Direct JSON credentials string
        2. Path to service account JSON file
        3. Default application credentials (ADC)
        
        Args:
            credentials: JSON string of service account credentials
            credentials_path: Path to service account JSON file
            
        Returns:
            Google credentials object
            
        Raises:
            ValueError: If no valid credentials found
        """
        creds = None
        
        if credentials:
            # Parse JSON string
            creds = service_account.Credentials.from_service_account_info(
                json.loads(credentials),
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        elif credentials_path:
            # Load from file
            creds = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        else:
            # Try default credentials
            try:
                creds, _ = default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
            except GoogleAuthError:
                pass
        
        if not creds:
            raise ValueError(
                "No valid credentials provided. Set GOOGLE_APPLICATION_CREDENTIALS "
                "environment variable, or provide credentials/credentials_path."
            )
        
        # Refresh token
        creds.refresh(Request())
        
        return creds
    
    def _build_generation_config(self, system_instruction: Optional[str] = None) -> GenerateContentConfig:
        """Build generation configuration for Gemini API."""
        params = {}
        
        # Use provided system instruction or instance default
        effective_system = system_instruction or self._system_instruction
        if effective_system:
            params["system_instruction"] = effective_system
        
        if self._temperature is not None:
            params["temperature"] = self._temperature
        
        if self._max_tokens:
            params["max_output_tokens"] = self._max_tokens
        
        if self._top_p is not None:
            params["top_p"] = self._top_p
        
        if self._top_k is not None:
            params["top_k"] = self._top_k
        
        return GenerateContentConfig(**params)
    
    def _extract_text_from_message(self, message: Any) -> Optional[str]:
        """
        Extract text content from various message formats.
        
        Handles:
        - Dict format: {"role": "user", "content": "text"}
        - Content objects with parts
        - String messages
        """
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
        
        # Handle Content objects
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
    
    def _convert_message_to_gemini_content(self, message: Any) -> Optional[Content]:
        """
        Convert a message from various formats to Gemini Content format.
        
        Handles:
        - Dict format: {"role": "user", "content": "text"}
        - Content objects (already in Gemini format)
        - Objects with role and content/parts attributes
        """
        # If already a Content object, return as-is
        if isinstance(message, Content):
            return message
        
        # Extract role
        role = None
        if isinstance(message, dict):
            role = message.get("role")
        elif hasattr(message, "role"):
            role = getattr(message, "role")
        
        if not role:
            return None
        
        # Convert role: Gemini uses "model" instead of "assistant"
        gemini_role = "model" if role == "assistant" else role
        
        # Extract text content
        text = self._extract_text_from_message(message)
        if not text:
            return None
        
        return Content(role=gemini_role, parts=[Part(text=text)])
    
    def _convert_context_to_gemini_messages(self, context: Any) -> Tuple[List[Content], Optional[str]]:
        """
        Convert a context object to Gemini Content messages.
        
        Returns:
            Tuple of (messages list, system_instruction)
        """
        messages = []
        system_instruction = self._system_instruction
        
        # Get messages from context
        context_messages = []
        if hasattr(context, "messages"):
            context_messages = context.messages
        elif hasattr(context, "_messages"):
            context_messages = context._messages
        
        # Get system message
        if hasattr(context, "system_message") and context.system_message:
            system_instruction = context.system_message
        
        # Convert each message to Gemini format
        for msg in context_messages:
            # Skip system messages (handled separately)
            if isinstance(msg, dict) and msg.get("role") == "system":
                if not system_instruction:
                    system_instruction = self._extract_text_from_message(msg)
                continue
            elif hasattr(msg, "role") and getattr(msg, "role") == "system":
                if not system_instruction:
                    system_instruction = self._extract_text_from_message(msg)
                continue
            
            # Convert message to Gemini Content
            gemini_content = self._convert_message_to_gemini_content(msg)
            if gemini_content:
                messages.append(gemini_content)
        
        return messages, system_instruction
    
    async def _process_context(self, context: Any):
        """
        Process a context and generate a response.
        
        Args:
            context: The LLM context containing conversation history
        """
        await self.push_frame(LLMFullResponseStartFrame())
        
        accumulated_text = ""
        
        try:
            # Convert context to Gemini format
            messages, system_instruction = self._convert_context_to_gemini_messages(context)
            
            logger.info(f"🔍 Converted context: {len(messages)} messages, system_instruction: {bool(system_instruction)}")
            
            # Update system instruction if provided
            if system_instruction:
                self._system_instruction = system_instruction
            
            # Only process if we have messages and last one is from user
            if not messages:
                logger.warning("⚠️ No messages in context")
                await self.push_frame(LLMFullResponseEndFrame())
                return
            
            last_msg = messages[-1]
            logger.info(f"🔍 Last message role: {last_msg.role}")
            
            if last_msg.role != "user":
                logger.debug(f"Skipping - last message is not from user: {last_msg.role}")
                await self.push_frame(LLMFullResponseEndFrame())
                return
            
            logger.info(f"🎤 Processing context with {len(messages)} messages")
            
            # Build generation config
            generation_config = self._build_generation_config()
            
            logger.info(f"🚀 Calling Gemini API with model: {self._model}")
            
            # Get the async iterator from the streaming call
            response_stream = await self._client.aio.models.generate_content_stream(
                model=self._model,
                contents=messages,
                config=generation_config,
            )
            
            chunk_count = 0
            async for chunk in response_stream:
                chunk_count += 1
                if not chunk.candidates:
                    continue
                
                for candidate in chunk.candidates:
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if part.text:
                                accumulated_text += part.text
                                logger.debug(f"📝 Pushing text chunk: {part.text[:50]}...")
                                await self.push_frame(LLMTextFrame(part.text))
            
            logger.info(f"✅ Generated response ({chunk_count} chunks): {accumulated_text[:100] if accumulated_text else 'empty'}...")
            
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
        await self._cleanup()
    
    async def cancel(self, frame):
        """Clean up on cancel."""
        await super().cancel(frame)
        await self._cleanup()
    
    async def _cleanup(self):
        """Clean up client resources."""
        try:
            if hasattr(self, "_client") and hasattr(self._client, "aio"):
                await self._client.aio.aclose()
        except Exception as e:
            logger.debug(f"Error during cleanup: {e}")


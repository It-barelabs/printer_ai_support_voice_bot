


#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
import os
import sys
import asyncio

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMRunFrame, LLMTextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.google import GeminiLiveVertexLLMService, GoogleVertexLLMService
# from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from gemini_bridge import GeminiVertexBridge
from langgraph_bridge import LangGraphBridge
from support_graph import app as support_graph
import logging

# Configure logging
logging.getLogger("pipecat.transports.smallwebrtc").setLevel(logging.WARNING)
logging.getLogger("langgraph_bridge").setLevel(logging.INFO)
logging.getLogger("support_graph").setLevel(logging.INFO)

load_dotenv(override=True)

SYSTEM_INSTRUCTION = f"""
"You are Printer support Chatbot "Easy print", a friendly, helpful support bot that help users solve issues with their printer.

Your output will be converted to audio so don't include special characters in your answers.

Respond to what the user said in a creative and helpful way. Keep your responses brief. One or two sentences at most.
"""

    

from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.frames.frames import (
    TextFrame,
    TranscriptionFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    LLMContextFrame,
    LLMRunFrame
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContextFrame





async def run_bot(webrtc_connection):
    logger.info("🚀 Starting bot with WebRTC connection")
    logger.info(f"🔧 WebRTC connection ID: {webrtc_connection.pc_id}")

    # if printer_info:
    #     logger.info(f"🖨️ Printer info received: {printer_info}")

    pipecat_transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            audio_out_10ms_chunks=2,
            # Enable sending messages through data channel
            enable_messaging=True,
            # Configure data channel for RTVI messages
            enable_transcription_reporting=True,
        ),
    )
    logger.info("✅ SmallWebRTC transport created with messaging enabled")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY", ""),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID", ""),
            params=ElevenLabsTTSService.InputParams(
                output_format="pcm_16000" # 16kHz with 320 samples per buffer (20ms chunks)
            )
        )
    # Initialize LangGraphBridge with proper state schema
    initial_state = {
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
    # bridge = LangGraphBridge(app=support_graph, thread_id="session_01", initial_state=initial_state)

    # llm = GeminiLiveVertexLLMService(project_id="voice-document-builder", location="global",
    #     voice_id="Puck",  # Aoede, Charon, Fenrir, Kore, Puck
    #     transcribe_model_audio=True,
    #     system_instruction=SYSTEM_INSTRUCTION,
    # )

    # llm = GoogleVertexLLMService(project_id="voice-document-builder", location="global")
    # llm = GeminiVertexBridge(project_id="voice-document-builder", location="global", model="gemini-2.5-flash", system_instruction=SYSTEM_INSTRUCTION)

    bridge = LangGraphBridge(app=support_graph, thread_id="session_01", initial_state=initial_state)

    system_content = "Start by greeting the user warmly and introducing yourself. use short response if possible and be friendly and helpful."
    # if printer_info:
    #     system_content += f"\n\nHere is information about the printer:\n{printer_info}"

    context = LLMContext(
        [
            {
                "role": "system",
                "content": system_content,
            }
        ],
    )
    context_aggregator = LLMContextAggregatorPair(context)

    # Add RTVI processor for sending transcripts to the client
    # RTVI uses an observer pattern to capture frames from anywhere in the pipeline
    # We pass transport so it can send messages to the client
    rtvi = RTVIProcessor(
        config=RTVIConfig(config=[]),
        # transport=pipecat_transport
    )
    
    pipeline = Pipeline(
        [
            pipecat_transport.input(),
            rtvi,
            stt,  # Speech-to-text: transcribes user audio
            context_aggregator.user(),
            bridge,  
            # llm,
            tts,  # Text-to-speech: generates bot audio
            pipecat_transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],  # Observer to send transcripts
    )

    @pipecat_transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("🔌 Pipecat Client connected")
        logger.info(f"🔌 Transport type: {type(transport)}")
        logger.info(f"🔌 Client info: {client}")

        # Check if data channel exists
        if hasattr(transport, '_dc') and transport._dc:
            logger.info(f"✅ Data channel exists: {transport._dc}")
            logger.info(f"✅ Data channel label: {transport._dc.label}")
            logger.info(f"✅ Data channel state: {transport._dc.readyState}")
        else:
            logger.warning("⚠️ No data channel found on transport!")

        # Kick off the conversation.
        await task.queue_frames([LLMRunFrame()])

    @pipecat_transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Pipecat Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)
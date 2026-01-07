


#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
import os
import sys

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.google import GeminiLiveVertexLLMService, GoogleVertexLLMService
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

load_dotenv(override=True)

SYSTEM_INSTRUCTION = f"""
"You are Gemini Chatbot, a friendly, helpful robot.

Your goal is to demonstrate your capabilities in a succinct way.

Your output will be converted to audio so don't include special characters in your answers.

Respond to what the user said in a creative and helpful way. Keep your responses brief. One or two sentences at most.
"""


async def run_bot(webrtc_connection):
    logger.info("🚀 Starting bot with WebRTC connection")
    logger.info(f"🔧 WebRTC connection ID: {webrtc_connection.pc_id}")

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

    # llm = GeminiLiveVertexLLMService(project_id="voice-document-builder", location="global",
    #     voice_id="Puck",  # Aoede, Charon, Fenrir, Kore, Puck
    #     transcribe_model_audio=True,
    #     system_instruction=SYSTEM_INSTRUCTION,
    # )

    llm = GoogleVertexLLMService(project_id="voice-document-builder", location="global")

    context = LLMContext(
        [
            {
                "role": "user",
                "content": "Start by greeting the user warmly and introducing yourself.",
            }
        ],
    )
    context_aggregator = LLMContextAggregatorPair(context)

    # Add RTVI processor for sending transcripts to the client
    # RTVI uses an observer pattern to capture frames from anywhere in the pipeline
    # It needs to be in the pipeline and have access to the transport for SmallWebRTC
    rtvi = RTVIProcessor(
        config=RTVIConfig(config=[]),
        transport=pipecat_transport  # Pass transport so RTVI can send via data channel
    )

    pipeline = Pipeline(
        [
            pipecat_transport.input(),
            stt,  # Speech-to-text: transcribes user audio
            rtvi,  # RTVI processor - observer will capture STT transcriptions and TTS text
            context_aggregator.user(),
            llm,  # LLM
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
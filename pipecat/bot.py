import os
import asyncio
import numpy as np
import whisper

from dotenv import load_dotenv
from loguru import logger

print("🚀 Starting Pipecat bot...")
print("⏳ Loading models and imports (20 seconds, first run only)\n")

logger.info("Loading Local Smart Turn Analyzer V3...")
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

logger.info("✅ Local Smart Turn Analyzer V3 loaded")
logger.info("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer

logger.info("✅ Silero VAD model loaded")

from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame, Frame, AudioRawFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

logger.info("Loading pipeline components...")
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
# from pipecat.transports.daily.transport import DailyParams  # Commented out - using WebRTC instead

logger.info("✅ All components loaded successfully!")

# Custom Whisper STT Processor
class WhisperSTTService(FrameProcessor):
    def __init__(self, model_size: str = "base"):
        super().__init__()
        logger.info(f"Loading Whisper {model_size} model...")
        self._model = whisper.load_model(model_size)
        self._audio_buffer = bytearray()
        self._sample_rate = 16000
        logger.info("✅ Whisper model loaded")
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, AudioRawFrame):
            # Accumulate audio
            self._audio_buffer.extend(frame.audio)
        
            # Process when we have ~1 second of audio (16000 samples * 2 bytes)
            if len(self._audio_buffer) >= 32000:
                try:
                    # Convert bytes to numpy array
                    audio_np = np.frombuffer(bytes(self._audio_buffer), dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Transcribe
                    result = await asyncio.to_thread(
                        self._model.transcribe,
                        audio_np,
                        language="en",
                        fp16=False
                    )
                    
                    text = result["text"].strip()
                    if text:
                        logger.info(f"🎤 Whisper transcription: {text}")
                        # Use current time in ISO format for timestamp
                        import datetime
                        timestamp = datetime.datetime.now().isoformat()
                        await self.push_frame(TranscriptionFrame(text=text, user_id="user", timestamp=timestamp))
                    
                    # Clear buffer
                    self._audio_buffer.clear()
                    
                except Exception as e:
                    logger.error(f"Whisper transcription error: {e}")
                    self._audio_buffer.clear()
        
        await self.push_frame(frame, direction)

load_dotenv(override=True)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    stt = WhisperSTTService(model_size="base")

    tts = DeepgramTTSService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        voice="aura-asteria-en",  # Natural female voice
    )

    from pipecat.services.openai.base_llm import BaseOpenAILLMService

    llm = OpenAILLMService(
        api_key="ollama",  
        base_url="http://localhost:11434/v1",  # Ollama's API endpoint
        model="llama3.2:1b",  # Local Ollama model
        params=BaseOpenAILLMService.InputParams(max_tokens=2048)
    )

    messages = [
        {
            "role": "system",
            "content": "You are a friendly AI assistant. Respond naturally and keep your answers conversational.",
        },
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            rtvi,  # RTVI processor
            stt,
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Say hello and briefly introduce yourself."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point for the bot starter."""

    transport_params = {
        # Removed Daily transport - using WebRTC instead (no Rust compiler needed)
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
    }

    transport = await create_transport(runner_args, transport_params)

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
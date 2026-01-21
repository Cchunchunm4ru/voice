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
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
import nemo.collections.asr as nemo_asr
logger.info("✅ All components loaded successfully!")

load_dotenv(override=True)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    # Define a simple customSTTService wrapper for the ASR model

    import soundfile as sf
    import librosa
    import torch

    class AudioChunkIterator:
        def __init__(self, samples, chunk_len_in_secs, sample_rate):
            self._samples = samples
            self._chunk_len = int(chunk_len_in_secs * sample_rate)
            self._start = 0
            self.output = True
        def __iter__(self):
            return self
        def __next__(self):
            if not self.output:
                raise StopIteration
            last = int(self._start + self._chunk_len)
            if last <= len(self._samples):
                chunk = self._samples[self._start: last]
                self._start = last
            else:
                chunk = np.zeros([int(self._chunk_len)], dtype='float32')
                samp_len = len(self._samples) - self._start
                chunk[0:samp_len] = self._samples[self._start:len(self._samples)]
                self.output = False
            return chunk


    class customSTTService(FrameProcessor):
        def __init__(self, model, sample_rate=16000, chunk_len_in_secs=15, context_len_in_secs=2):
            super().__init__()
            self.model = model
            self.sample_rate = sample_rate
            self.chunk_len_in_secs = chunk_len_in_secs
            self.context_len_in_secs = context_len_in_secs

        def link(self, next_processor):
            self._next = next_processor

        async def process(self, frame, direction="forward"):
            from loguru import logger
            logger.debug(f"customSTTService.process called with frame type: {type(frame).__name__} | frame={frame}")
            # Log all attributes for debugging
            for attr in dir(frame):
                if not attr.startswith("_"):
                    try:
                        logger.debug(f"  frame.{attr} = {getattr(frame, attr)}")
                    except Exception:
                        pass
            if isinstance(frame, AudioRawFrame):
                logger.debug(f"AudioRawFrame received: samples shape={getattr(frame, 'samples', None).shape if hasattr(frame, 'samples') else 'N/A'}, sample_rate={getattr(frame, 'sample_rate', None)}")
                # Save audio to temp file
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    sf.write(tmp.name, frame.samples, frame.sample_rate)
                    tmp_path = tmp.name
                async for tframe in self.run_stt(tmp_path):
                    logger.debug(f"Yielding TranscriptionFrame: {tframe.text}")
                    await self._next.process(tframe, direction)
            else:
                logger.debug("Non-audio frame received, passing through.")
                await self._next.process(frame, direction)

        def get_samples(self, audio_file, target_sr=16000):
            with sf.SoundFile(audio_file, 'r') as f:
                sample_rate = f.samplerate
                samples = f.read()
                if sample_rate != target_sr:
                    samples = librosa.core.resample(samples, orig_sr=sample_rate, target_sr=target_sr)
                samples = samples.transpose()
                return samples

        async def run_stt(self, audio_file):
            # audio_file: path to .wav file
            samples = self.get_samples(audio_file, self.sample_rate)
            chunk_len = self.chunk_len_in_secs
            context_len = self.context_len_in_secs
            buffer_len_in_secs = chunk_len + 2 * context_len
            buffer_len = int(self.sample_rate * buffer_len_in_secs)
            chunk_reader = AudioChunkIterator(samples, chunk_len, self.sample_rate)
            chunk_len_samples = int(self.sample_rate * chunk_len)
            context_len_samples = int(self.sample_rate * context_len)
            sampbuffer = np.zeros([buffer_len], dtype=np.float32)
            transcripts = []
            for chunk in chunk_reader:
                # Shift buffer and add new chunk
                sampbuffer[:-chunk_len_samples] = sampbuffer[chunk_len_samples:]
                sampbuffer[-chunk_len_samples:] = chunk
                # Only transcribe the middle part (chunk) for best context
                buffer_for_model = sampbuffer.copy()
                # Model expects shape (batch, time)
                input_tensor = torch.tensor(buffer_for_model, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    result = self.model.transcribe([input_tensor])[0]
                transcripts.append(result.text)
                # Yield partial transcription for this chunk
                yield TranscriptionFrame(text=result.text, is_final=False)
            # Merge all chunk transcriptions
            full_transcript = ' '.join(transcripts)
            yield TranscriptionFrame(text=full_transcript, is_final=True)

    # Force model to load on CPU to avoid CUDA errors
    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3", map_location="cpu")
    model = model.to("cpu")
    stt = customSTTService(
        model = model
    )

    tts = DeepgramTTSService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        voice="aura-asteria-en",  # Natural female voice
    )

    from pipecat.services.openai.base_llm import BaseOpenAILLMService

    llm = OpenAILLMService(
        api_key="ollama",  
        base_url="http://localhost:11434/v1",  # Ollama's API endpoint
        model="llama3.2:1b",  
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

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # Direct to STT
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
        observers=[],
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
        ),
    }

    transport = await create_transport(runner_args, transport_params)

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
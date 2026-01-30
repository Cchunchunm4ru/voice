import os
import asyncio
import numpy as np
import torch
import datetime
from loguru import logger
from dotenv import load_dotenv
from threading import Lock
from livekit import api
import queue
import threading
from scipy import signal
import requests
from flask import Flask, jsonify, render_template,request
import nemo.collections.asr as nemo_asr
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.pipeline.runner import PipelineRunner
from pipecat.frames.frames import LLMRunFrame  
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.openai import OpenAILLMService
from pipecat.services.deepgram import DeepgramTTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from flask_cors import CORS  # Import this!
app = Flask(__name__)
load_dotenv()
origins = [
    "http://localhost:3000",
]

CORS(app, resources={r"/*": {"origins": origins}})
load_dotenv()

SR = 16_000
CHUNK_SECONDS = 4
CHUNK_SAMPLES = SR * CHUNK_SECONDS
asr_lock = Lock()

class SharedModel:
    def __init__(self):
        logger.info("Downloading Parakeet-TDT once …")
        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt-0.6b-v2",
            map_location="cpu",
        ).eval()
        with torch.inference_mode():
            _ = self.model.transcribe([np.zeros(SR, dtype=np.float32)])

shared_model = SharedModel()

class SimpleNemoSTT(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    async def setup(self, params):
        """
        Asynchronous setup method to satisfy the Pipeline's requirements.
        """
        await asyncio.sleep(0)  # Placeholder awaitable

    def link(self, next_processor):
        """
        Placeholder link method to satisfy the Pipeline's requirements.
        """
        pass

    def transcribe(self, audio_bytes):
        """
        Transcribe audio using the ASR model.
        """
        audio_fp32 = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        audio_tensor = torch.tensor(audio_fp32).unsqueeze(0)

        with torch.inference_mode():
            hypotheses = self.model.transcribe(tokens_list=[audio_tensor])
            return hypotheses[0] if hypotheses else ""

    async def queue_frame(self, frame, *args, **kwargs):
        """
        Asynchronous queue_frame method to handle frames and satisfy the Pipeline's requirements.
        """
        await asyncio.sleep(0)  

    @app.route('/process', methods=['POST'])
    async def process_audio(self):
        """
        Flask route to process audio input and return transcription.
        """
        audio_bytes = request.data
        transcription = self.transcribe(audio_bytes)
        return jsonify({"transcription": transcription})

async def main():
    
    transport = LocalAudioTransport(LocalAudioTransportParams(audio_in_sample_rate=16000))
    
    logger.info("Loading NeMo...")
    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
    
    stt = SimpleNemoSTT(model=model)
    llm = OpenAILLMService(api_key="ollama", base_url="http://localhost:11434/v1", model="llama3.2:1b")
    tts = DeepgramTTSService(api_key=os.getenv("DEEPGRAM_API_KEY"), voice="aura-asteria-en")

    messages = [{"role": "system", "content": "You are a helpful AI. Keep it brief."}]
    context = OpenAILLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(pipeline)

    async def on_start(transport):
        await task.queue_frames([LLMRunFrame()])

    # Register the event handler on the task, not the transport
    task.add_event_handler("on_start", on_start)

    runner = PipelineRunner()
    await runner.run(task)

class ASRSession:
    def __init__(self):
        self.audio_q = queue.Queue(maxsize=8)
        self.txt_q = queue.Queue()
        self.transcripts = []
        self.active = True
        threading.Thread(target=self._worker, daemon=True).start()

    def close(self):
        self.active = False
        self.audio_q.put(None)

    def _worker(self):
        buf = np.array([], dtype=np.float32)
        while self.active:
            try:
                while len(buf) < CHUNK_SAMPLES and self.active:
                    audio_chunk = self.audio_q.get()
                    if audio_chunk is None:
                        self.active = False
                        break
                    buf = np.concatenate([buf, audio_chunk])
                if not self.active:
                    break
                while len(buf) >= CHUNK_SAMPLES and self.active:
                    chunk, buf = buf[:CHUNK_SAMPLES], buf[CHUNK_SAMPLES:]
                    with torch.inference_mode():
                        with asr_lock:
                            out = shared_model.model.transcribe([chunk])
                    self.txt_q.put(out[0].text)
            except Exception as e:
                logger.error(f"ASR error: {e}")
        while not self.txt_q.empty():
            self.txt_q.get()

    def preprocess(self, audio):
        sr, y = audio
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != SR:
            y = signal.resample_poly(y, SR, sr)
        y = y.astype(np.float32)
        y /= (np.abs(y).max() + 1e-9)
        return y

def stream_fn(audio, state: ASRSession):
    if state.active:
        state.audio_q.put(state.preprocess(audio))
    while not state.txt_q.empty():
        text = state.txt_q.get()
        state.transcripts.append(text)
    return (
        " ".join(state.transcripts) if state.transcripts else "…listening…",
        state,
    )

def send_to_flask(audio, state):
    # Convert audio to bytes
    audio_bytes = audio[1].astype(np.int16).tobytes()
    try:
        # Send audio to Flask endpoint
        response = requests.post("http://127.0.0.1:5000/process", data=audio_bytes)
        if response.status_code == 200:
            transcription = response.text
            state.transcripts.append(transcription)
        else:
            logger.error(f"Flask error: {response.status_code}")
    except Exception as e:
        logger.error(f"Request error: {e}")
    return (
        " ".join(state.transcripts) if state.transcripts else "…listening…",
        state,
    )


@app.route('/')
def index():
    return render_template('index.html')

def generate_livekit_token(room_name):
    """
    Generates the JWT token (hall pass) the browser needs to join.
    No async network calls required!
    """
    try:
        # Log the API key and secret for debugging
        logger.info(f"LIVEKIT_API_KEY: {LIVEKIT_API_KEY}")
        logger.info(f"LIVEKIT_API_SECRET: {LIVEKIT_API_SECRET}")

        grant = api.VideoGrants(
            room_join=True,
            room=room_name,
        )
        access_token = api.AccessToken(
            LIVEKIT_API_KEY,
            LIVEKIT_API_SECRET,
        ).with_grants(grant).with_identity(f"user-{int(datetime.datetime.now().timestamp())}")

        # Generate and return the token
        token = access_token.to_jwt()
        logger.info(f"Generated Token: {token}")
        return token
    except Exception as e:
        logger.error(f"Token generation failed: {e}")
        return None

@app.route('/start', methods=['POST', 'OPTIONS'])
def start():
    if request.method == 'OPTIONS':
        return jsonify({"message": "CORS preflight successful"}), 200

    try:
        room_name = f"room-{int(datetime.datetime.now().timestamp())}"
        token = generate_livekit_token(room_name)
        
        # Convert https:// to wss:// for the connection
        wss_url = LIVEKIT_BASE_URL.replace("https://", "wss://")

        # We provide BOTH formats to be safe
        response_data = {
            "room_name": room_name,
            "token": token,
            "url": wss_url,
            "room_url": wss_url  # <--- This is what your app.js is looking for!
        }
        
        return jsonify(response_data), 200
    except Exception as e:
        logger.error(f"Start route failed: {e}")
        return jsonify({"error": str(e)}), 500

LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_BASE_URL = os.getenv("LIVEKIT_URL", "https://livekit.cloud")

def run_flask():
    # Run Flask on port 8080 to match your frontend's request
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)
    
if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    import time
    time.sleep(2) 
    
    try:
        logger.info("Starting Pipecat Pipeline...")
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Pipeline stopped.")
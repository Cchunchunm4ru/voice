import nemo.collections.asr as nemo_asr
import librosa as lb
import soundfile as sf
import numpy as np
asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
file = r"C:\Users\Admin\Desktop\projects\pipe\ElevenLabs_2026-01-08T08_06_19_old british man_ivc_sp100_s50_sb71_v3.mp3"
y, sr = lb.load(file, sr=16000)
if isinstance(y, np.ndarray):
    if y.ndim == 1:
        y = np.expand_dims(y, axis=1)
elif y is None:
    raise ValueError("Audio data 'y' is None. Check audio loading step.")
else:
    raise TypeError(f"Unexpected type for 'y': {type(y)}")
sf.write("temp.wav", y, sr)
hypotheses = asr_model.transcribe(["temp.wav"])
best_hypothesis = hypotheses[0]
text = best_hypothesis.text
print(text)
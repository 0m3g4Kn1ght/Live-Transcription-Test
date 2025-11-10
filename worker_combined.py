import asyncio
import websockets
import json
import queue
import sounddevice as sd
import numpy as np
import torch
import librosa
from faster_whisper import WhisperModel
from vosk import Model as VoskModel, KaldiRecognizer
import sys
import os

SAMPLE_RATE = 16000
WS_SERVER_URI = "ws://localhost:9000"
MIC_DEVICE = 1  
BUFFER_DURATION = 5  
OVERLAP = 1
FWH_MODEL_SIZE = "tiny.en"
SILENCE_THRESHOLD = 0.01
VOSK_MODEL_PATH = r"F:\Allianz\Projects\Allianz Verint Product\vosk-model-small-en-us-0.15"  

speaker_counter = 0

device_info = sd.query_devices(MIC_DEVICE)
device_name = device_info['name'] if isinstance(device_info, dict) else device_info[1]
print("Using microphone:", device_name)



print(f"⏳ Loading Faster Whisper ({FWH_MODEL_SIZE}) on {'cuda' if torch.cuda.is_available() else 'cpu'} ...")
fwh_device = "cuda" if torch.cuda.is_available() else "cpu"
asr_model = WhisperModel(FWH_MODEL_SIZE, device=fwh_device, compute_type="int8")
print("Faster Whisper ready.\n")

try:
    vosk_model = VoskModel(VOSK_MODEL_PATH)
    vosk_rec = KaldiRecognizer(vosk_model, SAMPLE_RATE)
    vosk_q = queue.Queue()
    print("Vosk model ready for live transcription.\n")
except Exception as e:
    print("Failed to load Vosk model:", e)
    sys.exit(1)



def callback(indata, frames, time, status):
    if status:
        print("!!!", status)
    vosk_q.put(bytes(indata))


async def transcribe_file(ws, file_path):
    global speaker_counter
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Loaded file: {file_path}")
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    print(f"Resampled to {SAMPLE_RATE} Hz")


    start = 0
    chunk_samples = BUFFER_DURATION * SAMPLE_RATE
    while start < len(audio):
        chunk = audio[start:start + chunk_samples]
        start += chunk_samples - (OVERLAP * SAMPLE_RATE)  # overlap

        segments, _ = asr_model.transcribe(chunk, beam_size=5, language="en")
        text = " ".join([seg.text.strip() for seg in segments])
        print(f"Transcribed text: {text}")

        if text:
            speaker_counter += 1
            payload = json.dumps({"speaker": f"Speaker {speaker_counter}", "text": text})
            await ws.send(payload)





async def transcribe_mic(ws):
    global speaker_counter
    print("Listening to microphone... Press Ctrl+C to stop.")
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000, dtype='int16',
                           channels=1, callback=callback):
        while True:
            data = vosk_q.get()
            if vosk_rec.AcceptWaveform(data):
                text = json.loads(vosk_rec.Result()).get("text", "").strip()
            else:
                text = json.loads(vosk_rec.PartialResult()).get("partial", "").strip()

            if text:
                speaker_counter += 1
                payload = json.dumps({"speaker": f"Speaker {speaker_counter}", "text": text})
                print(f"Transcribed text: {text}")
                await ws.send(payload)




async def main():
    try:
        async with websockets.connect(WS_SERVER_URI) as ws:
            print("Connected to WebSocket server\n")

            if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
                await transcribe_file(ws, sys.argv[1])
            else:
                await transcribe_mic(ws)

    except Exception as e:
        print("WebSocket connection failed:", e)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped")

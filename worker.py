import asyncio
import websockets
import json
import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import datetime
import threading

SAMPLE_RATE = 16000
MODEL_PATH = r"F:\Allianz\Projects\Allianz Verint Product\Vosk_en_in\vosk-model-en-us-0.42-gigaspeech"
SERVER_URI = "ws://localhost:9000"

MIC_DEVICE = 1          
SYSTEM_AUDIO_DEVICE = 2

model = Model(MODEL_PATH)
rec_mic = KaldiRecognizer(model, SAMPLE_RATE)
rec_sys = KaldiRecognizer(model, SAMPLE_RATE)
rec_sys = KaldiRecognizer(model, SAMPLE_RATE)
rec_mic = KaldiRecognizer(model, SAMPLE_RATE)

q_mic = queue.Queue()
q_sys = queue.Queue()

def current_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def callback_mic(indata, frames, time, status):
    if status:
        print("Mic status:", status)
    q_mic.put(bytes(indata))

def callback_sys(indata, frames, time, status):
    if status:
        print("System status:", status)
    q_sys.put(bytes(indata))

async def process_audio(ws, recognizer, q, speaker_label):
    while True:
        data = q.get()
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result.get("text", "").strip()
            if text:
                payload = json.dumps({
                    "speaker": speaker_label,
                    "text": text,
                    "timestamp": current_timestamp()
                })
                print(f"{speaker_label}: {text}")
                await ws.send(payload)

async def main():
    async with websockets.connect(SERVER_URI) as ws:
        print("Connected to WebSocket server")

        def mic_stream():
            with sd.RawInputStream(
                samplerate=SAMPLE_RATE, blocksize=4000, dtype="int16",
                channels=1, device=MIC_DEVICE, callback=callback_mic):
                asyncio.run(process_audio(ws, rec_mic, q_mic, "Agent"))

        def sys_stream():
            with sd.RawInputStream(
                samplerate=SAMPLE_RATE, blocksize=4000, dtype="int16",
                channels=1, device=SYSTEM_AUDIO_DEVICE, callback=callback_sys):
                asyncio.run(process_audio(ws, rec_sys, q_sys, "Caller"))

        t1 = threading.Thread(target=mic_stream, daemon=True)
        t2 = threading.Thread(target=sys_stream, daemon=True)
        t1.start()
        t2.start()

        while True:
            await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopped.")

import asyncio
import websockets
import json
import numpy as np
import whisperx
import soundfile as sf
import librosa
import torch



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ASR_MODEL_SIZE = "tiny.en"       
SERVER_URI = "ws://localhost:9000"
LANGUAGE = "en"        
BUFFER_DURATION = 3           
SILENCE_THRESHOLD = 0.01
OVERLAP = 0.5                 
FILE_PATH = "sample.wav"       



print(f"Loading WhisperX model ({ASR_MODEL_SIZE}) on {DEVICE} ...")
asr_model = whisperx.load_model(ASR_MODEL_SIZE, DEVICE, compute_type="int8", language=LANGUAGE)
print("WhisperX ready.\n")

speaker_counter = 0 

async def stream_file(websocket, file_path):
    global speaker_counter

    data, sr = sf.read(file_path)
    print(f"Loaded file: {file_path}, shape={data.shape}, sample rate={sr}")

    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
        print("Converted to mono")

    if sr != 16000:
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        sr = 16000
        print("Resampled to 16kHz")

    data = np.ascontiguousarray(data.astype(np.float32))
    if np.max(np.abs(data)) > 0:
        data /= np.max(np.abs(data))
    print(f"Max amplitude after normalization: {np.max(np.abs(data))}")

    prev_audio = np.zeros(int(sr * OVERLAP), dtype=np.float32)
    samples_per_buffer = BUFFER_DURATION * sr

    for start in range(0, len(data), samples_per_buffer):
        buffer = data[start:start + samples_per_buffer]
        if len(buffer) == 0:
            continue

        buffer = np.concatenate([prev_audio, buffer]).astype(np.float32)
        prev_audio = buffer[-int(sr * OVERLAP):]
        buffer = np.ascontiguousarray(buffer, dtype=np.float32)

        if np.max(np.abs(buffer)) < SILENCE_THRESHOLD:
            continue

        result = asr_model.transcribe(buffer)
        text = result.get("text", "").strip()

        print(f"Transcribed text: {repr(text)}")

        if text:
            speaker_counter += 1
            speaker = f"Speaker {speaker_counter}"
            payload = json.dumps({"speaker": speaker, "text": text})
            print("Sending payload:", payload)
            try:
                await websocket.send(payload)
            except Exception as e:
                print("Failed to send payload:", e)

async def main():
    try:
        async with websockets.connect(SERVER_URI) as ws:
            print("Connected to WebSocket server")
            await stream_file(ws, FILE_PATH)
    except Exception as e:
        print("WebSocket connection failed:", e)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped")

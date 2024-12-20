import os
import wave
import pyaudio
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from rag.AIVoiceAssistant import AIVoiceAssistant
import voice_service as vs
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SILENCE_THRESHOLD = 1000  
SILENCE_DURATION = 1.5  
RECORD_SECONDS = 30  

class AudioRecorder:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.is_recording = False
        self.silence_counter = 0
    
    def start_recording(self):
        """Start recording audio"""
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        self.frames = []
        self.is_recording = True
        self.silence_counter = 0
        print("\nListening...")
        
        while self.is_recording:
            data = self.stream.read(CHUNK, exception_on_overflow=False)
            self.frames.append(data)
            
            audio_data = np.frombuffer(data, dtype=np.int16)
            if np.max(np.abs(audio_data)) < SILENCE_THRESHOLD:
                self.silence_counter += CHUNK / RATE
                if self.silence_counter >= SILENCE_DURATION:
                    self.is_recording = False
            else:
                self.silence_counter = 0
                
            if len(self.frames) * CHUNK / RATE >= RECORD_SECONDS:
                self.is_recording = False
        
        self.stream.stop_stream()
        self.stream.close()
        
        return self.save_audio()
    
    def save_audio(self):
        """Save recorded audio to file"""
        if not self.frames:
            return None
            
        filename = "temp_recording.wav"
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(self.frames))
        return filename
    
    def cleanup(self):
        """Cleanup resources"""
        self.audio.terminate()

def transcribe_audio(file_path):
    """Transcribe audio using OpenAI Whisper API"""
    try:
        with open(file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en"
            )
        return transcript.text.strip()
    except Exception as e:
        print(f"Error during transcription: {e}")
        return ""

def main():
    print("Initializing AI Voice Assistant...")
    ai_assistant = AIVoiceAssistant()
    recorder = AudioRecorder()
    
    try:
        while True:
            audio_file = recorder.start_recording()
            
            if audio_file and os.path.exists(audio_file):
                transcription = transcribe_audio(audio_file)
                os.remove(audio_file)
                
                if transcription:
                    print(f"\nCustomer: {transcription}")
                    
                    response = ai_assistant.interact_with_llm(transcription)
                    print(f"AI Assistant: {response}")
                    
                    if response:
                        vs.play_text_to_speech(response)
                else:
                    print("No speech detected, please try again.")
            
            print("\nPress Ctrl+C to exit or continue speaking...")
            
    except KeyboardInterrupt:
        print("\nStopping the assistant...")
    finally:
        recorder.cleanup()
        if os.path.exists("temp_recording.wav"):
            os.remove("temp_recording.wav")

if __name__ == "__main__":
    main()
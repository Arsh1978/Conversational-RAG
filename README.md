﻿# AI Voice Assistant with Conversational Retrieval

## Overview
This project implements an AI-powered voice assistant capable of:
- Processing queries with a conversational retrieval system built on LangChain.
- The assistant uses a pre-loaded PDF document as a knowledge base to answer questions.

## Features
- Conversational knowledge retrieval using LangChain with Chroma for vector storage.
- Voice synthesis to provide audio responses.
- Handles user interactions dynamically in real-time.

## Prerequisites
- Python 3.8+
- OpenAI API key (with Whisper and GPT-3.5 access).
- Required Python dependencies (listed in `requirements.txt`).

## Getting Started
Follow the steps below to set up and run the project.

### 1. Clone the Repository
```bash
git clone https://github.com/Arsh1978/Conversational-RAG.git
cd Conversational-RAG
```

### 2. Install Python Dependencies
Ensure you have Python 3.8+ installed. Then, install the required packages:
```bash
pip install -r requirements.txt
```

### 3. Environment Setup
Create a `.env` file in the project root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Place the Knowledge Base
Place the PDF document to be used as the knowledge base in the appropriate directory. Update the file path in the `AIVoiceAssistant.py` file:
```python
loader = PyPDFLoader(r"D:\zML\audio_conv_rag\rag\Harsh_Tyagi_1.4.pdf")
```
Replace this path with the actual location of your PDF file.

### 5. Run the Application
Run the assistant:
```bash
python app.py
```

### 6. Interaction Workflow
1. Speak into your microphone after running the application.
2. Your query will be processed.
3. The response is provided as both text on the terminal and audio playback.
4. Continue speaking or press `Ctrl+C` to exit the application.

## Project Structure
- **`AIVoiceAssistant.py`**: Core class implementing the conversational retrieval logic.
- **`app.py`**: Main script for real-time audio processing and interaction.
- **`requirements.txt`**: Python dependencies.


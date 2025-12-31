import gradio as gr
import numpy as np
import torch
from datasets import load_dataset

from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor, pipeline

# hard coding CPU since the space runs on a CPU-only environment
device = torch.device("cpu")

# load speech translation checkpoint
asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device)

# load translation pipeline
translation_pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-nl", device=device)

# load text-to-speech checkpoint and speaker embeddings
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")

model = SpeechT5ForTextToSpeech.from_pretrained("sanchit-gandhi/speecht5_tts_vox_nl").to(device)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", revision="ad29d262", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)


def english_transcript(audio):
    outputs = asr_pipe(audio, max_new_tokens=256)
    return outputs["text"]

def translate_to_nl(text):
    outputs = translation_pipe(text, max_new_tokens=256)
    return outputs[0]["translation_text"]

def synthesise(text):
    inputs = processor(text=text, return_tensors="pt")
    speech = model.generate_speech(inputs["input_ids"].to(device), speaker_embeddings.to(device), vocoder=vocoder)
    return speech.cpu()


def speech_to_speech_translation(audio):
    english_text = english_transcript(audio)
    translated_text = translate_to_nl(english_text)
    synthesised_speech = synthesise(translated_text)
    synthesised_speech = (synthesised_speech.numpy() * 32767).astype(np.int16)
    return 16000, synthesised_speech


title = "Cascaded STST"
description = """
Demo for cascaded speech-to-speech translation (STST), mapping from English to Dutch. Demo uses OpenAI's [Whisper Base](https://huggingface.co/openai/whisper-base) model for speech translation, and Microsoft's
[SpeechT5 TTS](https://huggingface.co/microsoft/speecht5_tts) model for text-to-speech:

![Cascaded STST](https://huggingface.co/datasets/huggingface-course/audio-course-images/resolve/main/s2st_cascaded.png "Diagram of cascaded speech to speech translation")
"""

demo = gr.Blocks()

# mic_translate = gr.Interface(
#     fn=speech_to_speech_translation,
#     inputs=gr.Audio(sources=["microphone"], type="filepath"),
#     outputs=gr.Audio(label="Generated Speech", type="numpy"),
#     title=title,
#     description=description,
# )

file_translate = gr.Interface(
    fn=speech_to_speech_translation,
    inputs=gr.Audio(sources=["upload"], type="filepath"),
    outputs=gr.Audio(label="Generated Speech", type="numpy"),
    examples=[["./example.wav"]],
    title=title,
    description=description,
)

with demo:
    gr.TabbedInterface([file_translate], ["Audio File"])

demo.launch()

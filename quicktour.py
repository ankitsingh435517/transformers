import torch
from datasets import load_dataset, Audio
from transformers import pipeline

# classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
# results = classifier(["This course is amazing", "I hope to learn more about the transformers library"])
# for result in results:
#     print(f"label: {result['label']}, score: {result['score']}")

speech_recognizer = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")

dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
dataset = dataset.cast_column("audio", Audio(sampling_rate=speech_recognizer.feature_extractor.sampling_rate))

result = speech_recognizer(dataset[:4]['audio'])
for d in result:
    print(d['text'])
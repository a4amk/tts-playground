from phonemizer import phonemize
text = "Hello world"
phonemes = phonemize(text, language='en-us', backend='espeak', strip=True)
print(f"Text: {text}")
print(f"Phonemes: {phonemes}")

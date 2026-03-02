# 🎛️ Standardized Synthesis Controls

The TTS Playground provides a set of standardized controls that allow for fine-tuned audio generation. Depending on the selected model, different controls will be available in the UI.

## Core Controls

### 🏃 Synthesis Speed
- **Parameter**: `speed`
- **Range**: `0.5` to `2.0` (Default: `1.0`)
- **Compatibility**: ✅ Streaming | ✅ Batch
- **Description**: Adjusts the overall pace of the speech. A value of `1.0` is the model's natural speed. Values < `1.0` slow down the speech, while > `1.0` speed it up.
- **Tip**: Most models maintain high quality between `0.8` and `1.2`. Speed is applied using native model scaling (Piper) or high-quality `librosa` time-stretching (Kokoro, ZipVoice, Genie).

### 🌡️ Sampling Temperature
- **Parameter**: `temp`
- **Range**: `0.1` to `2.0` (Default: `0.7`)
- **Compatibility**: ✅ Streaming | ✅ Batch
- **Description**: Controls the randomness and expressiveness of the model. 
    - **High Temp (1.0+)**: More creative, emotional, and varied, but can lead to "hallucinations" or unstable speech.
    - **Low Temp (< 0.5)**: More precise, consistent, and stable, but can sound robotic or monotone.
- **Recommended**: `0.7` is a good balance for most Moshi-based models (Chatterbox, Pocket-TTS).

### 🎲 Random Seed
- **Parameter**: `seed`
- **Range**: `0` to `999999` (Default: `0`)
- **Compatibility**: ✅ Streaming | ✅ Batch
- **Description**: Sets the random seed for deterministic generation. 
    - **0**: Random results every time.
    - **Fixed Value**: Reproducible speech with the exact same prosody and inflection.

## Advanced Model-Specific Controls

### 🎯 Top-K Filtering
- **Parameter**: `top_k`
- **Range**: `1` to `100` (Default: `50`)
- **Description**: Limits the sampling pool to the top K most likely tokens. Helps prevent the model from choosing very low-probability (and likely nonsensical) sounds.

### 📊 Repetition Penalty
- **Parameter**: `rep_pen`
- **Range**: `1.0` to `2.0` (Default: `1.1` - `1.2`)
- **Description**: Penalizes the model for repeating the same sound or word. Useful for preventing "loping" or stuttering in autoregressive models.

### 🎭 Exaggeration / Emotion
- **Parameter**: `exaggeration`
- **Range**: `0.0` to `2.0` (Default: `0.5`)
- **Description**: Specifically for **Moshi** and **Chatterbox** models. Controls the intensity of the emotional inflection and non-verbal cues (like sighs or breaths).

---

## Technical Integration
Engines declare their supported controls via the `get_standard_controls()` method in the `TTSPlugin` interface. This metadata drives the dynamic visibility and documentation (help text) in the Gradio UI.

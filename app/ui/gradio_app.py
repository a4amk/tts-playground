import gradio as gr
from ..engines.registry import models
from .js_snippets import INJECT_JS
from ..engines.clones import clones_manager
import os

def save_clone_action(model_id, name, audio_path, ref_text, base_voice):
    if not name or not audio_path:
        return "Please provide a name and an audio file."
    
    try:
        if model_id == "zipvoice":
            if not ref_text or not ref_text.strip():
                return "Error: ZipVoice requires a reference transcript."
            return clones_manager.save_zipvoice_clone(name, audio_path, ref_text)
        elif model_id == "pocket-tts":
            from ..engines.pocket_tts.model import pocket_tts_model
            from pocket_tts.models.tts_model import export_model_state
            
            # New high-quality cloning with optional transcript
            state = pocket_tts_model.create_clone_state(audio_path, ref_text)
            return clones_manager.save_pocket_tts_clone(name, state, export_model_state)
        elif model_id == "genie":
            if not ref_text or not ref_text.strip():
                return "Error: Genie requires a reference transcript."
            if not base_voice:
                return "Error: Genie requires a base voice selected to clone from."
            return clones_manager.save_genie_clone(name, audio_path, ref_text, base_voice)
        elif model_id == "chatterbox-turbo-onnx":
            return clones_manager.save_chatterbox_clone(name, audio_path)
        else:
            return f"Cloning not supported for {model_id} yet."
    except Exception as e:
        return f"Error saving clone: {str(e)}"

def update_cloning_ui(model_id):
    if model_id == "zipvoice":
        return gr.update(value="### Create a New Voice Clone\n*ZipVoice requires both a Reference Audio and exactly matching Reference Text.*\n\n**Recommended Ref. Length**: 5-10 seconds of clear, monotonic speech."), gr.update(visible=True, placeholder="Paste exactly what is spoken in the audio..."), gr.update(visible=True)
    elif model_id == "pocket-tts":
        return gr.update(value="### Create a New Voice Clone\n*Pocket-TTS only needs Reference Audio. Reference Text is optional but helps quality.*\n\n**Recommended Ref. Length**: 1-5 seconds. Short, punchy samples work great."), gr.update(visible=True, placeholder="(Optional) Text spoken in audio..."), gr.update(visible=True)
    elif model_id == "genie":
        return gr.update(value="### Create a New Voice Clone\n*Genie requires Reference Audio, Reference Text, AND you must select a Base Voice from the dropdown on the right.*\n\n**Recommended Ref. Length**: 10-15 seconds of expressive speech."), gr.update(visible=True, placeholder="Paste exactly what is spoken in the audio..."), gr.update(visible=True)
    elif model_id == "chatterbox-turbo-onnx":
        return gr.update(value="### Create a New Voice Clone\n*Chatterbox Turbo only requires Reference Audio.*\n\n**Recommended Ref. Length**: 5-10 seconds of natural, high-quality vocal audio."), gr.update(visible=False), gr.update(visible=True)
    elif model_id == "neutts":
        return gr.update(value="### Create a New Voice Clone\n*NeuTTS support is coming soon. Uploading a reference here will store it for the new engine.*\n\n**Recommended Ref. Length**: 10+ seconds for high-fidelity training."), gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(value="### Create a New Voice Clone\n*Model currently selected does NOT support Voice Cloning.*"), gr.update(visible=False), gr.update(visible=False)

def update_sliders_for_model(model_id):
    defaults = {
        "kokoro": {"temp": 0.2, "top_k": 50, "top_p": 0.95, "rep_pen": 1.0, "cfg": 1.0, "exag": 0.5},
        "kokoro-onnx": {"temp": 0.2, "top_k": 50, "top_p": 0.95, "rep_pen": 1.0, "cfg": 1.0, "exag": 0.5},
        "piper": {"temp": 0.66, "top_k": 50, "top_p": 0.8, "rep_pen": 1.0, "cfg": 1.0, "exag": 0.5},
        "pocket-tts": {"temp": 0.7, "top_k": 50, "top_p": 0.9, "rep_pen": 1.0, "cfg": 1.0, "exag": 0.5},
        "zipvoice": {"temp": 0.7, "top_k": 50, "top_p": 0.9, "rep_pen": 1.0, "cfg": 1.0, "exag": 0.5},
        "genie": {"temp": 0.7, "top_k": 50, "top_p": 0.9, "rep_pen": 1.0, "cfg": 1.0, "exag": 0.5},
        "chatterbox-turbo-onnx": {"temp": 0.8, "top_k": 50, "top_p": 1.0, "rep_pen": 1.2, "cfg": 0.5, "exag": 0.5},
        "neutts": {"temp": 0.5, "top_k": 50, "top_p": 1.0, "rep_pen": 1.0, "cfg": 1.0, "exag": 0.5},
    }
    cfg = defaults.get(model_id, defaults["kokoro"])
    return (
        gr.update(value=cfg["temp"]),
        gr.update(value=cfg["top_k"]),
        gr.update(value=cfg["top_p"]),
        gr.update(value=cfg["rep_pen"]),
        gr.update(value=cfg["cfg"]),
        gr.update(value=cfg["exag"])
    )

def tts_batch(text, model_id, voice, lang, speed, split_choice, custom_regex, temp, top_k, top_p, rep_pen, seed, cfg, exaggeration):
    import time
    if not text.strip():
        return None, "No text provided."
    engine = models.get(model_id)
    if not engine:
        return None, f"Model {model_id} not found"
        
    try:
        start_time = time.time()
        audio_payload = engine.generate_batch(
            text=text, voice=voice, speed=speed, lang=lang, split_choice=split_choice, custom_regex=custom_regex,
            temp=temp, top_k=top_k, top_p=top_p, rep_pen=rep_pen, seed=seed, cfg=cfg, exaggeration=exaggeration
        )
        if not audio_payload:
            return None, "No output generated."
            
        rate, final_audio = audio_payload
        gen_time = time.time() - start_time
        audio_dur = len(final_audio) / rate
        rtf = gen_time / audio_dur if audio_dur > 0 else 0
        
        status_msg = f"Batch Complete [{model_id}] | Audio Duration: {audio_dur:.2f}s | Gen Time: {gen_time:.2f}s | RTF: {rtf:.3f}"
        return (rate, final_audio), status_msg
    except Exception as e:
        return None, f"Error: {e}"

def update_dropdowns(model_id):
    engine = models.get(model_id)
    if engine:
        voices = engine.get_available_voices()
        langs = getattr(engine, 'get_available_languages', lambda: ["auto"])()
        
        voice_update = gr.update(choices=voices, value=voices[0] if voices else None)
        lang_update = gr.update(choices=langs, value=langs[0] if langs else "auto")
        return voice_update, lang_update
    return gr.update(choices=[], value=None), gr.update(choices=["auto"], value="auto")
CSS = """
body { font-family: 'Inter', sans-serif; }
.gradio-container { max-width: 950px !important; margin: auto; }
.header-row { align-items: center; justify-content: space-between; margin-bottom: 2rem;}
"""

def create_blocks():
    # In Gradio 6.0+, theme and css should ideally be passed to launch(), 
    # but for Blocks they are still accepted though deprecated. 
    # Let's keep them here but fix the UnboundLocalError.
    with gr.Blocks(title="TTS Multi-Model Control Center", css=CSS, theme=gr.themes.Base()) as blocks_app:
        blocks_app.load(fn=None, js=INJECT_JS)
        
        with gr.Row(elem_classes="header-row"):
            gr.Markdown("# 🎧 Universal TTS Generator\n*Extensible Low-Latency Streaming Frontend!*")
                
        with gr.Row():
            with gr.Column(scale=5):
                text_input = gr.Textbox(
                    label="Text to Synthesize", lines=12, 
                    value="The WebSocket approach is the final boss of low-latency streaming! By abstracting the models into a proper engine framework, we can now drop in additional engines like VITS or XTTS while maintaining flawless AudioContext buffer management!"
                )
                with gr.Accordion("Chunking Rules (Regex Patterns)", open=True):
                    split_choice_input = gr.Dropdown(
                        choices=[
                            "Both (Newlines & Sentences)",
                            "Sentences (Punctuation)",
                            "Paragraphs (Newlines)",
                            "Words (Spaces)",
                            "No Splitting (Single Pass)",
                            "Custom Regex"
                        ], 
                        value="Both (Newlines & Sentences)", 
                        label="Split Pattern Regex Selector"
                    )
                    custom_regex_input = gr.Textbox(label="Custom Regex Pattern", value=r'\n+', visible=False)
                    
                split_choice_input.change(lambda c: gr.update(visible=c == "Custom Regex"), inputs=split_choice_input, outputs=custom_regex_input)

                with gr.Accordion("Voice Cloning & Management", open=False):
                    clone_desc = gr.Markdown("### Create a New Voice Clone\n*Select a model capable of cloning to see instructions.*")
                    clone_name_input = gr.Textbox(label="Clone Name (e.g. MyVoice1)")
                    clone_audio_input = gr.Audio(label="Reference Audio (WAV)", type="filepath")
                    clone_text_input = gr.Textbox(label="Reference Text (ZipVoice & Genie ONLY)", lines=3, visible=True)
                    save_clone_btn = gr.Button("💾 Save Voice Clone", variant="secondary")
                    clone_status = gr.Textbox(label="Cloning Status", interactive=False)
                

                
            with gr.Column(scale=4):
                with gr.Group():
                    model_dropdown = gr.Dropdown(choices=list(models.keys()), value="kokoro", label="TTS Engine", interactive=True)
                    voice_dropdown = gr.Dropdown(choices=[], value=None, label="Voice Configuration", interactive=True)
                    lang_dropdown = gr.Dropdown(choices=["en", "auto"], value="en", label="Language Config", interactive=True)
                    speed_slider = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Relative Playback Speed")
                    
                with gr.Accordion("Advanced Synthesis Controls", open=False):
                    temp_slider = gr.Slider(minimum=0.0, maximum=2.0, value=0.7, step=0.1, label="Temperature (Randomness)")
                    top_k_slider = gr.Slider(minimum=0, maximum=100, value=50, step=1, label="Top K")
                    top_p_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top P")
                    rep_pen_slider = gr.Slider(minimum=1.0, maximum=2.0, value=1.0, step=0.05, label="Repetition Penalty")
                    cfg_slider = gr.Slider(minimum=0.0, maximum=5.0, value=0.5, step=0.1, label="CFG (Classifier-Free Guidance) Weight", info="Primarily used for Chat/Diffusion")
                    exag_slider = gr.Slider(minimum=0.0, maximum=2.0, value=0.5, step=0.1, label="Exaggeration / Emotion Control", info="0 = Flat, 0.5 = Neutral, 1.0+ = Extreme")
                    seed_input = gr.Number(value=0, label="Seed (0 for random)", precision=0)
                    
                # Show/Hide Ref Text based on Model (ZipVoice and Pocket-TTS both can use it now)
                model_dropdown.change(
                    fn=update_cloning_ui,
                    inputs=model_dropdown,
                    outputs=[clone_desc, clone_text_input, clone_audio_input]
                )
                
                model_dropdown.change(
                    fn=update_sliders_for_model,
                    inputs=model_dropdown,
                    outputs=[temp_slider, top_k_slider, top_p_slider, rep_pen_slider, cfg_slider, exag_slider]
                )
                    
                # Populate voices initially
                blocks_app.load(fn=update_dropdowns, inputs=model_dropdown, outputs=[voice_dropdown, lang_dropdown])
                # Change voices when model changes
                model_dropdown.change(fn=update_dropdowns, inputs=model_dropdown, outputs=[voice_dropdown, lang_dropdown])

                with gr.Tabs():
                    with gr.Tab("JavaScript WebSocket Stream ⚡"):
                        gr.Markdown("Real-time incremental text-to-speech! As you type (or as an LLM appends), text is buffered and synthesized sentence-by-sentence. Click 'Finish/Flush' to synthesize the last incomplete part.")
                        with gr.Row():
                            ws_stream_btn = gr.Button("▶️ Start Live Stream", variant="primary")
                            ws_flush_btn = gr.Button("💨 Finish/Flush", variant="secondary")
                            ws_stop_btn = gr.Button("⏹️ Stop & Clear", variant="stop")
                        gr.HTML("<div id='ws-status-logs' style='background:#111827; color:#10b981; font-family:monospace; padding:10px; border-radius:5px; height:150px; overflow-y:auto; font-size:12px;'>Frontend Interface initialized... Waiting for user.<br/></div>")
                        
                    with gr.Tab("Batch Generator 📦"):
                        batch_btn = gr.Button("Generate Full Download")
                        batch_output = gr.Audio(label="Merged Output Response", interactive=False)
                status_text = gr.Textbox(label="Batch Mode Status", interactive=False)
                
        # Incremental feeding!
        text_input.change(
            fn=None,
            inputs=[text_input],
            js="(t) => { if(window.onTextInputChange) window.onTextInputChange(t); return []; }"
        )

        ws_stream_btn.click(
            fn=None, 
            inputs=[text_input, model_dropdown, voice_dropdown, lang_dropdown, speed_slider, split_choice_input, custom_regex_input, temp_slider, top_k_slider, top_p_slider, rep_pen_slider, seed_input, cfg_slider, exag_slider], 
            js="(text, model, voice, lang, speed, split_choice, custom_regex, temp, top_k, top_p, rep_pen, seed, cfg, exaggeration) => { window.startWebSocketStream(text, model, voice, lang, speed, split_choice, custom_regex, temp, top_k, top_p, rep_pen, seed, cfg, exaggeration); return []; }"
        )
        ws_flush_btn.click(fn=None, js="() => { window.flushWsStream(); return []; }")
        ws_stop_btn.click(fn=None, js="() => { window.stopWsStream(); return []; }")
        batch_btn.click(
            fn=tts_batch, 
            inputs=[text_input, model_dropdown, voice_dropdown, lang_dropdown, speed_slider, split_choice_input, custom_regex_input, temp_slider, top_k_slider, top_p_slider, rep_pen_slider, seed_input, cfg_slider, exag_slider], 
            outputs=[batch_output, status_text]
        )

        save_clone_btn.click(
            fn=save_clone_action,
            inputs=[model_dropdown, clone_name_input, clone_audio_input, clone_text_input, voice_dropdown],
            outputs=[clone_status]
        ).then(
            fn=update_dropdowns,
            inputs=model_dropdown,
            outputs=[voice_dropdown, lang_dropdown]
        )
        
    return blocks_app

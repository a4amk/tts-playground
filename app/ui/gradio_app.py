import gradio as gr
import os
import time
import logging
from ..engines.manager import plugin_manager
from ..config import DEFAULT_TTS_ENGINE
from .js_snippets import INJECT_JS

logger = logging.getLogger(__name__)

def save_clone_action(model_id, name, audio_path, ref_text, ref_lang):
    if not name or not audio_path:
        return "Please provide a name and an audio file."
    
    try:
        engine = plugin_manager.get_plugin(model_id)
        if not engine:
            return f"Error: Model {model_id} not found."
            
        engine.save_clone(name, audio_path, ref_text, ref_lang=ref_lang)
        return f"Successfully saved clone '{name}' (Language: {ref_lang}) for engine '{model_id}'."
    except NotImplementedError:
        return f"Cloning not supported for {model_id}."
    except Exception as e:
        logger.exception("Error saving clone")
        return f"Error saving clone: {str(e)}"

def update_cloning_ui(model_id):
    engine = plugin_manager.get_plugin(model_id)
    if not engine:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value="Model not found.")

    config = engine.get_cloning_config()
    if not config.get("requires_cloning", False):
        return (
            gr.update(value="### Voice Cloning\n*This model does not support zero-shot cloning.*"),
            gr.update(visible=False), # Audio input
            gr.update(visible=False), # Text input
            gr.update(visible=False), # Lang input
            gr.update(visible=False)  # Save Button
        )

    instruction = config.get("instruction", "Upload audio to clone.")
    req_transcript = config.get("requires_transcript", False)
    
    return (
        gr.update(value=f"### Create a New Voice Clone\n{instruction}"),
        gr.update(visible=True), # Audio input
        gr.update(visible=req_transcript, placeholder="Enter transcript here..." if req_transcript else ""),
        gr.update(visible=True), # Lang dropdown
        gr.update(visible=True) # Save button
    )


def tts_batch(text, model_id, voice, lang, variant, speed, split_choice, custom_regex, temp, top_k, top_p, rep_pen, seed, cfg, exaggeration, *extras):
    if not text.strip():
        return None, "No text provided."
        
    engine = plugin_manager.get_plugin(model_id)
    if not engine:
        return None, f"Model {model_id} not found"
        
    # Map extras back to kwargs
    extra_definitions = engine.get_extra_controls()
    extra_kwargs = {}
    for i, ctrl in enumerate(extra_definitions):
        if i < len(extras):
            # Defensive check: avoid overriding core arguments
            if ctrl["id"] not in ["text", "model", "voice", "lang", "variant", "speed", "split_choice", "custom_regex", "temp", "top_k", "top_p", "rep_pen", "seed", "cfg", "exaggeration"]:
                extra_kwargs[ctrl["id"]] = extras[i]

    try:
        start_time = time.time()
        # Ensure model is loaded
        engine.load(variant=variant)
        
        audio_payload = engine.generate_batch(
            text=text, voice=voice, speed=speed, lang=lang, variant=variant, split_choice=split_choice, custom_regex=custom_regex,
            temp=temp, top_k=top_k, top_p=top_p, rep_pen=rep_pen, seed=seed, cfg=cfg, exaggeration=exaggeration,
            **extra_kwargs
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
        logger.exception("Batch generation failed")
        return None, f"Error: {e}"

def update_dropdowns(model_id):
    engine = plugin_manager.get_plugin(model_id)
    if engine:
        voices = engine.get_available_voices()
        langs = engine.get_available_languages()
        
        voice_update = gr.update(choices=voices, value=voices[0] if voices else None)
        lang_update = gr.update(choices=langs, value=langs[0] if langs else "auto")
        
        # Collect extra controls
        extra = engine.get_extra_controls()
        # Ensure we return valid updates for up to 5 controls
        extra_visible = []
        for i in range(5):
            if i < len(extra):
                ctrl = extra[i]
                choices = ctrl.get("choices")
                if not choices and ctrl.get("type") == "checkbox":
                    choices = [True, False]
                extra_visible.append(gr.update(
                    visible=True, label=ctrl["label"], info=ctrl.get("info"), 
                    value=ctrl["default"], choices=choices
                ))
            else:
                extra_visible.append(gr.update(visible=False))
        
        # Header visibility
        header_update = gr.update(visible=len(extra) > 0)
        
        # Variants
        variants = engine.get_variants()
        variant_choices = [(v["label"], v["id"]) for v in variants]
        default_variant_id = next((v["id"] for v in variants if v.get("default")), variant_choices[0][1] if variant_choices else None)
        variant_update = gr.update(choices=variant_choices, value=default_variant_id, visible=True)

        # Standard Controls
        std_meta = {m["id"]: m for m in engine.get_standard_controls()}
        std_ids = ["speed", "temp", "top_k", "top_p", "rep_pen", "cfg", "exaggeration", "seed"]
        std_updates = []
        for sid in std_ids:
            if sid in std_meta:
                m = std_meta[sid]
                std_updates.append(gr.update(
                    visible=True, label=m["label"], info=m.get("info"), 
                    value=m.get("default"), minimum=m.get("min"), 
                    maximum=m.get("max"), step=m.get("step")
                ))
            else:
                std_updates.append(gr.update(visible=False))

        return [voice_update, lang_update, variant_update, header_update] + std_updates + extra_visible
    
    # Defaults
    std_defaults = [gr.update(visible=False)] * 8
    return [gr.update(choices=[], value=None), gr.update(choices=["auto"], value="auto"), gr.update(visible=True, choices=[], value=None), gr.update(visible=False)] + std_defaults + [gr.update(visible=False)] * 5

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono&display=swap');

:root {
    --primary-color: #6366f1;
    --bg-color: #0f172a;
    --card-bg: #1e293b;
}

body { 
    font-family: 'Inter', sans-serif; 
    background-color: var(--bg-color) !important;
}

.gradio-container { 
    max-width: 1100px !important; 
    margin: auto;
    padding: 2rem !important;
}

.header-row { 
    align-items: center; 
    justify-content: space-between; 
    margin-bottom: 3rem;
    border-bottom: 1px solid #334155;
    padding-bottom: 1rem;
}

.header-row h1 {
    font-weight: 800;
    letter-spacing: -0.025em;
    background: linear-gradient(to right, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

#extra-controls-container {
    background: #1e293b;
    border-radius: 12px;
    padding: 1.5rem;
    border: 1px solid #334155;
    margin-top: 1rem;
}

#ws-status-logs {
    border: 1px solid #334155 !important;
    background: #020617 !important;
    border-radius: 8px !important;
    box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.06);
}

.gr-button-primary {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
    border: none !important;
    box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.39) !important;
}

.gr-button-primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px 0 rgba(99, 102, 241, 0.23) !important;
}

.gr-form {
    border-radius: 12px !important;
    border: 1px solid #334155 !important;
}

/* Clutter-free tweaks */
.gr-box { border-radius: 12px !important; }
.gr-padded { padding: 1.5rem !important; }

"""

def create_blocks():
    plugin_manager.discover_plugins()
    available_models = plugin_manager.get_all_ids()
    default_model = DEFAULT_TTS_ENGINE if DEFAULT_TTS_ENGINE in available_models else (available_models[0] if available_models else None)

    with gr.Blocks(title="TTS Multi-Model Control Center", css=CSS, theme=gr.themes.Base()) as blocks_app:
        blocks_app.load(fn=None, js=INJECT_JS)
        
        with gr.Row(elem_classes="header-row"):
            gr.Markdown("# 🎧 Universal TTS Generator\n*Dynamic Plug-and-Play Architecture*")
                
        with gr.Row():
            with gr.Column(scale=5):
                text_input = gr.Textbox(
                    label="Text to Synthesize", lines=12, 
                    value="The new plugin architecture is live! Engines are now self-contained and discovered automatically at runtime. No more hardcoded registries!"
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
                    clone_audio_input = gr.Audio(label="Reference Audio", type="filepath", visible=False)
                    clone_text_input = gr.Textbox(label="Reference Text / Transcript", lines=3, visible=False)
                    clone_lang_input = gr.Dropdown(choices=["English", "Chinese", "Japanese", "Korean", "Spanish", "French", "German"], value="English", label="Reference Language (The language of the file you just uploaded)", visible=False)
                    save_clone_btn = gr.Button("💾 Save Voice Clone", variant="secondary", visible=False)
                    clone_status = gr.Textbox(label="Cloning Status", interactive=False)
                
            with gr.Column(scale=4):
                with gr.Group():
                    model_dropdown = gr.Dropdown(choices=available_models, value=default_model, label="TTS Engine", interactive=True)
                    variant_dropdown = gr.Dropdown(choices=[], value=None, label="Model Variant (Runtime / Quantization)", interactive=True, visible=True)
                    voice_dropdown = gr.Dropdown(choices=[], value=None, label="Voice Configuration", interactive=True)
                    lang_dropdown = gr.Dropdown(choices=["auto"], value="auto", label="Language Config", interactive=True)
                    speed_slider = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Relative Playback Speed")
                    
                    with gr.Group(elem_id="extra-controls-container"):
                        extra_header = gr.Markdown("### Engine Specific Settings", visible=False)
                        extra_ctrls = []
                        for i in range(5):
                            extra_ctrls.append(gr.Dropdown(visible=False, label=f"Extra {i}"))

                with gr.Accordion("Advanced Synthesis Controls", open=False):
                    temp_slider = gr.Slider(minimum=0.0, maximum=2.0, value=0.7, step=0.1, label="Temperature (Randomness)")
                    top_k_slider = gr.Slider(minimum=0, maximum=100, value=50, step=1, label="Top K")
                    top_p_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top P")
                    rep_pen_slider = gr.Slider(minimum=1.0, maximum=2.0, value=1.0, step=0.05, label="Repetition Penalty")
                    cfg_slider = gr.Slider(minimum=0.0, maximum=5.0, value=0.5, step=0.1, label="CFG (Classifier-Free Guidance) Weight")
                    exag_slider = gr.Slider(minimum=0.0, maximum=2.0, value=0.5, step=0.1, label="Exaggeration / Emotion Control")
                    seed_input = gr.Number(value=0, label="Seed (0 for random)", precision=0)
                    
                model_dropdown.change(
                    fn=update_cloning_ui,
                    inputs=model_dropdown,
                    outputs=[clone_desc, clone_audio_input, clone_text_input, clone_lang_input, save_clone_btn]
                )
                
                blocks_app.load(
                    fn=update_dropdowns, 
                    inputs=model_dropdown, 
                    outputs=[voice_dropdown, lang_dropdown, variant_dropdown, extra_header, speed_slider, temp_slider, top_k_slider, top_p_slider, rep_pen_slider, cfg_slider, exag_slider, seed_input] + extra_ctrls
                )
                model_dropdown.change(
                    fn=update_dropdowns, 
                    inputs=model_dropdown, 
                    outputs=[voice_dropdown, lang_dropdown, variant_dropdown, extra_header, speed_slider, temp_slider, top_k_slider, top_p_slider, rep_pen_slider, cfg_slider, exag_slider, seed_input] + extra_ctrls
                )

                with gr.Tabs():
                    with gr.Tab("JavaScript WebSocket Stream ⚡"):
                        gr.Markdown("Real-time incremental text-to-speech!")
                        with gr.Row():
                            ws_stream_btn = gr.Button("▶️ Start Live Stream", variant="primary")
                            ws_flush_btn = gr.Button("💨 Finish/Flush", variant="secondary")
                            ws_stop_btn = gr.Button("⏹️ Stop & Clear", variant="stop")
                        gr.HTML("<div id='ws-status-logs' style='background:#111827; color:#10b981; font-family:monospace; padding:10px; border-radius:5px; height:150px; overflow-y:auto; font-size:12px;'>Frontend Interface initialized... Waiting for user.<br/></div>")
                        
                    with gr.Tab("Batch Generator 📦"):
                        batch_btn = gr.Button("Generate Full Download")
                        batch_output = gr.Audio(label="Merged Output Response", interactive=False)
                status_text = gr.Textbox(label="Batch Mode Status", interactive=False)
                
        text_input.change(
            fn=None,
            inputs=[text_input],
            js="(t) => { if(window.onTextInputChange) window.onTextInputChange(t); return []; }"
        )

        ws_stream_btn.click(
            fn=None, 
            inputs=[text_input, model_dropdown, voice_dropdown, lang_dropdown, variant_dropdown, speed_slider, split_choice_input, custom_regex_input, temp_slider, top_k_slider, top_p_slider, rep_pen_slider, seed_input, cfg_slider, exag_slider] + extra_ctrls, 
            js="(text, model, voice, lang, variant, speed, split_choice, custom_regex, temp, top_k, top_p, rep_pen, seed, cfg, exaggeration, ...extras) => { window.startWebSocketStream(text, model, voice, lang, variant, speed, split_choice, custom_regex, temp, top_k, top_p, rep_pen, seed, cfg, exaggeration, extras); return []; }"
        )
        ws_flush_btn.click(fn=None, js="() => { window.flushWsStream(); return []; }")
        ws_stop_btn.click(fn=None, js="() => { window.stopWsStream(); return []; }")
        batch_btn.click(
            fn=tts_batch, 
            inputs=[text_input, model_dropdown, voice_dropdown, lang_dropdown, variant_dropdown, speed_slider, split_choice_input, custom_regex_input, temp_slider, top_k_slider, top_p_slider, rep_pen_slider, seed_input, cfg_slider, exag_slider] + extra_ctrls, 
            outputs=[batch_output, status_text]
        )

        save_clone_btn.click(
            fn=save_clone_action,
            inputs=[model_dropdown, clone_name_input, clone_audio_input, clone_text_input, clone_lang_input],
            outputs=[clone_status]
        ).then(
            fn=update_dropdowns,
            inputs=model_dropdown,
            outputs=[voice_dropdown, lang_dropdown, variant_dropdown, extra_header] + extra_ctrls
        )
        
    return blocks_app

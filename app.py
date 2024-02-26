import gradio as gr
import os, torch, io
os.system('python -m unidic download')
from melo.api import TTS
speed = 1.0
import tempfile
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TTS(language='EN', device=device)
speaker_ids = model.hps.data.spk2id
def synthesize(speaker, text, speed=1.0, progress=gr.Progress()):
    bio = io.BytesIO()
    model.tts_to_file(text, speaker_ids[speaker], bio, speed=speed, pbar=progress.tqdm, format='wav')
    return bio.getvalue()
with gr.Blocks() as demo:
    gr.Markdown('# MeloTTS\n\nAn unofficial demo of [MeloTTS](https://github.com/myshell-ai/MeloTTS) from MyShell AI. MeloTTS is a permissively licensed (MIT) SOTA multi-speaker TTS model.\n\nI am not affiliated with MyShell AI in any way.\n\nThis demo currently only supports English, but the model itself supports other languages.')
    with gr.Group():
        speaker = gr.Dropdown(speaker_ids.keys(), interactive=True, value='EN-Default', label='Speaker')
        speed = gr.Slider(label='Speed', minimum=0.1, maximum=10.0, value=1.0, interactive=True, step=0.1)
        text = gr.Textbox(label="Text to speak", value='The field of text to speech has seen rapid development recently')
    btn = gr.Button('Synthesize', variant='primary')
    aud = gr.Audio(interactive=False)
    btn.click(synthesize, inputs=[speaker, text, speed], outputs=[aud])
demo.queue(api_open=False, default_concurrency_limit=10).launch(show_api=False)

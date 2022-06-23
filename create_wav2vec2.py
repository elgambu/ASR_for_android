import torch
from torch import Tensor
from torch.utils.mobile_optimizer import optimize_for_mobile
import torchaudio
from torchaudio.models.wav2vec2.utils.import_huggingface import import_huggingface_model
from transformers import Wav2Vec2ForCTC

# Wav2vec2 model emits sequences of probability (logits) distributions over the characters
# The following class adds steps to decode the transcript (best path)
class SpeechRecognizer(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.labels = ["", "<s>", "</s>", "⁇", " ", "'", "-", "a", "b", "c", "d", "e", "f", "g",
         "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y",
          "z", "á", "é", "í", "ñ", "ó", "ö", "ú", "ü"]
    def forward(self, waveforms: Tensor) -> str:
        """Given a single channel speech data, return transcription.

        Args:
            waveforms (Tensor): Speech tensor. Shape `[1, num_frames]`.

        Returns:
            str: The resulting transcript
        """
        logits, _ = self.model(waveforms)  # [batch, num_seq, num_label]
        best_path = torch.argmax(logits[0], dim=-1)  # [num_seq,]
        prev = ''
        hypothesis = ''
        for i in best_path:
            #print (i)
            char = self.labels[i]
            if char == prev:
                continue
            if char == '<s>':
                prev = ''
                continue
            hypothesis += char
            prev = char
        return hypothesis.replace('|', ' ')


# Load Wav2Vec2 pretrained model from Hugging Face Hub
model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-spanish")
# Convert the model to torchaudio format, which supports TorchScript.
model = import_huggingface_model(model)

# Remove weight normalization which is not supported by quantization.
model.encoder.transformer.pos_conv_embed.__prepare_scriptable__()
model = model.eval()
# Attach decoder
model = SpeechRecognizer(model)

# Apply quantization / script / optimize for motbile
quantized_model = torch.quantization.quantize_dynamic(
    model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)

scripted_model = torch.jit.script(quantized_model)
optimized_model = optimize_for_mobile(scripted_model)

# Sanity check
waveform , _ = torchaudio.load('common_voice_es_18309702.wav')
print('Result:', optimized_model(waveform))


optimized_model._save_for_lite_interpreter("wav2vec2ES.ptl")
print("check")


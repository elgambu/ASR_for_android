import torch
import torchaudio
from datasets import load_dataset, load_metric
import string

def log_results(result):

    # load metrics
    wer = load_metric("wer")
    cer = load_metric("cer")

    # compute metrics
    wer_result = wer.compute(references=result[1], predictions=result[0])
    cer_result = cer.compute(references=result[1], predictions=result[0])

    # print results
    result_str = (
        f"WER: {wer_result}\n"
        f"CER: {cer_result}"
    )
    print(result_str)


def predict(dataset, result):
    for input in dataset:
        waveform , sample_rate = torchaudio.load(input["path"])
        transform = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = transform(waveform)
        prediction = model(waveform)
        result[0].append(prediction)
        result[1].append(input["sentence"].translate(str.maketrans('', '', string.punctuation)).replace('¿','').replace('¡','').lower())
        print("Prediction:", prediction)
        print("Sentence:", input["sentence"].translate(str.maketrans('', '', string.punctuation)).replace('¿','').replace('¡','').lower())

    return result

if __name__=="__main__":

    # load optimized model
    model = torch.jit.load('wav2vec2.ptl')
    model.eval()
    model.cuda()

    # load database
    dataset = load_dataset("common_voice", "es", split="test")

    # predictions and metrics
    result = [[],[]]
    result = predict(dataset, result)
    log_results(result)
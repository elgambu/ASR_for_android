import torch
import torchaudio
from datasets import load_dataset, load_metric
import string


def log_results(result):

    # load metric
    wer = load_metric("wer")
    cer = load_metric("cer")
    print("EMPIEZA A COMPUTAR")
    # compute metrics
    wer_result = wer.compute(references=result[1], predictions=result[0])
    cer_result = cer.compute(references=result[1], predictions=result[0])

    # print & log results
    result_str = (
        f"WER: {wer_result}\n"
        f"CER: {cer_result}"
    )
    print(result_str)


def predict(dataset, result):
    i = 0
    print("Buscando hombres")
    for input in dataset:
        if input["gender"]== "male":
            waveform , sample_rate = torchaudio.load(input["path"])
            transform = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = transform(waveform)
            prediction = model(waveform)
            result[0].append(prediction)
            result[1].append(input["sentence"].translate(str.maketrans('', '', string.punctuation)).replace('¿','').replace('¡','').lower())
            print("Prediction:", prediction)
            print("Sentence:", input["sentence"].translate(str.maketrans('', '', string.punctuation)).replace('¿','').replace('¡','').lower())
            i = i + 1
    print (f"Hay {i} muestras masculinas")
    return result

if __name__=="__main__":

    model = torch.jit.load('wav2vec2.ptl')
    model.eval()
    model.cuda()

    dataset = load_dataset("common_voice", "es", split="test")
    #dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

    #print(dataset[0]["path"])
    #print(dataset[0]["sentence"])
    '''
    waveform , sample_rate = torchaudio.load(dataset[0]["path"])
    transform = torchaudio.transforms.Resample(sample_rate, 16000)
    waveform = transform(waveform)
    prediction = model(waveform)
    print("Prediction: ", prediction)
    '''
    result = [[],[]]

    result = predict(dataset, result)
    '''
    for i in range(11):
        waveform , sample_rate = torchaudio.load(dataset[i]["path"])
        transform = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = transform(waveform)
        prediction = model(waveform)
        result[0].append(prediction)
        result[1].append(dataset[i]["sentence"].lower())
        print("Prediction:", prediction)
        print("Sentence:", dataset[i]["sentence"].lower())

    '''
    log_results(result)

    '''
    waveform , _ = torchaudio.load('common_voice_es_18309702_16.mp3')
    prediction = model(waveform)
    log_results(prediction, target)
    '''
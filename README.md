# Speech Recognition on Android with Wav2Vec2

## Introduction

Facebook AI's [wav2vec 2.0](https://github.com/pytorch/fairseq/tree/master/examples/wav2vec) is one of the leading models in speech recognition. It is also available in the [Huggingface Transformers](https://github.com/huggingface/transformers) library.

This a project adapted from https://github.com/pytorch/android-demo-app/tree/master/SpeechRecognition. We adapt the PyTorch demo to optimize https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-spanish model. Then we use the android app to evaluate the performance. This repository include three new scripts to evaluate WER and CER of the optimized .ptl model.

Follow https://github.com/pytorch/android-demo-app/tree/master/SpeechRecognition steps to execute the code.

## Optimized model metrics

*Result (WER)*:

| "Female" | "Male" | "Total" |
|---|---|---|
| 11.88 | 11.58 | 11.90 |

*Result (CER)*:

| "Female" | "Male" | "Total" |
|---|---|---|
| 4.11 | 3.6 | 3.78 |

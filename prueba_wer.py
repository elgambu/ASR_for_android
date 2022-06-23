from operator import length_hint
from datasets import load_metric
import string

wer = load_metric("wer")
predictions = ["hello world", "good night moon"]
references = ["hello,?\" world", "good, night! moon."]
print(references)

length = len(references)
for i in range(length):
    references[i] = references[i].translate(str.maketrans('', '', string.punctuation))

print(references)
wer_score = wer.compute(predictions=predictions, references=references)
print(wer_score)
1.0
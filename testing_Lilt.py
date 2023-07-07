from datasets import load_dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("nielsr/lilt-roberta-en-base-finetuned-funsd")

model = AutoModelForTokenClassification.from_pretrained("nielsr/lilt-roberta-en-base-finetuned-funsd")

dataset = load_dataset("nielsr/funsd-layoutlmv3")

label_list = dataset["train"].features["ner_tags"].feature.names
id2label = {id:label for id, label in enumerate(label_list)}
print(id2label)
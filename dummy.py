from datasets import load_dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("SCUT-DLVCLab/lilt-roberta-en-base")
model = AutoModelForTokenClassification.from_pretrained("SCUT-DLVCLab/lilt-roberta-en-base")

dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
example = dataset[0]
words = example["tokens"]
boxes = example["bboxes"]

encoding = tokenizer(words, boxes=boxes, return_tensors="pt")

outputs = model(**encoding)
predicted_class_indices = outputs.logits.argmax(-1)
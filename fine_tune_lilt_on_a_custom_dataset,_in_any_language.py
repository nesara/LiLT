# # -*- coding: utf-8 -*-
# """Fine_tune_LiLT_on_a_custom_dataset,_in_any_language.ipynb

# Automatically generated by Colaboratory.

# Original file is located at
#     https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LiLT/Fine_tune_LiLT_on_a_custom_dataset%2C_in_any_language.ipynb

# ## Set-up environment
# """
# """## Load dataset"""

from datasets import load_dataset

dataset = load_dataset("nielsr/funsd-iob-original")

label_list = dataset["train"].features["ner_tags"].feature.names
id2label = {id:label for id, label in enumerate(label_list)}
print(id2label)

"""## Create PyTorch Dataset"""

dataset["train"].features

import torch
from PIL import Image
from torch.utils.data import Dataset


def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]

class CustomDataset(Dataset):
  def __init__(self, dataset, tokenizer):
    self.dataset = dataset
    self.tokenizer = tokenizer

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    # get item
    example = self.dataset[idx]
    image = example["image"]
    words = example["words"]
    boxes = example["original_bboxes"]
    ner_tags = example["ner_tags"]

    # prepare for the model
    width, height = image.size

    bbox = []
    labels = []
    for word, box, label in zip(words, boxes, ner_tags): #COMMENT: words= words, label= ner_tags
        box = normalize_bbox(box, width, height)
        n_word_tokens = len(tokenizer.tokenize(word))
        bbox.extend([box] * n_word_tokens)
        labels.extend([label] + ([-100] * (n_word_tokens - 1)))

    cls_box = sep_box = [0, 0, 0, 0]
    bbox = [cls_box] + bbox + [sep_box]
    labels = [-100] + labels + [-100]

    encoding = self.tokenizer(" ".join(words), truncation=True, max_length=512)
    sequence_length = len(encoding.input_ids)
    # truncate boxes and labels based on length of input ids
    labels = labels[:sequence_length]
    bbox = bbox[:sequence_length]

    encoding["bbox"] = bbox
    encoding["labels"] = labels

    return encoding

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("nielsr/lilt-xlm-roberta-base")

train_dataset = CustomDataset(dataset["train"], tokenizer)
eval_dataset = CustomDataset(dataset["test"], tokenizer)

example = train_dataset[0]

tokenizer.decode(example["input_ids"])

for k,v in example.items():
  print(k,len(v))

for word, box, label in zip(dataset["train"][0]["words"], dataset["train"][0]["original_bboxes"], dataset["train"][0]["ner_tags"]):
  print(word, box, id2label[label])

len(example["input_ids"])

for id, box, label in zip(example["input_ids"], example["bbox"], example["labels"]):
  if label != -100: #COMMENT: -100 indicates the non important entities/words. hence markint them with -100
    print(tokenizer.decode([id]), box, id2label[label])
  else:
    print(tokenizer.decode([id]), box, -100)

"""## Define PyTorch DataLoader"""

from torch.utils.data import DataLoader


def collate_fn(features):
  boxes = [feature["bbox"] for feature in features]
  labels = [feature["labels"] for feature in features]
  # use tokenizer to pad input_ids
  batch = tokenizer.pad(features, padding="max_length", max_length=512)

  sequence_length = torch.tensor(batch["input_ids"]).shape[1]
  batch["labels"] = [labels_example + [-100] * (sequence_length - len(labels_example)) for labels_example in labels]
  batch["bbox"] = [boxes_example + [[0, 0, 0, 0]] * (sequence_length - len(boxes_example)) for boxes_example in boxes]

  # convert to PyTorch
  # batch = {k: torch.tensor(v, dtype=torch.int64) if isinstance(v[0], list) else v for k, v in batch.items()}
  batch = {k: torch.tensor(v) for k, v in batch.items()}

  return batch

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
eval_dataloader = DataLoader(eval_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

batch = next(iter(train_dataloader))

for k,v in batch.items():
  print(k,v.shape)

tokenizer.decode(batch["input_ids"][0])

for id, box, label in zip(batch["input_ids"][0], batch["bbox"][0], batch["labels"][0]):
  if label.item() != -100:
    print(tokenizer.decode([id]), box, id2label[label.item()])
  else:
    print(tokenizer.decode([id]), box, label.item())

"""## Define model"""

from transformers import LiltForTokenClassification

model = LiltForTokenClassification.from_pretrained("nielsr/lilt-xlm-roberta-base", id2label=id2label)

"""## Train the model in native PyTorch

Uncomment the code below if you want to train the model just in native PyTorch.
"""

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)

# optimizer = torch.optim.AdamW(model.parameters(), lr=5-5)

# model.train()
# for epoch in range(2):
#   for batch in train_dataloader:
#       # zero the parameter gradients
#       optimizer.zero_grad()

#       inputs = {k:v.to(device) for k,v in batch.items()}

#       outputs = model(**inputs)

#       loss = outputs.loss
#       loss.backward()

#       optimizer.step()

"""## Train the model using 🤗 Trainer

We first define a compute_metrics function as well as TrainingArguments.
"""

import evaluate

metric = evaluate.load("seqeval")

import numpy as np
from seqeval.metrics import classification_report

return_entity_level_metrics = False

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    if return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(output_dir="test",
                                  num_train_epochs=30,
                                  learning_rate=5e-5,
                                  evaluation_strategy="steps",
                                  eval_steps=100,
                                  load_best_model_at_end=True,
                                  metric_for_best_model="f1")

"""Next we define a custom Trainer which uses the DataLoaders we created above."""

from transformers.data.data_collator import default_data_collator


class CustomTrainer(Trainer):
  def get_train_dataloader(self):
    return train_dataloader

  def get_eval_dataloader(self, eval_dataset = None):
    return eval_dataloader

# Initialize our Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()



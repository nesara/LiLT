from datasets import load_dataset
#TODO:
#COMMENT
#FIXME:
#BUG
#COMMENT : can also use the other dataset "nielsr/FUNSD_layoutlmv2"
dataset = load_dataset("nielsr/funsd-iob-original") 

#%%
label_list = dataset["train"].features["ner_tags"].feature.names
id2label = {id:label for id, label in enumerate(label_list)}
print(id2label)
dataset["train"].features


# %%

from torch.utils.data import Dataset
from PIL import Image
import torch

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
    for word, box, label in zip(words, boxes, ner_tags):
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
#FIXME: check with the lilt form loaded in the tokenzer  below
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
  if label != -100:
    print(tokenizer.decode([id]), box, id2label[label])
  else:
    print(tokenizer.decode([id]), box, -100)
    
    
#COMMENT: Defining pytorch Data Loader 
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
  
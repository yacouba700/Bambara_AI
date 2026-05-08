"" charger dataset ""
from datasets import Dataset
import json

with open("dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)

"" CHARGER NLLB ""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

"" TOKENIZATION  ""

def preprocess(example):

    input_text = example["fr"]
    target_text = example["bm"]

    model_inputs = tokenizer(
        input_text,
        max_length=128,
        truncation=True
    )

    labels = tokenizer(
        target_text,
        max_length=128,
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

tokenized_dataset = dataset.map(preprocess)

""  TRAINING  ""

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./bambara-model",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_steps=100,
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()

"" SAUVEGARDE DU MODÈLE ""

model.save_pretrained("./bambara-model")
tokenizer.save_pretrained("./bambara-model")

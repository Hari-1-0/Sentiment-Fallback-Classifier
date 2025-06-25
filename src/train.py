from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import torch

dataset = load_dataset('imdb')

model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(10000))  # Smaller subset for faster training
test_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(2000))

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"]
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="./models",
    learning_rate=2e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    save_strategy="epoch",
    save_steps=500,
    logging_dir="./logs",
    logging_steps=100
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)

trainer.train()

model.save_pretrained("./models/lora_finetuned")
tokenizer.save_pretrained("./models/lora_finetuned")

print("Training Complete and Model Saved!")

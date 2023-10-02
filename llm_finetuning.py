from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
import transformers
transformers.set_seed(42)

import wandb

wandb.login(anonymous="allow")
model_checkpoint = "roneneldan/TinyStories-33M"

ds = load_dataset("MohamedRashad/characters_backstories")

ds["train"][400]

ds = ds["train"].train_test_split(test_size=0.2, seed=42)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)

tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(example):
    merged = example["text"] + " " + example["target"]
    batch = tokenizer(merged, padding="max_length", truncation=True, max_length=128)
    batch["labels"] = batch["input_ids"].copy()
    return batch

tokenized_datasets = ds.map(tokenize_function, remove_columns=["text", "target"])

print(tokenizer.decode(tokenized_datasets["train"][900]["input_ids"]))

model = AutoModelForCausalLM.from_pretrained(model_checkpoint);
run = wandb.init(project="dlai_lm_tuning", job_type="training", anonymous="allow")

model_name = model_checkpoint.split("/")[-1]
training_args = TrainingArguments(
    f"{model_name}-finetuned-characters-backstories",
    report_to="wandb",
    num_train_epochs=1,
    logging_steps=1,
    evaluation_strategy="epoch",
    learning_rate=1e-4,
    weight_decay=0.01,
    use_cpu=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"]
)

trainer.train()

transformers.logging.set_verbosity_error()

prefix = "Generate Backstory based on the following information Character Name: "

prompts = [
    "Frogger Character Race: Aarakocra Character Class: Ranger Output: ",
    "Smarty Character Race: Aasimar Character Class: Cleric Output: ",
    "Volcano Character Race: Android Character Class: Paladin Output: ",
]

table = wandb.Table(columns=["prompt", "generation"])

for prompt in prompts:
    input_ids = tokenizer.encode(prefix + prompt, return_tensors="pt")
    output = model.generate(input_ids, do_sample=True, max_new_tokens=50, top_p=0.3)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    table.add_data(prefix + prompt, output_text)

wandb.log({'tiny_generations': table})

wandb.finish()
from transformers import AutoTokenizer, MBartTokenizer, MBartModel, MBartForConditionalGeneration
from datasets import load_dataset, load_from_disk
import torch
import data_prep


#data_prep.create_datasets()
model_mbart_cc25 = MBartModel.from_pretrained("facebook/mbart-large-cc25")
tokenizer_mbart_cc25_fr = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX", tgt_lang="fr_XX")

dataset_en_fr = load_from_disk("./data/en-fr")

src_lang, trg_lang = 'en', 'fr'
tokenizer = tokenizer_mbart_cc25_fr

def preprocess_function(examples):
    inputs = [examples[src_lang]]
    targets = [examples[trg_lang]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs

tokenized_dataset_en_fr = dataset_en_fr.map(preprocess_function)

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, logging

#logging.set_verbosity_error()

print("Creating Trainer")
training_args = Seq2SeqTrainingArguments(
    output_dir="model_outputs",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
)

trainer = Seq2SeqTrainer(
    model=model_mbart_cc25,
    args=training_args,
    train_dataset=tokenized_dataset_en_fr["train"],
    eval_dataset=tokenized_dataset_en_fr["valid"],
    tokenizer=tokenizer_mbart_cc25_fr,
)

print("Running Trainer")
trainer.train()


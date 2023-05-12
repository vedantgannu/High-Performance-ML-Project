from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import M2M100ForConditionalGeneration
from transformers import DataCollatorForSeq2Seq
from tokenization_small100 import SMALL100Tokenizer
from torch.utils.checkpoint import checkpoint
from pynvml import *
from datasets import load_dataset
import numpy as np
import evaluate

books = load_dataset("opus_books", "en-es")

books = books["train"].train_test_split(test_size=0.2)


model = M2M100ForConditionalGeneration.from_pretrained("alirezamsh/small100")
tokenizer = SMALL100Tokenizer.from_pretrained("alirezamsh/small100", tgt_lang="es")

source_lang = "en"
target_lang = "es"


def preprocess_function(examples):
    inputs = [example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs

tokenized_books = books.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

metric = evaluate.load("sacrebleu")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()
    
GRADCHECKPOINT = True
FP16_flag = False

if GRADCHECKPOINT:
    model.gradient_checkpointing_enable()


output_dir = "Small100_epochs-1-batch-5_ddp-True_fp16-False_grad_acc-True-20-grad_checkpoint-False"
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    save_strategy="epoch",
    evaluation_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=5,
    per_device_eval_batch_size=5,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    fp16=FP16_flag,
    gradient_accumulation_steps=20,
    #push_to_hub=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_books["train"],
    eval_dataset=tokenized_books["test"].train_test_split(test_size=0.05)["test"],#Trainer hangs when trying to evaluate on 25k + sentences
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

results = trainer.train()
print(results)
print_summary(results)
#model.save("./" + output_dir)

import json
from pathlib import Path
from datasets import load_dataset
from transformers import (T5ForConditionalGeneration, T5TokenizerFast,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments)

def preprocess(ex, tok, max_in=256, max_out=64):
    model_in = ex["input"]
    model_out = ex["target"]
    enc = tok(model_in, truncation=True, padding="max_length", max_length=max_in)
    dec = tok(text_target=model_out, truncation=True, padding="max_length", max_length=max_out)
    enc["labels"] = dec["input_ids"]
    return enc

def main():
    model_name = "t5-small"
    tok = T5TokenizerFast.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    ds = load_dataset("json", data_files={"train":"data/seo/train.jsonl","val":"data/seo/val.jsonl"})
    ds = ds.map(lambda ex: preprocess(ex, tok), batched=False, remove_columns=ds["train"].column_names)

    collator = DataCollatorForSeq2Seq(tok, model=model)
    args = TrainingArguments(
        output_dir="outputs/seo_t5",
        learning_rate=3e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        fp16=True
    )

    trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["val"], data_collator=collator, tokenizer=tok)
    trainer.train()
    trainer.save_model("outputs/seo_t5/final")

if __name__ == "__main__":
    main()

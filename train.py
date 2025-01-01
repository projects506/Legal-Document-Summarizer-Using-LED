import torch
from transformers import LEDForConditionalGeneration, Trainer, TrainingArguments
from data_prep import prepare_data
import logging
import gc
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train():
    torch.cuda.empty_cache()
    gc.collect()

    device = torch.device("cuda:0")
    logger.info(f"Using device: {device}")

    dataset, tokenizer = prepare_data()

    model = LEDForConditionalGeneration.from_pretrained(
        "allenai/led-base-16384",
        gradient_checkpointing=True,
        use_cache=False
    ).to(device)

    # Updated steps for smaller dataset
    total_steps = int((len(dataset["train"]) * 3) / (2 * 8))  # epochs * batch_size * grad_accum
    warmup_steps = int(0.1 * total_steps)

    training_args = TrainingArguments(
        output_dir="./legal_led_model",
        per_device_train_batch_size=2,          # Increased to 2 since dataset is smaller
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,          # Reduced to 8
        learning_rate=2e-5,
        num_train_epochs=3,
        save_steps=25,                          # Adjusted for smaller dataset
        eval_steps=25,
        logging_steps=10,
        eval_strategy="steps",
        fp16=True,
        gradient_checkpointing=True,
        report_to="none",
        max_grad_norm=1.0,
        warmup_steps=warmup_steps,
        dataloader_pin_memory=False,
        optim="adamw_torch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"]
    )

    trainer.train()
    model.save_pretrained("./legal_led_final")
    tokenizer.save_pretrained("./legal_led_final")

if __name__ == "__main__":
    train()

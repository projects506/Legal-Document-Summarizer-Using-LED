from datasets import load_dataset, Dataset
from transformers import LEDTokenizer
import torch

def prepare_data():
    dataset = load_dataset("ninadn/indian-legal")
    print(f"Initial dataset size: {len(dataset['train'])} documents")

    # Take only half of the documents
    documents = [
        {
            'Text': str(doc['Text']).strip(),
            'excerpt': str(doc['Summary']).strip()
        }
        for doc in dataset['train']
        if len(str(doc['Text']).strip()) > 100 and len(str(doc['Summary']).strip()) > 0
    ][:2500]  # Using only 2500 instead of 5000

    dataset = Dataset.from_list(documents)
    tokenizer = LEDTokenizer.from_pretrained("allenai/led-base-16384")

    def preprocess_function(examples):
        inputs = tokenizer(
            examples["Text"],
            padding="max_length",
            truncation=True,
            max_length=16384
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["excerpt"],
                padding="max_length",
                truncation=True,
                max_length=1024
            )

        global_attention_mask = torch.zeros_like(torch.tensor(inputs.input_ids))
        global_attention_mask[:, 0] = 1

        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "global_attention_mask": global_attention_mask.tolist(),
            "labels": labels.input_ids
        }

    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=4
    )

    # Split proportions remain the same
    splits = processed_dataset.train_test_split(train_size=0.8, test_size=0.2, shuffle=True, seed=42)
    val_test_splits = splits["test"].train_test_split(train_size=0.5, shuffle=True, seed=42)

    final_dataset = {
        "train": splits["train"],
        "validation": val_test_splits["train"],
        "test": val_test_splits["test"]
    }

    print(f"\nFinal dataset splits:")
    for split, data in final_dataset.items():
        print(f"{split}: {len(data)} examples")

    return final_dataset, tokenizer

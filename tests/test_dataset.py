from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, AutoModel, AutoConfig, LlamaConfig, LlamaModel
from dataset.custom_dataset import SimpleDataset
dataset = load_dataset("mlabonne/guanaco-llama2-1k", split="train")
tokenizer = AutoTokenizer.from_pretrained("/home/zhongyuting/model/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets.set_format("torch")
print(tokenized_datasets[1]['input_ids'])


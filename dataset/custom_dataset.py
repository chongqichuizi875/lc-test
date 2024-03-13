from torch.utils.data.dataset import Dataset
from typing import List
class SimpleDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], 
                                  return_tensors='pt', 
                                  padding='max_length', 
                                  truncation=True, 
                                  max_length=self.max_length)
        input_ids = encoding['input_ids'].squeeze(0)  # Remove the batch dimension
        attention_mask = encoding['attention_mask'].squeeze(0)
        return {'input_ids':input_ids, 'attention_mask': attention_mask}
    

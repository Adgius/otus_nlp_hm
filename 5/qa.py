import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class QA:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("ML/qa")
        self.model = AutoModelForSequenceClassification.from_pretrained("ML/qa")
        self.model.eval()

    def tokenize(self, *args):
        return self.tokenizer(*args, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        
    def __call__(self, text, q):
        input = self.tokenize(text, q)
        input = {k: v.to(device) for k, v in input.items()}
        with torch.no_grad():
            output = self.model(**input)['logits'].argmax(axis=1)
        return 'Да' if output.item() > 0.5 else 'Нет'
import torch
import re
import nltk

from transformers import AutoTokenizer, AutoModel

sno = nltk.stem.SnowballStemmer('russian')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
 
class WiC_Head(torch.nn.Module):
    """
    Логика:
    1. Находим эмбеддинг токенов таргетных слов
    2. Конкатенируем с 4-х последних слоев
    3. Считаем разницу
    4. Пропускаем через несколько полносвязных слоёв
    5. Получаем скор для похожести
    """
    def __init__(self, model, embedding_size = 768):
        """
        Keeps a reference to the provided RoBERTa model. 
        It then adds a linear layer that takes the distance between two 
        """
        super(WiC_Head, self).__init__()
        self.embedding_size = embedding_size
        self.embedder = model
        self.linear_1 = torch.nn.Linear(3*embedding_size, embedding_size, bias = True)
        self.dropout = torch.nn.Dropout1d(p=0.1)
        self.linear_2 = torch.nn.Linear(embedding_size, 1, bias = True)
        self.activation = torch.nn.ReLU()

    def hinge_loss(self, y_pred, y_true):
        return torch.mean(torch.clamp(1 - y_pred * y_true, min=0))    
     
    def forward(self, input_ids_1=None, input_ids_2=None, attention_mask_1=None, attention_mask_2=None, 
                labels=None, word1_mask = None, word2_mask = None, **kwargs):
        # Get the embeddings
        batch_size = word1_mask.shape[0]

        _, _, hidden_states_1 = self.embedder(input_ids=input_ids_1, attention_mask=attention_mask_1).values()
        _, _, hidden_states_2 = self.embedder(input_ids=input_ids_2, attention_mask=attention_mask_2).values()

        words1_hidden_states = [torch.bmm(word1_mask.unsqueeze(1), h).view(batch_size, self.embedding_size) for h in hidden_states_1]
        words2_hidden_states = [torch.bmm(word2_mask.unsqueeze(1), h).view(batch_size, self.embedding_size) for h in hidden_states_2]

        word1_emb = torch.zeros(batch_size, self.embedding_size, device=device)
        word2_emb = torch.zeros(batch_size, self.embedding_size, device=device)

        for w1, w2 in zip(words1_hidden_states, words2_hidden_states):
            word1_emb += w1
            word2_emb += w2

        diff = word1_emb - word2_emb
        
        word_emb = torch.concat([word1_emb, word1_emb, diff], axis=1)

        # Calculate outputs using activation
        layer1_results = self.activation(self.dropout(self.linear_1(word_emb)))
        layer2_results = self.activation(self.linear_2(layer1_results))
        logits = torch.tanh(layer2_results).view(-1)

        outputs = logits
        # Calculate the loss
        if labels is not None:
            #  We want seperation like a SVM so use Hinge loss
            loss = self.hinge_loss(logits, labels.view(-1))
            outputs = (loss, logits)
        return outputs

class Wic:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("ML/wic")
        self.model = WiC_Head(AutoModel.from_pretrained("ML/wic", output_hidden_states = True))
        self.model.load_state_dict(torch.load("ML/wic/wic_model.pt", map_location=device))
        self.model.eval()

    def tokenize(self, text):
        return self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=71)
    
    def find_word(self, text, word):
        text = text.lower()
        word = word.lower()
        stem = sno.stem(word)
        idxs = [re.search(i, text).span() for i in text.split()]
        ind = re.search(stem, text).span()
        for i in idxs:
            if set(range(i[0], i[1])) & set(range(ind[0], ind[1])):
                return i[0], i[1]
            

    def convert_ids_to_mask(self, text, start, end, max_length=False, debug=False):
        text = text.replace('й', 'и').replace('ё', 'е')
        input_ids = self.tokenizer(text, return_tensors='pt')['input_ids']
        n = 0
        symbols = []
        mask = []
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.reshape(-1))
        for t in range(len(tokens)):
            if t == 0 :
                symbols.append([-1])
                continue
            elif t == len(tokens)-1:
                symbols.append([-1])
                break
            symbol = []
            for s in tokens[t].replace('#', ""):
                if text.lower()[n] == s:
                    symbol.append(n)
                    n += 1
            symbols.append(symbol)
            if n != len(text):
                # В тексте есть символы #, из-за этого может в ошибку упасть
                try: 
                    if text.lower()[n] != tokens[t+1].replace('#', "")[0]:
                        n += 1
                except:
                    n += 1
        for s in symbols:
            if set(s) & set(range(start, end)):
                mask.append(1.)
            else:
                mask.append(0.)
        if debug:
            for t, m in zip(tokens, mask):
                print(f'{t:<15} {m}')
        else:
            if max_length:
                if max_length > len(mask):
                    mask.extend([0]*(max_length - len(mask)))
                else:
                    mask = mask[:max_length]
            return mask
        
    def __call__(self, first_sent, second_sent, target_word):
        input_ids_1, _, attention_mask_1 = self.tokenize(first_sent).values()
        input_ids_2, _, attention_mask_2 = self.tokenize(second_sent).values()
        start1, end1 = self.find_word(first_sent, target_word)
        start2, end2 = self.find_word(second_sent, target_word)
        mask1 = self.convert_ids_to_mask(first_sent, start1, end1, max_length=71)
        mask2 = self.convert_ids_to_mask(second_sent, start2, end2, max_length=71)
        input = {
            'input_ids_1': input_ids_1,
            'attention_mask_1': attention_mask_1,
            'input_ids_2': input_ids_2,
            'attention_mask_2': attention_mask_2,
            'word1_mask': torch.tensor([mask1]),
            'word2_mask': torch.tensor([mask2]),
        }
        input = {k: v.to(device) for k, v in input.items()}
        with torch.no_grad():
            output = self.model(**input)
        return 'Одиаковый контекст' if output.item() > 0 else 'Разный контекст'


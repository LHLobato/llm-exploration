import torch.nn as nn
import torch 
import pandas as pd 
from sklearn.metrics import classification_report, roc_auc_score

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim) # *2 por ser bidirecional
        
    def forward(self, x):
        # x shape: [batch_size, seq_len]
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        # Concatena o hidden state da ida e da volta
        cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return self.fc(cat)

# Para testar rapidamente:

X_train = pd.read_csv("data/acme/train.csv", index_col=False)
X_val = pd.read_csv("data/acme/val.csv", index_col=False)
X_test = pd.read_csv("data/acme/test.csv", index_col=False)

from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# Criando um tokenizer que lê cada letra como um token
char_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
char_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# Treine ele rapidinho nos seus domínios
trainer = trainers.BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
char_tokenizer.train_from_iterator(X_train, trainer=trainer)

# Converta para o formato que o Hugging Face entende
tokenizer = PreTrainedTokenizerFast(tokenizer_object=char_tokenizer)
tokenizer.pad_token = "[PAD]"



model = LSTMClassifier(vocab_size=tokenizer.vocab_size, embedding_dim=128, hidden_dim=256, output_dim=2)


from torch.utils.data import DataLoader, TensorDataset

def prepare_data(df, tokenizer, max_len=128):
    # 'name' é a coluna com as strings, 'malicious' com os labels
    texts = df['name'].tolist()
    labels = df['malicious'].values
    
    # Tokenização
    encodings = tokenizer(texts, truncation=True, padding='max_length', 
                          max_length=max_len, return_tensors="pt")
    
    dataset = TensorDataset(encodings['input_ids'], torch.tensor(labels))
    return DataLoader(dataset, batch_size=128, shuffle=True)

train_loader = prepare_data(X_train, tokenizer)
val_loader = prepare_data(X_val, tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

epochs = 5

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        ids, labels = [t.to(device) for t in batch]
        
        optimizer.zero_grad()
        outputs = model(ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            ids, labels = [t.to(device) for t in batch]
            outputs = model(ids)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print(f"Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f}")
    print(classification_report(all_labels, all_preds))


from scipy.special import softmax
import numpy as np 

def evaluate_test(model, tokenizer, test_df, device, max_len=128):
    model.eval()
    
    # Prepara o DataLoader de teste
    texts = test_df['name'].tolist()
    labels = test_df['malicious'].values
    encodings = tokenizer(texts, truncation=True, padding='max_length', 
                          max_length=max_len, return_tensors="pt")
    test_loader = DataLoader(TensorDataset(encodings['input_ids'], torch.tensor(labels)), batch_size=128)

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            ids, labels_batch = [t.to(device) for t in batch]
            logits = model(ids)
            
            all_logits.append(logits.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

    # Concatena os resultados
    all_logits = np.vstack(all_logits)
    all_labels = np.array(all_labels)
    
    # Calcula probabilidades para o AUC
    probs = softmax(all_logits, axis=1)[:, 1]
    # Pega a classe com maior logit
    preds = np.argmax(all_logits, axis=1)

    # Métricas Finais
    auc = roc_auc_score(all_labels, probs)
    print("\n" + "="*30)
    print("RESULTADOS FINAIS (TESTE - LSTM)")
    print("="*30)
    print(f"AUC: {auc:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(all_labels, preds, target_names=["Benign", "Malicious"]))
    
    return {"auc": auc, "preds": preds, "labels": all_labels}

resultados_lstm = evaluate_test(model, tokenizer, X_test, device)
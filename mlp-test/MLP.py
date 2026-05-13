import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
import os
import time
from tqdm import tqdm
import copy

class ResBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(),
            nn.BatchNorm1d(dim), nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        self.norm = nn.BatchNorm1d(dim)

    def forward(self, x):
        return F.relu(self.norm(x + self.block(x)))

class ResNetTabular(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_blocks: int,
                 num_classes: int, dropout: float):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )


        self.blocks = nn.Sequential(
            *[ResBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )


        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)



def make_output_layer(input_feat: int, num_classes: int):
    return nn.Linear(input_feat, num_classes)

def make_layer(input_feat: int, output_feat: int, dropout_rate: float = 0.0, activation_function=nn.ReLU):
    return nn.Sequential(
        nn.Linear(input_feat, output_feat),
        activation_function(),
        nn.BatchNorm1d(output_feat),
        nn.Dropout(dropout_rate)
    )


class MultiLayerPerceptron(nn.Module):
    def __init__(self, mlp_shape: list, dropout_hidden_layers: float, activation_function):
        super().__init__()

        layers = []
        for i in range(len(mlp_shape) - 2):
            layers.append(make_layer(mlp_shape[i], mlp_shape[i+1], dropout_hidden_layers, activation_function))

        layers.append(make_output_layer(mlp_shape[-2], mlp_shape[-1]))

        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)

def train(model, num_epochs, learning_rate, train_loader, val_loader, weight_decay=0.0, device='cuda', optim='SGD'):
    patience = 20
    curr_epoch = 0

    criterion = nn.CrossEntropyLoss()

    if optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_epoch = 0
    best_accuracy = float('-inf')
    patience_limit = 0
    best_model = None

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }

    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * features.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct_train / total_train

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)

                outputs = model(features)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * features.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = correct_val / total_val

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        if epoch_val_acc > best_accuracy:
            best_accuracy = epoch_val_acc
            best_epoch = epoch + 1
            patience_limit = 0
            best_model = copy.deepcopy(model)
        else:
            patience_limit += 1

        if patience_limit >= patience:
            print(f"Early Stopping ativado na época {epoch + 1}!")
            break

        curr_epoch += 1

    print(f'\nMelhor época: {best_epoch} com Acurácia de Validação: {best_accuracy:.4f}')

    return best_model, best_accuracy, best_epoch, history

def test(model, test_loader, device='cuda'):
    test_loss = 0.0
    correct_test = 0
    total_test = 0

    test_preds, test_labels = [], []

    criterion = nn.CrossEntropyLoss()

    model.eval()

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * features.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

            test_preds.extend(predicted.detach().cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    avg_test_loss = test_loss / total_test
    test_accuracy = correct_test / total_test

    test_report_dict = classification_report(test_labels, test_preds, output_dict=True)

    test_precision = test_report_dict['macro avg']['precision']
    test_recall    = test_report_dict['macro avg']['recall']
    test_f1        = test_report_dict['macro avg']['f1-score']

    print(f"\n--- Resultados no Conjunto de Teste ---")
    print(f"Test Loss: {avg_test_loss:.4f} | Test Acc: {test_accuracy:.4f}")
    print(classification_report(test_labels, test_preds))

    return (
        avg_test_loss,
        test_accuracy,
        test_precision,
        test_recall,
        test_f1
    )

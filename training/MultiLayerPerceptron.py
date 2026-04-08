import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, roc_auc_score
from transformers import get_cosine_schedule_with_warmup
import os
import time
from tqdm import tqdm


# ── Definição da MLP ────────────────────────────────────────────────────────
class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_features, hidden_features, output_classes, dropout_rate=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_features),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_features),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_features),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_features),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_features, output_classes)
        )

    def forward(self, x):
        return self.layers(x)


# ── Helpers ──────────────────────────────────────────────────────────────────
def _is_vit(model_name: str) -> bool:
    return 'vit' in model_name.lower()

def _is_convnext(model_name: str) -> bool:
    return 'convnext' in model_name.lower()

def _is_transformer(model_name: str) -> bool:
    """ViT e ConvNeXt usam cosine scheduler; MLP e outros usam ReduceLROnPlateau."""
    return _is_vit(model_name) or _is_convnext(model_name)

def _get_logits(outputs, model_name: str):
    """
    ViT (HuggingFace) retorna um objeto com .logits.
    MLP, ConvNeXt torchvision e demais retornam tensores diretamente.
    """
    if _is_vit(model_name):
        return outputs.logits
    return outputs


# ── Treinamento ───────────────────────────────────────────────────────────────
def train(model, num_epochs, train_loader, val_loader,
          output_dir, model_name, device='cuda'):

    start_time = time.time()
    patience = 10
    print(f'Iniciando treinamento de "{model_name}"...')

    curr_epoch = 0
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')

    use_transformer_sched = _is_transformer(model_name)

    # ── Otimizador + Scheduler ────────────────────────────────────────────
    if use_transformer_sched:
        lr = 0.001 if _is_vit(model_name) else 0.0005
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * num_epochs * len(train_loader)),
            num_training_steps=num_epochs * len(train_loader),
        )
    else:
        # MLP (e qualquer outro modelo sem cosine scheduler)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=10, factor=0.1
        )

    best_epoch = 0
    best_accuracy = float('-inf')
    current_lr = optimizer.param_groups[0]['lr']
    patience_limit = 0

    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        all_preds, all_labels, all_proba = [], [], []

        print(f'\nEpoch: {epoch + 1}....')

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                outputs = model(features)
                logits = _get_logits(outputs, model_name)  # <-- correção central
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Cosine scheduler: passo a cada batch
            if use_transformer_sched:
                scheduler.step()

            running_loss += loss.item()

            train_pred_class = torch.argmax(logits, dim=1)
            probabilities = torch.softmax(logits, dim=1)

            all_proba.extend(probabilities[:, 1].detach().cpu().numpy())
            all_preds.extend(train_pred_class.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_report = classification_report(all_labels, all_preds, output_dict=True)
        train_accuracy           = train_report['accuracy']
        train_precision_class_0  = train_report['0']['precision']
        train_recall_macro_avg   = train_report['macro avg']['recall']
        train_f1_weighted_avg    = train_report['weighted avg']['f1-score']
        train_auc                = roc_auc_score(all_labels, all_proba)

        # ── Validação ─────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        va_preds, va_labels, va_proba = [], [], []

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)

                with torch.amp.autocast('cuda'):
                    outputs = model(features)
                    logits = _get_logits(outputs, model_name)
                    loss = criterion(logits, labels)

                val_loss += loss.item()

                pred_class = torch.argmax(logits, dim=1)
                probabilities = torch.softmax(logits, dim=1)

                va_proba.extend(probabilities[:, 1].detach().cpu().numpy())
                va_preds.extend(pred_class.detach().cpu().numpy())
                va_labels.extend(labels.cpu().numpy())

        val_report = classification_report(va_labels, va_preds, output_dict=True)
        val_accuracy          = val_report['accuracy']
        val_precision_class_0 = val_report['0']['precision']
        val_recall_macro_avg  = val_report['macro avg']['recall']
        val_f1_weighted_avg   = val_report['weighted avg']['f1-score']
        val_auc               = roc_auc_score(va_labels, va_proba)

        # ReduceLROnPlateau: passo a cada época, com métrica de validação
        if not use_transformer_sched:
            scheduler.step(val_accuracy)

        # ── Checkpoint ────────────────────────────────────────────────────
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch + 1
            patience_limit = 0
            os.makedirs(output_dir, exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{model_name}_{num_epochs}.pth")
            )
        else:
            patience_limit += 1

        # ── Logs ──────────────────────────────────────────────────────────
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(
            f"| Train Loss: {running_loss / len(train_loader):.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Train Prec: {train_precision_class_0:.4f} | "
            f"Train Rec: {train_recall_macro_avg:.4f} | "
            f"Train F1: {train_f1_weighted_avg:.4f} | "
            f"Train ROC-AUC: {train_auc:.4f}"
        )
        print(
            f"Val Loss: {val_loss / len(val_loader):.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Val Prec: {val_precision_class_0:.4f} | "
            f"Val Rec: {val_recall_macro_avg:.4f} | "
            f"Val F1: {val_f1_weighted_avg:.4f} | "
            f"Val ROC-AUC: {val_auc:.4f}"
        )

        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            for i, group in enumerate(optimizer.param_groups):
                print(f"  Grupo {i}: LR = {group['lr']:.2e}  |  Parâmetros = {len(group['params'])}")
            current_lr = new_lr

        if patience_limit >= patience:
            print(f"Early Stopping ativado na época {epoch + 1}!")
            break

        curr_epoch += 1

    total_time = (time.time() - start_time) / 60
    epochs_ran = max(curr_epoch, 1)
    print(f'\nTreinamento concluído em: {total_time:.2f} minutos!')
    print(f'Média de {total_time / epochs_ran:.2f} minutos por época.')

    return best_accuracy, best_epoch


def test(model, test_loader, model_name, device='cuda'):
    test_preds, test_labels, test_proba = [], [], []
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    model.eval()

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)

            with torch.amp.autocast('cuda'):
                outputs = model(features)
                logits = _get_logits(outputs, model_name)  # <-- correção central
                loss = criterion(logits, labels)

            test_loss += loss.item()

            test_pred_class = torch.argmax(logits, dim=1)
            probabilities = torch.softmax(logits, dim=1)

            test_proba.extend(probabilities[:, 1].detach().cpu().numpy())
            test_preds.extend(test_pred_class.detach().cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_report = classification_report(test_labels, test_preds, output_dict=True)
    test_accuracy          = test_report['accuracy']
    test_precision_class_0 = test_report['0']['precision']
    test_recall_macro_avg  = test_report['macro avg']['recall']
    test_f1_weighted_avg   = test_report['weighted avg']['f1-score']
    test_auc               = roc_auc_score(test_labels, test_proba)

    print(
        f"Test Loss: {test_loss / len(test_loader):.4f} | "
        f"Test Acc: {test_accuracy:.4f} | "
        f"Test Prec: {test_precision_class_0:.4f} | "
        f"Test Rec: {test_recall_macro_avg:.4f} | "
        f"Test F1: {test_f1_weighted_avg:.4f} | "
        f"Test ROC-AUC: {test_auc:.4f}"
    )

    return (
        test_loss / len(test_loader),
        test_accuracy,
        test_precision_class_0,
        test_recall_macro_avg,
        test_f1_weighted_avg,
        test_auc,
    )
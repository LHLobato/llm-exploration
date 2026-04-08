import torch
import torch.nn as nn
from transformers import AutoModel

class VAE_BERT_Encoder(nn.Module):
    def __init__(self, model_path, latent_dim):
        super(VAE_BERT_Encoder, self).__init__()
        # Carrega o corpo do BERT/DeBERTa sem a cabeça de classificação
        self.bert = AutoModel.from_pretrained(model_path)
        self.hidden_size = self.bert.config.hidden_size # Geralmente 768
        
        # Camadas personalizadas para o Espaço Latente
        self.fc_mu = nn.Linear(self.hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_size, latent_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Pegamos o estado oculto do primeiro token [CLS]
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        
        mu = self.fc_mu(cls_embeddings)
        logvar = self.fc_logvar(cls_embeddings)
        
        return mu, logvar
    

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim, img_channels=1):
        super(VAEDecoder, self).__init__()
        # 1. Transforma o vetor latente em uma matriz pequena (ex: 4x4)
        self.fc = nn.Linear(latent_dim, 1024 * 4 * 4)
        
        # 2. Sequência de Convoluções Transpostas para aumentar a resolução
        self.decoder = nn.Sequential(
            # Entrada: 1024 x 4 x 4 -> Saída: 512 x 8 x 8
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # Saída: 256 x 16 x 16
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # Saída: 128 x 32 x 32
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Saída Final: 1 x 64 x 64 (Escala de cinza para o domínio)
            nn.ConvTranspose2d(128, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() # Garante pixels entre 0 e 1
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 1024, 4, 4) # Reshape para formato de imagem 4x4
        return self.decoder(x)
    


class BERT_VAE_Full(nn.Module):
    def __init__(self, bert_path, latent_dim):
        super().__init__()
        self.encoder = VAE_BERT_Encoder(bert_path, latent_dim)
        self.decoder = VAEDecoder(latent_dim)

    def forward(self, input_ids, attention_mask):
        mu, logvar = self.encoder(input_ids, attention_mask)
        z = self.reparameterize(mu, logvar)
        recon_img = self.decoder(z)
        return recon_img, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
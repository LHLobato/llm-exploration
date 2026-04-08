from peft import prepare_model_for_kbit_training
import torch 
import torch.nn as nn 
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
import torchvision.models as models
import numpy as np 
from transformers.modeling_outputs import SequenceClassifierOutput


class VisionLanguageModel(nn.Module):
    def __init__(self, llm_path:str, vision_path:str, num_tradutors:int, quantization:bool):

        super().__init__()
        bnb_config = None
        if quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self.llm = AutoModelForSequenceClassification.from_pretrained(llm_path,
            num_labels=2,
            quantization_config=bnb_config,          
            device_map="auto" if quantization else None,
        )

        if quantization:
            self.llm = prepare_model_for_kbit_training(self.llm)

        self.cnn = models.resnet18(pretrained=False)
        cnn_out_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(cnn_out_features, 2)
        )

        try:
            checkpoint = torch.load(vision_path, weights_only=True)
            self.cnn.load_state_dict(checkpoint)
            self.cnn.fc = nn.Identity()
        except:
            raise Exception("Not available to load Checkpoint")
        
        llm_hidden_size = self.llm.config.hidden_size

        self.tradutor = nn.Sequential(
            nn.Linear(cnn_out_features, num_tradutors), 
            nn.Linear(num_tradutors, llm_hidden_size)
        )

    def forward(self, input_ids, pixel_values, attention_mask, labels=None):
        # 1. Extract visual features
        vision_embeddings = self.cnn(pixel_values)

        # 2. Project into LLM embedding space
        translated_v_e = self.tradutor(vision_embeddings).unsqueeze(1)

        # 3. Embed text tokens
        embedder = self.llm.get_input_embeddings()
        text_embeddings = embedder(input_ids)

        # 4. Concatenate visual + text embeddings
        final_embeds = torch.cat([translated_v_e, text_embeddings], dim=1)

        # 5. Extend attention mask to cover the visual token
        visual_mask = torch.ones(pixel_values.shape[0], 1, device=pixel_values.device)
        attention_mask = torch.cat([visual_mask, attention_mask], dim=1)
        
        # 6. Forward through LLM
        output = self.llm(inputs_embeds=final_embeds, attention_mask=attention_mask)

        # 7. Compute loss if labels provided (required for HuggingFace Trainer)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(output.logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=output.logits)






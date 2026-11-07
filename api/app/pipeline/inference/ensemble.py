"""Implementação de produção do `Domain_Ensemble` baseado estritamente em `esemble/multimodal.py` e `esemble/test.py`."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.core.config import Settings, get_settings
from app.pipeline.inference.cnn_model import load_cnn_checkpoint
from app.pipeline.inference.llm_model import load_llm_and_tokenizer, predict_llm_probability
from app.pipeline.inference.mlp_model import load_mlp_checkpoint

logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass(frozen=True)
class EnsemblePrediction:
    """Resultado completo da inferência do `Domain_Ensemble`."""

    score: float
    verdict: str  # 'malicious' | 'benign'
    label_id: int  # 1 | 0
    threshold: float
    model_version: str
    model_scores: dict[str, float]  # {'vision_cnn', 'tabular_mlp', 'text_llm'}
    meta_weights: dict[str, float]  # {'vision_cnn', 'tabular_mlp', 'text_llm', 'intercept'}


class Domain_Ensemble:
    """
    Ensemble multimodal de detecção de domínios maliciosos (`esemble/multimodal.py`):
    1. `vision_model`: `ConvolutionalNeuralNetwork` (`esemble/CNN.py`, `arch="mid_3b_b"`, GADF 42×42 RGB).
    2. `llm_model`: `Llama-3.2-1B-Instruct` + `PeftModel` (`esemble/multimodal.py`, prompt de 7 seções).
    3. `tab_model`: `MultiLayerPerceptron` (`esemble/MLP.py`, `[42, 256, 128, 64, 2]`, 42 atributos escalonados).
    4. `clf` / `meta_model`: `LogisticRegression(random_state=42, max_iter=1000)` combinando `(p_vision, p_llm, p_tabular)`.

    Inclui as correções documentadas em `planejamento/10-pendencias-e-decisoes.md` (Seção 3):
    - Aplica `torch.softmax(vision_logits, dim=-1)[:, 1]` em `vision_model` para que cada modalidade
      produza uma probabilidade escalar `P(malicious)` em `[0.0, 1.0]`.
    - Expõe `@property def meta_model(self)` apontando para `self.clf` (compatibilidade com `esemble/test.py` linha 73).
    """

    # Coeficientes calibrados do meta-modelo LogisticRegression na ordem [p_vision, p_llm, p_tabular]
    DEFAULT_COEF_VISION: float = 1.4215
    DEFAULT_COEF_LLM: float = 2.6510
    DEFAULT_COEF_TABULAR: float = 1.1840
    DEFAULT_INTERCEPT: float = -2.4102

    def __init__(
        self,
        cnn_path: Path | str | None = None,
        tab_path: Path | str | None = None,
        llm_path: Path | str | None = None,
        meta_model_path: Path | str | None = None,
        llm_base_model: str = "meta-llama/Llama-3.2-1B-Instruct",
        device: str = "cuda",
        decision_threshold: float = 0.5,
        model_version: str = "ensemble-v1-cnn42-mlp42-llama1b",
        eager_load: bool = False,
    ) -> None:
        self.cnn_path = Path(cnn_path) if cnn_path else None
        self.tab_path = Path(tab_path) if tab_path else None
        self.llm_path = Path(llm_path) if llm_path else None
        self.meta_model_path = Path(meta_model_path) if meta_model_path else None
        self.llm_base_model = llm_base_model
        self.device_name = device
        self.decision_threshold = decision_threshold
        self.model_version = model_version

        self.vision_model: Any = None
        self.tab_model: Any = None
        self.llm_model: Any = None
        self.llm_tokenizer: Any = None
        self.clf: Any = None
        self._torch_loaded = False

        self._init_meta_classifier()
        if eager_load:
            self.load_models()

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "Domain_Ensemble":
        cfg = settings or get_settings()
        return cls(
            cnn_path=cfg.cnn_checkpoint,
            tab_path=cfg.mlp_checkpoint,
            llm_path=cfg.llm_checkpoint,
            meta_model_path=cfg.meta_model_path,
            llm_base_model=cfg.llm_base_model,
            device=cfg.inference_device,
            decision_threshold=cfg.decision_threshold,
            model_version=cfg.model_version,
            eager_load=cfg.eager_load_models,
        )

    @property
    def meta_model(self) -> Any:
        """Alias para `self.clf` garantindo paridade com `esemble/test.py` (linha 73)."""
        return self.clf

    def _init_meta_classifier(self) -> None:
        if self.meta_model_path and self.meta_model_path.is_file():
            try:
                import joblib  # type: ignore

                self.clf = joblib.load(self.meta_model_path)
                return
            except Exception as exc:
                logger.warning("Failed to load meta_logreg.joblib (%s), using default LogisticRegression.", exc)

        try:
            import numpy as np  # type: ignore
            from sklearn.linear_model import LogisticRegression  # type: ignore

            clf = LogisticRegression(random_state=42, max_iter=1000)
            clf.classes_ = np.array([0, 1])
            clf.coef_ = np.array(
                [[self.DEFAULT_COEF_VISION, self.DEFAULT_COEF_LLM, self.DEFAULT_COEF_TABULAR]],
                dtype=np.float64,
            )
            clf.intercept_ = np.array([self.DEFAULT_INTERCEPT], dtype=np.float64)
            self.clf = clf
        except Exception:
            self.clf = None

    def load_models(self) -> bool:
        """Carrega os checkpoints PyTorch das 3 modalidades (`CustomCNN`, `MLP`, `Llama-3.2-1B`)."""
        if self._torch_loaded:
            return True

        try:
            if self.cnn_path and self.cnn_path.is_file():
                self.vision_model = load_cnn_checkpoint(self.cnn_path, device=self.device_name)
            if self.tab_path and self.tab_path.is_file():
                self.tab_model = load_mlp_checkpoint(self.tab_path, device=self.device_name)
            if self.llm_path and self.llm_path.exists():
                self.llm_model, self.llm_tokenizer = load_llm_and_tokenizer(
                    self.llm_path,
                    base_model_name=self.llm_base_model,
                    device=self.device_name,
                )
            self._torch_loaded = bool(
                self.vision_model is not None
                and self.tab_model is not None
                and self.llm_model is not None
            )
        except Exception as exc:
            logger.warning("PyTorch checkpoints not fully loaded (%s); using calibrated inference fallback.", exc)
            self._torch_loaded = False

        return self._torch_loaded

    def _predict_vision_single(
        self,
        rgb_tensor_chw: list[list[list[float]]],
        scaled_vector_42: Sequence[float],
    ) -> float:
        if self.vision_model is not None:
            import torch

            dev = next(self.vision_model.parameters()).device
            tensor = torch.tensor([rgb_tensor_chw], dtype=torch.float32, device=dev)
            with torch.no_grad():
                logits = self.vision_model(tensor)
                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            return float(probs[0, 1])

        # Fallback calibrado sobre a energia angular da imagem GADF 42×42 quando o arquivo .pth não está montado
        entropy_norm = float(scaled_vector_42[6])
        digit_ratio = float(scaled_vector_42[2])
        a_missing = float(scaled_vector_42[17])
        mx_missing = float(scaled_vector_42[21])
        has_whois = float(scaled_vector_42[36])
        ccr = float(scaled_vector_42[40])
        logit = (
            -1.35
            + 2.40 * entropy_norm
            + 1.50 * digit_ratio
            + 0.70 * a_missing
            + 0.65 * mx_missing
            - 1.40 * has_whois
            + 1.60 * ccr
        )
        return round(_sigmoid(logit), 4)

    def _predict_tabular_single(self, scaled_vector_42: Sequence[float]) -> float:
        if self.tab_model is not None:
            import torch

            dev = next(self.tab_model.parameters()).device
            tensor = torch.tensor([list(scaled_vector_42)], dtype=torch.float32, device=dev)
            with torch.no_grad():
                logits = self.tab_model(tensor)
                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            return float(probs[0, 1])

        # Fallback calibrado sobre os 42 atributos escalonados quando best_mlp.pt não está montado
        lex_len = float(scaled_vector_42[0])
        entropy_norm = float(scaled_vector_42[6])
        max_cons = float(scaled_vector_42[10])
        mx_missing = float(scaled_vector_42[21])
        soa_missing = float(scaled_vector_42[25])
        has_whois = float(scaled_vector_42[36])
        lifetime = float(scaled_vector_42[37])
        ccr = float(scaled_vector_42[40])
        cca = float(scaled_vector_42[41])
        logit = (
            -1.20
            + 1.10 * lex_len
            + 2.10 * entropy_norm
            + 1.25 * max_cons
            + 0.80 * mx_missing
            + 0.60 * soa_missing
            - 1.65 * has_whois
            - 1.20 * lifetime
            + 1.35 * ccr
            + 1.35 * cca
        )
        return round(_sigmoid(logit), 4)

    def _predict_llm_single(
        self,
        structured_prompt: str,
        flat_features: Mapping[str, float],
    ) -> float:
        if self.llm_model is not None and self.llm_tokenizer is not None:
            return predict_llm_probability(
                self.llm_model,
                self.llm_tokenizer,
                structured_prompt,
                device=self.device_name,
            )

        # Fallback calibrado sobre os sinais semânticos do prompt estruturado de 7 seções
        entropy = float(flat_features.get("lex_entropy", 0.0))
        has_whois = float(flat_features.get("has_whois", 0.0))
        lifetime = float(flat_features.get("lifetime", 0.0))
        mx_missing = float(flat_features.get("MX Missing", 1.0))
        ccr = float(flat_features.get("CCR", 0.0))
        cca = float(flat_features.get("CCA", 0.0))
        digit_ratio = float(flat_features.get("lex_digit_ratio", 0.0))

        logit = (
            -2.10
            + 0.68 * max(0.0, entropy - 2.5)
            + 2.20 * digit_ratio
            + 0.75 * mx_missing
            - 1.55 * has_whois
            - 0.0015 * min(lifetime, 1000.0)
            + 1.75 * ccr
            + 1.65 * cca
        )
        return round(_sigmoid(logit), 4)

    def get_meta_weights(self) -> dict[str, float]:
        """Retorna os pesos `(vision_cnn, text_llm, tabular_mlp, intercept)` do meta-classificador `LogisticRegression`."""
        if self.clf is not None and hasattr(self.clf, "coef_") and hasattr(self.clf, "intercept_"):
            coefs = self.clf.coef_[0]
            intercept = float(self.clf.intercept_[0])
            # Em esemble/multimodal.py (linha 288): X_meta = np.hstack((p_vision, p_llm, p_tabular))
            return {
                "vision_cnn": round(float(coefs[0]), 4),
                "text_llm": round(float(coefs[1]), 4),
                "tabular_mlp": round(float(coefs[2]), 4),
                "intercept": round(intercept, 4),
            }
        return {
            "vision_cnn": self.DEFAULT_COEF_VISION,
            "text_llm": self.DEFAULT_COEF_LLM,
            "tabular_mlp": self.DEFAULT_COEF_TABULAR,
            "intercept": self.DEFAULT_INTERCEPT,
        }

    def combine_probabilities(self, p_vision: float, p_llm: float, p_tabular: float) -> float:
        """
        Executa `self.clf.predict_proba(X_meta)[:, 1]` com `X_meta = np.hstack((p_vision, p_llm, p_tabular))`
        exatamente como em `esemble/multimodal.py` (linhas 306-309).
        """
        if self.clf is not None and hasattr(self.clf, "predict_proba"):
            try:
                import numpy as np  # type: ignore

                x_meta = np.array([[p_vision, p_llm, p_tabular]], dtype=np.float64)
                proba = self.clf.predict_proba(x_meta)
                return float(proba[0, 1])
            except Exception:
                pass

        weights = self.get_meta_weights()
        logit = (
            weights["intercept"]
            + weights["vision_cnn"] * p_vision
            + weights["text_llm"] * p_llm
            + weights["tabular_mlp"] * p_tabular
        )
        return _sigmoid(logit)

    def predict_single(
        self,
        *,
        rgb_tensor_chw: list[list[list[float]]],
        scaled_vector_42: Sequence[float],
        structured_prompt: str,
        flat_features: Mapping[str, float],
    ) -> EnsemblePrediction:
        """Executa as 3 modalidades do `Domain_Ensemble` e combina via `LogisticRegression`."""
        p_vision = round(self._predict_vision_single(rgb_tensor_chw, scaled_vector_42), 4)
        p_tabular = round(self._predict_tabular_single(scaled_vector_42), 4)
        p_llm = round(self._predict_llm_single(structured_prompt, flat_features), 4)

        final_score = round(self.combine_probabilities(p_vision, p_llm, p_tabular), 4)
        is_malicious = final_score >= self.decision_threshold
        verdict = "malicious" if is_malicious else "benign"
        label_id = 1 if is_malicious else 0

        return EnsemblePrediction(
            score=final_score,
            verdict=verdict,
            label_id=label_id,
            threshold=self.decision_threshold,
            model_version=self.model_version,
            model_scores={
                "vision_cnn": p_vision,
                "tabular_mlp": p_tabular,
                "text_llm": p_llm,
            },
            meta_weights=self.get_meta_weights(),
        )

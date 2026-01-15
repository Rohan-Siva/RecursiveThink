import os
import math
from dataclasses import dataclass
from typing import Optional, List
from abc import ABC, abstractmethod


@dataclass
class ModelResponse:
    text: str
    true_confidence: Optional[float] = None


class BaseModel(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> ModelResponse:
        pass
    
    def _calc_confidence_from_logprobs(self, logprobs: List[float]) -> float:
        if not logprobs:
            return None
        probs = [math.exp(lp) for lp in logprobs]
        avg_prob = sum(probs) / len(probs)
        return round(avg_prob, 4)


class MistralWrapper(BaseModel):
    
    def __init__(
        self,
        model_name: str = "mistral-large-latest",
        max_tokens: int = 1024,
        api_key: Optional[str] = None
    ):
        from mistralai import Mistral
        
        self.model_name = model_name
        self.max_tokens = max_tokens
        
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError(
                "MISTRAL_API_KEY not found. Set it in environment or pass api_key parameter."
            )
        
        print(f"Initializing Mistral client with model: {model_name}")
        self.client = Mistral(api_key=self.api_key)
        print("Mistral client ready.")
    
    def generate(self, system_prompt: str, user_prompt: str) -> ModelResponse:
        chat_response = self.client.chat.complete(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
            max_tokens=self.max_tokens,
            temperature=0.7,
        )
        
        text = chat_response.choices[0].message.content.strip()
        return ModelResponse(text=text, true_confidence=None)


class GeminiWrapper(BaseModel):
    
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None
    ):
        from google import genai
        
        self.model_name = model_name
        
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Set it in environment or pass api_key parameter."
            )
        
        print(f"Initializing Gemini client with model: {model_name}")
        self.client = genai.Client(api_key=self.api_key)
        print("Gemini client ready.")
    
    def generate(self, system_prompt: str, user_prompt: str) -> ModelResponse:
        from google.genai import types
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_logprobs=True,
                    logprobs=5
                ),
                contents=user_prompt
            )
        except Exception as e:
            if "Logprobs" in str(e) or "logprobs" in str(e):
                response = self.client.models.generate_content(
                    model=self.model_name,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt
                    ),
                    contents=user_prompt
                )
            else:
                raise
        
        text = response.text.strip()
        
        true_confidence = None
        try:
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'logprobs_result') and candidate.logprobs_result:
                    logprobs_result = candidate.logprobs_result
                    if hasattr(logprobs_result, 'chosen_candidates') and logprobs_result.chosen_candidates:
                        logprob_values = []
                        for token_info in logprobs_result.chosen_candidates:
                            if hasattr(token_info, 'log_probability'):
                                logprob_values.append(token_info.log_probability)
                        true_confidence = self._calc_confidence_from_logprobs(logprob_values)
        except Exception:
            pass
        
        return ModelResponse(text=text, true_confidence=true_confidence)


def create_model(provider: str, model_name: Optional[str] = None) -> BaseModel:
    if provider == "mistral":
        name = model_name or "mistral-large-latest"
        return MistralWrapper(model_name=name)
    elif provider == "gemini":
        name = model_name or "gemini-2.5-flash"
        return GeminiWrapper(model_name=name)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'mistral' or 'gemini'.")

import os
from typing import Optional
from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        pass


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
    
    def generate(self, system_prompt: str, user_prompt: str) -> str:
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
        
        return chat_response.choices[0].message.content.strip()


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
    
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        from google.genai import types
        
        response = self.client.models.generate_content(
            model=self.model_name,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt
            ),
            contents=user_prompt
        )
        
        return response.text.strip()


def create_model(provider: str, model_name: Optional[str] = None) -> BaseModel:
    if provider == "mistral":
        name = model_name or "mistral-large-latest"
        return MistralWrapper(model_name=name)
    elif provider == "gemini":
        name = model_name or "gemini-2.5-flash"
        return GeminiWrapper(model_name=name)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'mistral' or 'gemini'.")


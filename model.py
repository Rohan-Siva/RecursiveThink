import os
from typing import Optional
from mistralai import Mistral


class ModelWrapper:
    
    def __init__(
        self,
        model_name: str = "mistral-large-latest",
        max_tokens: int = 1024,
        api_key: Optional[str] = None
    ):
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
    
    def generate(self, prompt: str) -> str:
        chat_response = self.client.chat.complete(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=self.max_tokens,
            temperature=0.7,
        )
        
        return chat_response.choices[0].message.content.strip()

"""
model.py - LLM Wrapper using Mistral API
"""
import os
from typing import Optional
from mistralai import Mistral


class ModelWrapper:
    """
    Wrapper for Mistral API.
    Provides a simple generate() interface for the controller.
    """
    
    def __init__(
        self,
        model_name: str = "mistral-large-latest",
        max_tokens: int = 1024,
        api_key: Optional[str] = None
    ):
        """
        Initialize the model wrapper.
        
        Args:
            model_name: Mistral model identifier (default: mistral-large-latest)
            max_tokens: Maximum tokens to generate
            api_key: Mistral API key (defaults to MISTRAL_API_KEY env var)
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        
        # Get API key from env if not provided
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError(
                "MISTRAL_API_KEY not found. Set it in environment or pass api_key parameter."
            )
        
        print(f"Initializing Mistral client with model: {model_name}")
        self.client = Mistral(api_key=self.api_key)
        print("Mistral client ready.")
    
    def generate(self, prompt: str) -> str:
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: Input prompt string
            
        Returns:
            Generated text from Mistral API
        """
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

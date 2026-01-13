"""
model.py - LLM Wrapper using HuggingFace Transformers
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional


class ModelWrapper:
    """
    Wrapper for open-source LLMs using HuggingFace Transformers.
    Provides a simple generate() interface for the controller.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/phi-2",
        device: Optional[str] = None,
        max_new_tokens: int = 1024,
        load_in_8bit: bool = False
    ):
        """
        Initialize the model wrapper.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on (cuda/cpu/mps, auto-detect if None)
            max_new_tokens: Maximum tokens to generate
            load_in_8bit: Whether to use 8-bit quantization
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        print(f"Loading model {model_name} on {self.device}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
        }
        
        if load_in_8bit and self.device == "cuda":
            model_kwargs["load_in_8bit"] = True
        else:
            model_kwargs["device_map"] = self.device
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        self.model.eval()
        print(f"Model loaded successfully.")
    
    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: Input prompt string
            
        Returns:
            Generated text (model output only, not including prompt)
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode only the new tokens
        generated = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return generated.strip()

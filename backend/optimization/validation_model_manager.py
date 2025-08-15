#!/usr/bin/env python3
"""
Validation Model Manager with Automatic Quantization
Applies INT8 quantization to Teacher/Sentinel models while keeping main model at FP16
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging
from pathlib import Path

from backend.optimization.validation_quantization import (
    MixedPrecisionValidator,
    Int8Quantizer,
    QuantizationConfig
)

logger = logging.getLogger(__name__)

class ValidationModelManager:
    """
    Manages validation models (Teacher/Sentinel) with automatic quantization.
    Main AI model stays at FP16 for quality preservation.
    """
    
    def __init__(self, model_dir: Path = Path("./models")):
        self.model_dir = model_dir
        self.mixed_precision = MixedPrecisionValidator()
        
        # Model storage
        self.models: Dict[str, nn.Module] = {}
        self.model_types: Dict[str, str] = {}
        
        # Quantization stats
        self.quantization_stats: Dict[str, Dict[str, float]] = {}
        
    def load_model(
        self,
        model_name: str,
        model_type: str,  # 'main', 'teacher', 'sentinel', 'reward_model'
        apply_quantization: bool = True
    ) -> nn.Module:
        """
        Load a model with appropriate quantization based on type.
        """
        logger.info(f"Loading {model_type} model: {model_name}")
        
        # Load the base model
        model_path = self.model_dir / model_name
        
        try:
            # Try loading from local path first
            if model_path.exists():
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(str(model_path))
            else:
                # Load from HuggingFace
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(model_name)
                
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
        
        # Apply quantization based on model type
        if apply_quantization:
            original_size = self._get_model_size(model)
            
            if model_type == 'main':
                # Main model: Keep at FP16 for quality
                model = model.half()
                logger.info(f"Main model using FP16 precision (preserving quality)")
                quantized_size = self._get_model_size(model)
                
            elif model_type == 'teacher':
                # Teacher model: INT8 quantization for speed
                quantizer = Int8Quantizer(
                    config=QuantizationConfig(bits=8, method='dynamic')
                )
                model = quantizer.quantize_model(model)
                logger.info(f"Teacher model quantized to INT8 (4x memory reduction)")
                quantized_size = quantizer.quantized_size_mb
                
            elif model_type == 'sentinel':
                # Sentinel model: Aggressive INT8 quantization
                quantizer = Int8Quantizer(
                    config=QuantizationConfig(
                        bits=8,
                        method='dynamic',
                        per_channel=False,  # More aggressive
                        symmetric=False     # Even more aggressive
                    )
                )
                model = quantizer.quantize_model(model)
                logger.info(f"Sentinel model quantized to INT8 (aggressive settings)")
                quantized_size = quantizer.quantized_size_mb
                
            elif model_type == 'reward_model':
                # Reward model: FP16 for reasonable precision
                model = model.half()
                logger.info(f"Reward model using FP16 precision")
                quantized_size = self._get_model_size(model)
                
            else:
                logger.warning(f"Unknown model type: {model_type}, keeping original precision")
                quantized_size = original_size
            
            # Store quantization stats
            self.quantization_stats[model_name] = {
                'original_size_mb': original_size,
                'quantized_size_mb': quantized_size,
                'compression_ratio': original_size / quantized_size if quantized_size > 0 else 1.0,
                'model_type': model_type
            }
            
            logger.info(
                f"Model {model_name} ({model_type}): "
                f"{original_size:.1f}MB → {quantized_size:.1f}MB "
                f"(Compression: {original_size/quantized_size:.2f}x)"
            )
        
        # Store model
        self.models[model_name] = model
        self.model_types[model_name] = model_type
        
        return model
    
    def load_validation_suite(self) -> Dict[str, nn.Module]:
        """
        Load a complete validation suite with appropriate quantization.
        """
        suite = {}
        
        # Load main model (FP16)
        try:
            suite['main'] = self.load_model(
                'gpt_oss_20b',
                'main',
                apply_quantization=True
            )
        except Exception as e:
            logger.warning(f"Could not load main model: {e}")
        
        # Load teacher model (INT8)
        try:
            suite['teacher'] = self.load_model(
                'distilbert-base-uncased',
                'teacher',
                apply_quantization=True
            )
        except Exception as e:
            logger.warning(f"Could not load teacher model: {e}")
        
        # Load sentinel model (INT8 aggressive)
        try:
            suite['sentinel'] = self.load_model(
                'gpt2',
                'sentinel',
                apply_quantization=True
            )
        except Exception as e:
            logger.warning(f"Could not load sentinel model: {e}")
        
        return suite
    
    def validate_with_quantized_models(
        self,
        input_text: str,
        candidate_output: str
    ) -> Dict[str, Any]:
        """
        Run validation using quantized Teacher/Sentinel models.
        """
        results = {}
        
        # Teacher validation (INT8)
        if 'teacher' in self.models:
            teacher_model = self.models['teacher']
            try:
                # Run teacher validation
                teacher_score = self._run_validation(
                    teacher_model,
                    input_text,
                    candidate_output
                )
                results['teacher_score'] = teacher_score
                logger.info(f"Teacher validation (INT8): {teacher_score:.3f}")
            except Exception as e:
                logger.error(f"Teacher validation failed: {e}")
                results['teacher_score'] = None
        
        # Sentinel validation (INT8 aggressive)
        if 'sentinel' in self.models:
            sentinel_model = self.models['sentinel']
            try:
                # Run sentinel validation
                sentinel_score = self._run_validation(
                    sentinel_model,
                    input_text,
                    candidate_output
                )
                results['sentinel_score'] = sentinel_score
                logger.info(f"Sentinel validation (INT8): {sentinel_score:.3f}")
            except Exception as e:
                logger.error(f"Sentinel validation failed: {e}")
                results['sentinel_score'] = None
        
        # Combine scores
        if results.get('teacher_score') and results.get('sentinel_score'):
            results['combined_score'] = (
                0.7 * results['teacher_score'] +
                0.3 * results['sentinel_score']
            )
        
        return results
    
    def _run_validation(
        self,
        model: nn.Module,
        input_text: str,
        candidate_output: str
    ) -> float:
        """
        Run validation using a quantized model.
        """
        # Simple perplexity-based validation
        # In practice, this would be more sophisticated
        
        from transformers import AutoTokenizer
        
        # Get tokenizer
        model_name = None
        for name, stored_model in self.models.items():
            if stored_model is model:
                model_name = name
                break
        
        if model_name:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained('gpt2')
        
        # Tokenize
        inputs = tokenizer(
            input_text + candidate_output,
            return_tensors='pt',
            truncation=True,
            max_length=512
        )
        
        # Get model device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Calculate perplexity
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            perplexity = torch.exp(loss)
        
        # Convert to score (lower perplexity = higher score)
        score = 1.0 / (1.0 + perplexity.item())
        
        return score
    
    def _get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB."""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        return (param_size + buffer_size) / (1024 * 1024)
    
    def get_memory_savings(self) -> Dict[str, Any]:
        """
        Calculate total memory savings from quantization.
        """
        if not self.quantization_stats:
            return {"message": "No models loaded yet"}
        
        total_original = sum(
            stats['original_size_mb']
            for stats in self.quantization_stats.values()
        )
        
        total_quantized = sum(
            stats['quantized_size_mb']
            for stats in self.quantization_stats.values()
        )
        
        return {
            "total_original_mb": total_original,
            "total_quantized_mb": total_quantized,
            "total_saved_mb": total_original - total_quantized,
            "overall_compression": total_original / total_quantized if total_quantized > 0 else 1.0,
            "models": self.quantization_stats
        }

# Singleton instance
_validation_manager = None

def get_validation_manager() -> ValidationModelManager:
    """Get or create validation model manager."""
    global _validation_manager
    if _validation_manager is None:
        _validation_manager = ValidationModelManager()
    return _validation_manager

if __name__ == "__main__":
    # Test validation model manager
    manager = get_validation_manager()
    
    print("=" * 60)
    print("VALIDATION MODEL QUANTIZATION TEST")
    print("=" * 60)
    
    # Load validation suite
    print("\nLoading validation models with quantization...")
    suite = manager.load_validation_suite()
    
    # Show memory savings
    savings = manager.get_memory_savings()
    print(f"\nMemory Savings Report:")
    print(f"  Original total: {savings['total_original_mb']:.1f} MB")
    print(f"  Quantized total: {savings['total_quantized_mb']:.1f} MB")
    print(f"  Saved: {savings['total_saved_mb']:.1f} MB")
    print(f"  Compression: {savings['overall_compression']:.2f}x")
    
    # Test validation
    if suite:
        print("\nTesting quantized validation...")
        results = manager.validate_with_quantized_models(
            "What is machine learning?",
            "Machine learning is a subset of artificial intelligence."
        )
        
        print("\nValidation Results:")
        for key, value in results.items():
            if value is not None:
                print(f"  {key}: {value:.3f}")
#!/usr/bin/env python3
"""
Quantization System for Validation Models
4/8-bit quantization for Teacher/Sentinel models to reduce memory and increase speed
Main AI models remain at full precision for quality
"""

import torch
import torch.nn as nn
import torch.quantization as tq
from typing import Dict, Any, Optional, Tuple, List
import logging
import time
from dataclasses import dataclass
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    bits: int  # 4 or 8
    method: str  # 'dynamic', 'static', 'int8', 'int4'
    calibration_samples: int = 100
    symmetric: bool = True
    per_channel: bool = True
    backend: str = 'fbgemm'  # or 'qnnpack' for mobile

class Int8Quantizer:
    """
    INT8 quantization for validation models.
    Reduces memory by 4x, speeds up inference by 2-4x.
    """
    
    def __init__(self, config: QuantizationConfig = None):
        self.config = config or QuantizationConfig(bits=8, method='dynamic')
        
        # Performance metrics
        self.original_size_mb = 0
        self.quantized_size_mb = 0
        self.speedup_factor = 0
        
    def quantize_model(self, model: nn.Module, calibration_data: Optional[torch.Tensor] = None) -> nn.Module:
        """
        Quantize a model to INT8.
        """
        logger.info(f"Starting INT8 quantization with method: {self.config.method}")
        
        # Record original size
        self.original_size_mb = self._get_model_size(model)
        
        if self.config.method == 'dynamic':
            # Dynamic quantization (no calibration needed)
            quantized_model = self._dynamic_quantize(model)
        elif self.config.method == 'static':
            # Static quantization (requires calibration)
            if calibration_data is None:
                raise ValueError("Static quantization requires calibration data")
            quantized_model = self._static_quantize(model, calibration_data)
        else:
            raise ValueError(f"Unknown quantization method: {self.config.method}")
        
        # Record quantized size
        self.quantized_size_mb = self._get_model_size(quantized_model)
        
        # Calculate compression ratio
        compression_ratio = self.original_size_mb / self.quantized_size_mb
        logger.info(
            f"Quantization complete: {self.original_size_mb:.1f}MB → {self.quantized_size_mb:.1f}MB "
            f"(Compression: {compression_ratio:.2f}x)"
        )
        
        return quantized_model
    
    def _dynamic_quantize(self, model: nn.Module) -> nn.Module:
        """
        Apply dynamic quantization (weights only, activations computed in INT8).
        """
        try:
            # Specify which layers to quantize
            quantized_model = tq.quantize_dynamic(
                model,
                qconfig_spec={
                    nn.Linear: tq.default_dynamic_qconfig,
                    nn.LSTM: tq.default_dynamic_qconfig,
                    nn.GRU: tq.default_dynamic_qconfig,
                },
                dtype=torch.qint8
            )
            return quantized_model
        except RuntimeError as e:
            if "NoQEngine" in str(e) or "quantized" in str(e):
                logger.warning("Quantization not supported on this platform, using FP16 instead")
                # Fall back to FP16
                return model.half()
            raise
    
    def _static_quantize(self, model: nn.Module, calibration_data: torch.Tensor) -> nn.Module:
        """
        Apply static quantization (both weights and activations in INT8).
        """
        # Prepare model for quantization
        model.eval()
        
        # Fuse modules (Conv+BN, Linear+ReLU, etc.)
        model = self._fuse_modules(model)
        
        # Prepare for static quantization
        model.qconfig = tq.get_default_qconfig(self.config.backend)
        tq.prepare(model, inplace=True)
        
        # Calibrate with data
        with torch.no_grad():
            for i in range(min(self.config.calibration_samples, len(calibration_data))):
                _ = model(calibration_data[i:i+1])
        
        # Convert to quantized model
        quantized_model = tq.convert(model, inplace=False)
        
        return quantized_model
    
    def _fuse_modules(self, model: nn.Module) -> nn.Module:
        """
        Fuse modules for better quantization performance.
        """
        # This is model-specific, example for common patterns
        try:
            tq.fuse_modules(model, 
                [['conv', 'bn'], ['conv', 'bn', 'relu']], 
                inplace=True
            )
        except:
            # Skip fusion if modules don't exist
            pass
        return model
    
    def _get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB."""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def benchmark_inference(self, original_model: nn.Module, quantized_model: nn.Module, 
                           test_input: torch.Tensor, num_runs: int = 100) -> Dict[str, float]:
        """
        Benchmark inference speed of original vs quantized model.
        """
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = original_model(test_input)
                _ = quantized_model(test_input)
        
        # Benchmark original
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = original_model(test_input)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        original_time = time.time() - start
        
        # Benchmark quantized
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = quantized_model(test_input)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        quantized_time = time.time() - start
        
        self.speedup_factor = original_time / quantized_time
        
        return {
            "original_time_ms": (original_time / num_runs) * 1000,
            "quantized_time_ms": (quantized_time / num_runs) * 1000,
            "speedup": self.speedup_factor,
            "original_size_mb": self.original_size_mb,
            "quantized_size_mb": self.quantized_size_mb,
            "compression_ratio": self.original_size_mb / self.quantized_size_mb
        }

class Int4Quantizer:
    """
    INT4 quantization for extreme compression (experimental).
    Reduces memory by 8x but may impact accuracy more significantly.
    """
    
    def __init__(self):
        self.scale_factors = {}
        self.zero_points = {}
        
    def quantize_tensor_int4(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, float, int]:
        """
        Quantize a tensor to INT4 (stored in INT8 with packing).
        """
        # Calculate scale and zero point
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        
        # INT4 range: -8 to 7
        qmin = -8
        qmax = 7
        
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - min_val / scale
        
        # Quantize
        quantized = torch.round(tensor / scale + zero_point)
        quantized = torch.clamp(quantized, qmin, qmax).to(torch.int8)
        
        return quantized, scale, int(zero_point)
    
    def dequantize_tensor_int4(self, quantized: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
        """
        Dequantize INT4 tensor back to float.
        """
        return (quantized.float() - zero_point) * scale
    
    def pack_int4(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
        """
        Pack two INT4 tensors into one INT8 tensor.
        """
        # Ensure values are in INT4 range
        tensor1 = torch.clamp(tensor1, -8, 7)
        tensor2 = torch.clamp(tensor2, -8, 7)
        
        # Pack: high 4 bits from tensor1, low 4 bits from tensor2
        packed = ((tensor1 + 8) << 4) | (tensor2 + 8)
        
        return packed.to(torch.uint8)
    
    def unpack_int4(self, packed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unpack INT8 tensor into two INT4 tensors.
        """
        tensor1 = ((packed >> 4) & 0x0F) - 8
        tensor2 = (packed & 0x0F) - 8
        
        return tensor1.to(torch.int8), tensor2.to(torch.int8)

class MixedPrecisionValidator:
    """
    Mixed precision setup for validation models.
    Main model: FP16/BF16 for quality
    Teacher model: INT8 for speed
    Sentinel model: INT8/INT4 for extreme speed
    """
    
    def __init__(self):
        self.precision_map = {
            'main': torch.float16,      # or torch.bfloat16 on A100
            'teacher': torch.qint8,
            'sentinel': torch.qint8,     # or INT4 for extreme cases
            'reward_model': torch.float16
        }
        
    def setup_model_precision(self, model: nn.Module, model_type: str) -> nn.Module:
        """
        Set up appropriate precision for model type.
        """
        if model_type == 'main':
            # Keep main model in FP16 for quality
            model = model.half()
            logger.info("Main model using FP16 precision")
            
        elif model_type == 'teacher':
            # Quantize teacher to INT8
            quantizer = Int8Quantizer(QuantizationConfig(bits=8, method='dynamic'))
            model = quantizer.quantize_model(model)
            logger.info("Teacher model quantized to INT8")
            
        elif model_type == 'sentinel':
            # Aggressive quantization for sentinel
            quantizer = Int8Quantizer(QuantizationConfig(
                bits=8,
                method='dynamic',
                per_channel=False  # More aggressive
            ))
            model = quantizer.quantize_model(model)
            logger.info("Sentinel model quantized to INT8 (aggressive)")
            
        elif model_type == 'reward_model':
            # Reward model in FP16
            model = model.half()
            logger.info("Reward model using FP16 precision")
            
        else:
            logger.warning(f"Unknown model type: {model_type}, keeping original precision")
        
        return model
    
    def estimate_memory_savings(self, model_sizes: Dict[str, float]) -> Dict[str, Any]:
        """
        Estimate memory savings from quantization.
        """
        original_total = sum(model_sizes.values())
        
        # Estimate quantized sizes
        quantized_sizes = {
            'main': model_sizes.get('main', 0),  # No change
            'teacher': model_sizes.get('teacher', 0) / 4,  # INT8 is 4x smaller
            'sentinel': model_sizes.get('sentinel', 0) / 4,
            'reward_model': model_sizes.get('reward_model', 0) / 2  # FP16 is 2x smaller than FP32
        }
        
        quantized_total = sum(quantized_sizes.values())
        
        return {
            "original_total_gb": original_total / 1024,
            "quantized_total_gb": quantized_total / 1024,
            "memory_saved_gb": (original_total - quantized_total) / 1024,
            "compression_ratio": original_total / quantized_total if quantized_total > 0 else 0,
            "breakdown": {
                name: {
                    "original_mb": model_sizes.get(name, 0),
                    "quantized_mb": quantized_sizes.get(name, 0),
                    "saved_mb": model_sizes.get(name, 0) - quantized_sizes.get(name, 0)
                }
                for name in ['main', 'teacher', 'sentinel', 'reward_model']
            }
        }

class CalibrationDataGenerator:
    """
    Generate calibration data for static quantization.
    """
    
    def __init__(self, dataset_path: Optional[Path] = None):
        self.dataset_path = dataset_path or Path("./data/calibration")
        
    def generate_calibration_batch(
        self,
        batch_size: int = 32,
        seq_length: int = 512,
        vocab_size: int = 50000
    ) -> torch.Tensor:
        """
        Generate synthetic calibration data if real data unavailable.
        """
        # In production, load real validation data
        if self.dataset_path.exists():
            # Load from file
            return self._load_real_data(batch_size, seq_length)
        else:
            # Generate synthetic data (fallback)
            logger.warning("Using synthetic calibration data")
            return torch.randint(0, vocab_size, (batch_size, seq_length))
    
    def _load_real_data(self, batch_size: int, seq_length: int) -> torch.Tensor:
        """Load real calibration data from dataset."""
        # Implementation would load actual validation samples
        pass

# Singleton instances
_int8_quantizer = None
_int4_quantizer = None
_mixed_precision_validator = None

def get_int8_quantizer() -> Int8Quantizer:
    """Get or create INT8 quantizer."""
    global _int8_quantizer
    if _int8_quantizer is None:
        _int8_quantizer = Int8Quantizer()
    return _int8_quantizer

def get_int4_quantizer() -> Int4Quantizer:
    """Get or create INT4 quantizer."""
    global _int4_quantizer
    if _int4_quantizer is None:
        _int4_quantizer = Int4Quantizer()
    return _int4_quantizer

def get_mixed_precision_validator() -> MixedPrecisionValidator:
    """Get or create mixed precision validator."""
    global _mixed_precision_validator
    if _mixed_precision_validator is None:
        _mixed_precision_validator = MixedPrecisionValidator()
    return _mixed_precision_validator
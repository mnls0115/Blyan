"""
Adaptive Precision Strategy for Blyan
Layer-specific compression methods for optimal speed/performance balance
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import json

from .delta_compression import DeltaCompressor, INT8Delta, SparseDelta, LoRADelta, DeltaBase

class PrecisionMethod(Enum):
    """Available precision methods."""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"
    SPARSE = "sparse"
    LORA = "lora"
    DYNAMIC = "dynamic"

@dataclass
class LayerPrecisionProfile:
    """Precision profile for a specific layer type."""
    layer_type: str
    primary_method: PrecisionMethod
    fallback_method: PrecisionMethod
    compression_target: float  # Target compression ratio
    quality_threshold: float   # Minimum quality requirement
    speed_weight: float = 0.3  # How much to prioritize speed
    quality_weight: float = 0.7  # How much to prioritize quality
    
    # Constraints
    max_compression_ratio: float = 50.0
    min_precision_bits: int = 4
    allow_mixed_precision: bool = True

@dataclass
class CompressionResult:
    """Result of adaptive compression."""
    method_used: PrecisionMethod
    compressed_data: DeltaBase
    compression_ratio: float
    quality_loss: float
    processing_time: float
    metadata: Dict[str, Any]

class AdaptivePrecisionManager:
    """
    Manages adaptive precision strategies across different layer types.
    
    Automatically selects optimal compression methods based on:
    - Layer type (attention, FFN, LayerNorm, etc.)
    - Tensor characteristics (shape, distribution, etc.)
    - Performance requirements
    - Quality constraints
    """
    
    def __init__(self):
        # Default precision profiles for different layer types
        self.precision_profiles = self._create_default_profiles()
        
        # Compression managers for different methods
        self.compressors = {
            PrecisionMethod.INT8: DeltaCompressor(int8_enabled=True, sparse_enabled=False, lora_enabled=False),
            PrecisionMethod.SPARSE: DeltaCompressor(int8_enabled=False, sparse_enabled=True, lora_enabled=False, sparsity_threshold=0.9),
            PrecisionMethod.LORA: DeltaCompressor(int8_enabled=False, sparse_enabled=False, lora_enabled=True, lora_rank=16),
            PrecisionMethod.DYNAMIC: DeltaCompressor(int8_enabled=True, sparse_enabled=True, lora_enabled=True)
        }
        
        # Performance tracking
        self.compression_history: Dict[str, List[Dict[str, Any]]] = {}
        self.layer_performance: Dict[str, Dict[str, float]] = {}
        
        print("🎯 Adaptive Precision Manager initialized")
        print(f"   Profiles: {len(self.precision_profiles)} layer types")
        print(f"   Methods: {len(self.compressors)} compression types")
    
    def _create_default_profiles(self) -> Dict[str, LayerPrecisionProfile]:
        """Create default precision profiles for common layer types."""
        profiles = {}
        
        # Attention layers - prefer LoRA for weight matrices
        profiles['attention'] = LayerPrecisionProfile(
            layer_type='attention',
            primary_method=PrecisionMethod.LORA,
            fallback_method=PrecisionMethod.FP16,
            compression_target=5.0,
            quality_threshold=0.95,
            speed_weight=0.2,
            quality_weight=0.8
        )
        
        # FFN layers - can handle more aggressive compression
        profiles['ffn'] = LayerPrecisionProfile(
            layer_type='ffn',
            primary_method=PrecisionMethod.INT8,
            fallback_method=PrecisionMethod.SPARSE,
            compression_target=8.0,
            quality_threshold=0.90,
            speed_weight=0.4,
            quality_weight=0.6
        )
        
        # LayerNorm - keep high precision
        profiles['layernorm'] = LayerPrecisionProfile(
            layer_type='layernorm',
            primary_method=PrecisionMethod.FP16,
            fallback_method=PrecisionMethod.FP32,
            compression_target=2.0,
            quality_threshold=0.98,
            speed_weight=0.1,
            quality_weight=0.9
        )
        
        # Embedding layers - sparse compression works well
        profiles['embedding'] = LayerPrecisionProfile(
            layer_type='embedding',
            primary_method=PrecisionMethod.SPARSE,
            fallback_method=PrecisionMethod.INT8,
            compression_target=10.0,
            quality_threshold=0.92,
            speed_weight=0.3,
            quality_weight=0.7
        )
        
        # Output/Classification layers - maintain precision
        profiles['output'] = LayerPrecisionProfile(
            layer_type='output',
            primary_method=PrecisionMethod.FP16,
            fallback_method=PrecisionMethod.LoRA,
            compression_target=3.0,
            quality_threshold=0.96,
            speed_weight=0.2,
            quality_weight=0.8
        )
        
        # Default for unknown layer types
        profiles['default'] = LayerPrecisionProfile(
            layer_type='default',
            primary_method=PrecisionMethod.DYNAMIC,
            fallback_method=PrecisionMethod.FP16,
            compression_target=4.0,
            quality_threshold=0.93,
            speed_weight=0.3,
            quality_weight=0.7
        )
        
        return profiles
    
    def detect_layer_type(self, layer_name: str, tensor_shape: Tuple[int, ...]) -> str:
        """Automatically detect layer type from name and shape."""
        layer_name_lower = layer_name.lower()
        
        # Attention patterns
        if any(keyword in layer_name_lower for keyword in ['attention', 'attn', 'self_attn', 'q_proj', 'k_proj', 'v_proj', 'o_proj']):
            return 'attention'
        
        # FFN patterns  
        if any(keyword in layer_name_lower for keyword in ['ffn', 'mlp', 'feed_forward', 'fc', 'linear']):
            return 'ffn'
        
        # LayerNorm patterns
        if any(keyword in layer_name_lower for keyword in ['layernorm', 'layer_norm', 'ln', 'norm']):
            return 'layernorm'
        
        # Embedding patterns
        if any(keyword in layer_name_lower for keyword in ['embedding', 'embed', 'token', 'position']):
            return 'embedding'
        
        # Output patterns
        if any(keyword in layer_name_lower for keyword in ['output', 'classifier', 'head', 'lm_head']):
            return 'output'
        
        # Shape-based detection
        if len(tensor_shape) == 2:
            if tensor_shape[0] > tensor_shape[1] * 2:  # Tall matrix - might be embedding
                return 'embedding'
            elif tensor_shape[1] > tensor_shape[0] * 2:  # Wide matrix - might be FFN
                return 'ffn'
        
        return 'default'
    
    def analyze_tensor_characteristics(self, tensor: torch.Tensor) -> Dict[str, float]:
        """Analyze tensor characteristics to guide compression method selection."""
        characteristics = {}
        
        # Basic statistics
        characteristics['mean'] = tensor.mean().item()
        characteristics['std'] = tensor.std().item()
        characteristics['min'] = tensor.min().item()
        characteristics['max'] = tensor.max().item()
        
        # Distribution characteristics
        abs_tensor = tensor.abs()
        characteristics['sparsity_90'] = (abs_tensor < abs_tensor.quantile(0.9)).float().mean().item()
        characteristics['sparsity_95'] = (abs_tensor < abs_tensor.quantile(0.95)).float().mean().item()
        characteristics['sparsity_99'] = (abs_tensor < abs_tensor.quantile(0.99)).float().mean().item()
        
        # Rank characteristics (for LoRA suitability)
        if len(tensor.shape) == 2:
            try:
                U, S, V = torch.svd(tensor.float())
                total_variance = S.sum()
                
                # Calculate effective rank (captures 90% of variance)
                cumsum = S.cumsum(0)
                effective_rank_90 = ((cumsum / total_variance) < 0.9).sum().item()
                effective_rank_95 = ((cumsum / total_variance) < 0.95).sum().item()
                
                characteristics['effective_rank_90'] = effective_rank_90 / min(tensor.shape)
                characteristics['effective_rank_95'] = effective_rank_95 / min(tensor.shape)
                characteristics['rank_concentration'] = (S[:10].sum() / total_variance).item()
            except:
                # Fallback if SVD fails
                characteristics['effective_rank_90'] = 0.8
                characteristics['effective_rank_95'] = 0.9
                characteristics['rank_concentration'] = 0.5
        
        # Quantization friendliness
        dynamic_range = characteristics['max'] - characteristics['min']
        characteristics['dynamic_range'] = dynamic_range
        characteristics['quantization_error_estimate'] = dynamic_range / 256.0  # INT8 estimate
        
        return characteristics
    
    def select_optimal_method(self, 
                            layer_name: str,
                            tensor: torch.Tensor,
                            layer_type: Optional[str] = None) -> Tuple[PrecisionMethod, Dict[str, Any]]:
        """Select optimal compression method for a tensor."""
        
        # Detect layer type if not provided
        if layer_type is None:
            layer_type = self.detect_layer_type(layer_name, tensor.shape)
        
        # Get precision profile
        profile = self.precision_profiles.get(layer_type, self.precision_profiles['default'])
        
        # Analyze tensor characteristics
        characteristics = self.analyze_tensor_characteristics(tensor)
        
        # Method selection logic
        selected_method = profile.primary_method
        selection_reasoning = {'primary_choice': True, 'reasons': []}
        
        # Override based on tensor characteristics
        if selected_method == PrecisionMethod.DYNAMIC:
            # Dynamic selection based on characteristics
            sparsity = characteristics['sparsity_95']
            rank_concentration = characteristics.get('rank_concentration', 0.5)
            dynamic_range = characteristics['dynamic_range']
            
            if sparsity > 0.8:
                selected_method = PrecisionMethod.SPARSE
                selection_reasoning['reasons'].append(f'High sparsity: {sparsity:.2f}')
            elif rank_concentration > 0.7 and len(tensor.shape) == 2:
                selected_method = PrecisionMethod.LORA
                selection_reasoning['reasons'].append(f'High rank concentration: {rank_concentration:.2f}')
            elif dynamic_range < 10.0:
                selected_method = PrecisionMethod.INT8
                selection_reasoning['reasons'].append(f'Low dynamic range: {dynamic_range:.2f}')
            else:
                selected_method = PrecisionMethod.FP16
                selection_reasoning['reasons'].append('Default fallback')
        
        # Feasibility checks
        if selected_method == PrecisionMethod.LORA and len(tensor.shape) != 2:
            selected_method = profile.fallback_method
            selection_reasoning['primary_choice'] = False
            selection_reasoning['reasons'].append('LoRA requires 2D tensor')
        
        if selected_method == PrecisionMethod.SPARSE and characteristics['sparsity_95'] < 0.5:
            selected_method = profile.fallback_method
            selection_reasoning['primary_choice'] = False
            selection_reasoning['reasons'].append('Insufficient sparsity for sparse compression')
        
        selection_reasoning.update({
            'layer_type': layer_type,
            'tensor_shape': tensor.shape,
            'characteristics': characteristics,
            'profile': profile.layer_type
        })
        
        return selected_method, selection_reasoning
    
    def compress_with_adaptive_precision(self,
                                       layer_name: str,
                                       tensor: torch.Tensor,
                                       layer_type: Optional[str] = None) -> CompressionResult:
        """Compress tensor using adaptive precision strategy."""
        import time
        start_time = time.time()
        
        try:
            # Select optimal method
            method, reasoning = self.select_optimal_method(layer_name, tensor, layer_type)
            
            # Get appropriate compressor
            compressor = self.compressors.get(method, self.compressors[PrecisionMethod.DYNAMIC])
            
            # Perform compression
            if method == PrecisionMethod.FP16:
                # Simple FP16 conversion
                compressed_tensor = tensor.to(torch.float16)
                compressed_data = self._create_fp16_delta(compressed_tensor)
                compression_ratio = 2.0  # FP32 -> FP16
                quality_loss = 0.01  # Minimal loss for FP16
            
            elif method == PrecisionMethod.FP32:
                # No compression
                compressed_data = self._create_fp32_delta(tensor)
                compression_ratio = 1.0
                quality_loss = 0.0
            
            else:
                # Use delta compressor
                compressed_data = compressor.compress_gradient(tensor)
                compression_ratio = compressed_data.get_compression_ratio()
                quality_loss = self._estimate_quality_loss(tensor, compressed_data)
            
            processing_time = time.time() - start_time
            
            # Create result
            result = CompressionResult(
                method_used=method,
                compressed_data=compressed_data,
                compression_ratio=compression_ratio,
                quality_loss=quality_loss,
                processing_time=processing_time,
                metadata={
                    'layer_name': layer_name,
                    'layer_type': reasoning['layer_type'],
                    'selection_reasoning': reasoning,
                    'original_size': tensor.numel() * tensor.element_size(),
                    'compressed_size': len(compressed_data.to_bytes()) if hasattr(compressed_data, 'to_bytes') else 0
                }
            )
            
            # Record performance
            self._record_compression_performance(layer_name, result)
            
            return result
            
        except Exception as e:
            # Fallback to FP16 on error
            print(f"⚠️ Adaptive compression failed for {layer_name}: {e}")
            compressed_data = self._create_fp16_delta(tensor.to(torch.float16))
            
            return CompressionResult(
                method_used=PrecisionMethod.FP16,
                compressed_data=compressed_data,
                compression_ratio=2.0,
                quality_loss=0.01,
                processing_time=time.time() - start_time,
                metadata={'error': str(e), 'fallback': True}
            )
    
    def _create_fp16_delta(self, tensor: torch.Tensor) -> DeltaBase:
        """Create a simple FP16 delta (placeholder)."""
        # This is a simplified implementation
        # In practice, you'd create a proper FP16Delta class
        compressor = self.compressors[PrecisionMethod.DYNAMIC]
        return compressor.compress_gradient(tensor)
    
    def _create_fp32_delta(self, tensor: torch.Tensor) -> DeltaBase:
        """Create a FP32 delta (no compression)."""
        # This is a simplified implementation
        compressor = self.compressors[PrecisionMethod.DYNAMIC]
        return compressor.compress_gradient(tensor)
    
    def _estimate_quality_loss(self, original: torch.Tensor, compressed: DeltaBase) -> float:
        """Estimate quality loss from compression."""
        try:
            # Apply delta to zero tensor to get reconstructed tensor
            zero_tensor = torch.zeros_like(original)
            reconstructed = compressed.apply_to_tensor(zero_tensor)
            
            # Calculate relative error
            mse = torch.mean((original - reconstructed).pow(2))
            original_norm = torch.norm(original)
            
            if original_norm > 0:
                relative_error = mse.sqrt() / original_norm
                return relative_error.item()
            else:
                return 0.0
                
        except Exception:
            # Fallback estimate based on compression ratio
            compression_ratio = compressed.get_compression_ratio()
            return min(0.1, 1.0 / compression_ratio)
    
    def _record_compression_performance(self, layer_name: str, result: CompressionResult):
        """Record compression performance for analysis."""
        if layer_name not in self.compression_history:
            self.compression_history[layer_name] = []
        
        performance_record = {
            'timestamp': time.time(),
            'method': result.method_used.value,
            'compression_ratio': result.compression_ratio,
            'quality_loss': result.quality_loss,
            'processing_time': result.processing_time,
            'layer_type': result.metadata.get('layer_type', 'unknown')
        }
        
        self.compression_history[layer_name].append(performance_record)
        
        # Keep only recent history
        if len(self.compression_history[layer_name]) > 100:
            self.compression_history[layer_name].pop(0)
        
        # Update layer performance averages
        if layer_name not in self.layer_performance:
            self.layer_performance[layer_name] = {}
        
        recent_records = self.compression_history[layer_name][-10:]  # Last 10
        self.layer_performance[layer_name].update({
            'avg_compression_ratio': np.mean([r['compression_ratio'] for r in recent_records]),
            'avg_quality_loss': np.mean([r['quality_loss'] for r in recent_records]),
            'avg_processing_time': np.mean([r['processing_time'] for r in recent_records]),
            'primary_method': max(set(r['method'] for r in recent_records), 
                                key=[r['method'] for r in recent_records].count)
        })
    
    def get_layer_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            'summary': {
                'total_layers': len(self.layer_performance),
                'total_compressions': sum(len(history) for history in self.compression_history.values()),
            },
            'layer_performance': self.layer_performance.copy(),
            'method_usage': {},
            'precision_profiles': {k: v.__dict__ for k, v in self.precision_profiles.items()}
        }
        
        # Calculate method usage statistics
        all_records = []
        for history in self.compression_history.values():
            all_records.extend(history)
        
        if all_records:
            methods = [r['method'] for r in all_records]
            for method in set(methods):
                report['method_usage'][method] = {
                    'count': methods.count(method),
                    'percentage': methods.count(method) / len(methods) * 100
                }
        
        return report
    
    def optimize_precision_profiles(self):
        """Optimize precision profiles based on historical performance."""
        for layer_name, performance in self.layer_performance.items():
            # Detect layer type
            layer_type = 'default'
            for lt in self.precision_profiles.keys():
                if lt in layer_name.lower():
                    layer_type = lt
                    break
            
            profile = self.precision_profiles[layer_type]
            
            # Adjust based on performance
            avg_quality_loss = performance.get('avg_quality_loss', 0.0)
            avg_compression_ratio = performance.get('avg_compression_ratio', 1.0)
            
            # If quality loss is too high, be more conservative
            if avg_quality_loss > profile.quality_threshold:
                if profile.primary_method == PrecisionMethod.INT8:
                    profile.primary_method = PrecisionMethod.FP16
                elif profile.primary_method == PrecisionMethod.SPARSE:
                    profile.primary_method = PrecisionMethod.INT8
            
            # If compression ratio is too low, be more aggressive
            elif avg_compression_ratio < profile.compression_target:
                if profile.primary_method == PrecisionMethod.FP16:
                    profile.primary_method = PrecisionMethod.INT8
                elif profile.primary_method == PrecisionMethod.INT8:
                    profile.primary_method = PrecisionMethod.SPARSE
        
        print("🎯 Precision profiles optimized based on performance data")

# Export main classes
__all__ = ['AdaptivePrecisionManager', 'PrecisionMethod', 'LayerPrecisionProfile', 'CompressionResult']
"""
Blyan Evolution System - Evolutionary MoE Manager
Handles dynamic model reconstruction and evolution-aware inference
"""

from __future__ import annotations
import json
import time
import torch
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from contextlib import contextmanager

from backend.core.chain import Chain
from backend.core.param_index import ParameterIndex
from backend.model.moe_infer import ExpertUsageTracker
from .meta_v2 import MetaSpecV2, SemVer
from .migration import MigrationBlock, MigrationExecutor
from .loader import EvolutionAwareBlockLoader, load_block_payload
from .block import Block


class EvolutionaryMoEManager:
    """Enhanced MoE manager with evolutionary capabilities"""
    
    def __init__(
        self,
        meta_chain: Chain,
        param_chain: Chain,
        param_index: ParameterIndex,
        usage_tracker: ExpertUsageTracker,
        device: Optional[str] = None
    ):
        self.meta_chain = meta_chain
        self.param_chain = param_chain
        self.param_index = param_index
        self.usage_tracker = usage_tracker
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Evolution-specific components
        self.loader = EvolutionAwareBlockLoader()
        self.migration_executor = MigrationExecutor(self, param_chain)
        
        # Cache for loaded specs and models
        self._spec_cache: Dict[str, MetaSpecV2] = {}
        self._model_cache: Dict[str, Any] = {}
        self._expert_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        
        # Current model state
        self._current_spec: Optional[MetaSpecV2] = None
        self._current_version: Optional[str] = None
        self._evolution_history: List[str] = []
    
    # -------------------------------------------------------------------------
    # Core Evolution Methods
    # -------------------------------------------------------------------------
    
    def reconstruct_evolving_model(
        self, 
        target_version: Optional[str] = None,
        generation: Optional[int] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Reconstruct model from blockchain with evolutionary awareness"""
        
        # 1. Get target specification
        target_spec = self._get_target_spec(target_version, generation)
        if not target_spec:
            raise ValueError("Could not determine target specification")
        
        print(f"🧬 Reconstructing evolving model: {target_spec.model_id}@{target_spec.version}")
        
        # 2. Find compatible experts
        compatible_experts = self._find_compatible_experts(target_spec)
        print(f"🔍 Found {len(compatible_experts)} compatible experts")
        
        # 3. Build dynamic architecture
        model_architecture = self._build_dynamic_architecture(target_spec, compatible_experts)
        
        # 4. Prepare execution contexts (code blocks, etc.)
        execution_contexts = self._prepare_execution_contexts(target_spec, compatible_experts)
        
        # 5. Cache the reconstructed model
        self._model_cache[target_spec.version] = model_architecture
        self._current_spec = target_spec
        self._current_version = target_spec.version
        
        return model_architecture, execution_contexts
    
    def apply_migration(self, migration: MigrationBlock) -> MetaSpecV2:
        """Apply a migration to evolve the model"""
        if not self._current_spec:
            raise ValueError("No current specification loaded")
        
        print(f"🔄 Applying migration: {migration.from_version} → {migration.to_version}")
        
        # Validate migration compatibility
        if migration.from_version != self._current_spec.version:
            raise ValueError(f"Migration source version mismatch: {migration.from_version} != {self._current_spec.version}")
        
        # Apply migration
        new_spec = self.migration_executor.apply_migration(migration, self._current_spec)
        
        # Update evolution history
        self._evolution_history.append(self._current_spec.version)
        
        # Cache new spec
        self._spec_cache[new_spec.version] = new_spec
        
        print(f"✅ Migration completed: {new_spec.model_id}@{new_spec.version}")
        
        return new_spec
    
    def evolve_expert(
        self,
        expert_name: str,
        new_weights: Dict[str, torch.Tensor],
        evolution_type: str = "expansion",
        target_version: Optional[str] = None
    ) -> str:
        """Evolve a specific expert and create new block"""
        
        if not self._current_spec:
            raise ValueError("No current specification loaded")
        
        # Determine target version
        if not target_version:
            current_semver = SemVer(self._current_spec.version)
            target_version = str(current_semver.bump_patch())
        
        print(f"🧬 Evolving expert {expert_name}: {evolution_type} → {target_version}")
        
        # Calculate dimension changes
        dimension_changes = self._calculate_dimension_changes(expert_name, new_weights)
        
        # Create evolved expert block
        block_hash = self._create_evolved_expert_block(
            expert_name=expert_name,
            weights=new_weights,
            evolution_type=evolution_type,
            target_version=target_version,
            dimension_changes=dimension_changes
        )
        
        # Update expert cache
        self._expert_cache[f"{expert_name}@{target_version}"] = new_weights
        
        print(f"✅ Expert evolution completed: {expert_name}@{target_version} [{block_hash[:8]}]")
        
        return block_hash
    
    # -------------------------------------------------------------------------
    # Model Reconstruction
    # -------------------------------------------------------------------------
    
    def _get_target_spec(self, version: Optional[str], generation: Optional[int]) -> Optional[MetaSpecV2]:
        """Get target specification by version or generation"""
        
        if version:
            # Try cache first
            if version in self._spec_cache:
                return self._spec_cache[version]
            
            # Load from blockchain
            spec = self._load_spec_by_version(version)
            if spec:
                self._spec_cache[version] = spec
            return spec
        
        elif generation is not None:
            # Find spec by generation
            return self._load_spec_by_generation(generation)
        
        else:
            # Get latest spec
            return self._get_latest_spec()
    
    def _find_compatible_experts(self, spec: MetaSpecV2) -> List[Block]:
        """Find expert blocks compatible with target specification"""
        all_expert_blocks = self.param_chain.get_blocks_by_type('expert')
        compatible = []
        
        for block in all_expert_blocks:
            if self._is_expert_compatible(block, spec):
                compatible.append(block)
        
        return compatible
    
    def _is_expert_compatible(self, expert_block: Block, spec: MetaSpecV2) -> bool:
        """Check if expert block is compatible with specification"""
        
        # Check version compatibility
        if expert_block.header.version:
            if not spec.is_compatible_with(expert_block.header.version):
                return False
        
        # Check layer compatibility
        if expert_block.header.layer_id:
            layer_num = int(expert_block.header.layer_id.replace('layer', ''))
            if layer_num >= spec.base_num_layers:
                return False
        
        # Check expert name pattern
        if expert_block.header.expert_name:
            # Ensure expert fits within current expert count limits
            layer_key = expert_block.header.layer_id
            if layer_key and layer_key in spec.dynamic_experts:
                config = spec.dynamic_experts[layer_key]
                expert_num = self._extract_expert_number(expert_block.header.expert_name)
                if expert_num and expert_num >= config.current_experts:
                    return False
        
        return True
    
    def _build_dynamic_architecture(self, spec: MetaSpecV2, expert_blocks: List[Block]) -> Dict[str, Any]:
        """Build model architecture dynamically based on spec and available experts"""
        
        architecture = {
            "model_id": spec.model_id,
            "version": spec.version,
            "architecture_type": spec.architecture,
            "num_layers": spec.base_num_layers,
            "experts_by_layer": {},
            "routers": {},
            "dynamic_config": spec.dynamic_experts,
            "runtime_requirements": spec.runtime_req
        }
        
        # Group experts by layer
        experts_by_layer = {}
        for block in expert_blocks:
            layer_id = block.header.layer_id
            if layer_id:
                if layer_id not in experts_by_layer:
                    experts_by_layer[layer_id] = []
                experts_by_layer[layer_id].append(block)
        
        # Load expert weights for each layer
        for layer_id, expert_blocks_for_layer in experts_by_layer.items():
            layer_experts = {}
            
            for expert_block in expert_blocks_for_layer:
                expert_name = expert_block.header.expert_name
                if expert_name:
                    try:
                        expert_weights = self.loader.load_block(expert_block)
                        layer_experts[expert_name] = {
                            "weights": expert_weights,
                            "block_hash": expert_block.compute_hash(),
                            "version": expert_block.header.version,
                            "dimension_changes": expert_block.header.dimension_changes
                        }
                    except Exception as e:
                        print(f"⚠️ Failed to load expert {expert_name}: {e}")
            
            architecture["experts_by_layer"][layer_id] = layer_experts
        
        return architecture
    
    def _prepare_execution_contexts(self, spec: MetaSpecV2, expert_blocks: List[Block]) -> Dict[str, Any]:
        """Prepare execution contexts including code blocks"""
        
        contexts = {
            "runtime_environment": spec.runtime_req,
            "code_blocks": {},
            "router_configs": {},
            "execution_plan": self._create_execution_plan(spec, expert_blocks)
        }
        
        # Load code blocks
        code_blocks = self.param_chain.get_blocks_by_type('code')
        for code_block in code_blocks:
            if code_block.header.is_compatible_with_version(spec.version):
                try:
                    code_data = self.loader.load_block(code_block)
                    target_expert = code_block.header.target_expert
                    if target_expert:
                        contexts["code_blocks"][target_expert] = code_data
                except Exception as e:
                    print(f"⚠️ Failed to load code block for {code_block.header.target_expert}: {e}")
        
        return contexts
    
    def _create_execution_plan(self, spec: MetaSpecV2, expert_blocks: List[Block]) -> Dict[str, Any]:
        """Create execution plan for the evolved model"""
        
        plan = {
            "type": "evolutionary_moe",
            "version": spec.version,
            "layer_sequence": [],
            "expert_routing": {},
            "fallback_strategy": "previous_version"
        }
        
        # Create layer execution sequence
        for layer_idx in range(spec.base_num_layers):
            layer_id = f"layer{layer_idx}"
            layer_config = spec.dynamic_experts.get(layer_id)
            
            if layer_config:
                plan["layer_sequence"].append({
                    "layer_id": layer_id,
                    "num_experts": layer_config.current_experts,
                    "routing_strategy": "top_k",
                    "top_k": min(2, layer_config.current_experts)
                })
        
        return plan
    
    # -------------------------------------------------------------------------
    # Expert Evolution
    # -------------------------------------------------------------------------
    
    def _calculate_dimension_changes(
        self, 
        expert_name: str, 
        new_weights: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Calculate dimension changes for expert evolution"""
        
        changes = {}
        
        # Try to get previous expert weights for comparison
        existing_experts = self.param_chain.get_expert_blocks(expert_name)
        if existing_experts:
            try:
                latest_expert = max(existing_experts, key=lambda b: b.header.timestamp)
                old_weights = self.loader.load_block(latest_expert)
                
                # Compare dimensions
                for param_name, new_tensor in new_weights.items():
                    if param_name in old_weights:
                        old_shape = old_weights[param_name].shape
                        new_shape = new_tensor.shape
                        
                        if old_shape != new_shape:
                            changes[param_name] = {
                                "from": list(old_shape),
                                "to": list(new_shape),
                                "expansion_ratio": new_tensor.numel() / old_weights[param_name].numel()
                            }
            except Exception as e:
                print(f"⚠️ Could not compare with previous expert weights: {e}")
        
        return changes
    
    def _create_evolved_expert_block(
        self,
        expert_name: str,
        weights: Dict[str, torch.Tensor],
        evolution_type: str,
        target_version: str,
        dimension_changes: Dict[str, Any]
    ) -> str:
        """Create a new expert block for evolved weights"""
        
        # Serialize weights
        payload = self._serialize_expert_weights(weights)
        
        # Get parent block hash
        parent_hash = None
        existing_experts = self.param_chain.get_expert_blocks(expert_name)
        if existing_experts:
            latest_expert = max(existing_experts, key=lambda b: b.header.timestamp)
            parent_hash = latest_expert.compute_hash()
        
        # Extract layer ID from expert name
        layer_id = expert_name.split('.')[0] if '.' in expert_name else "layer0"
        
        # Create evolved expert block
        block = self.param_chain.add_block(
            payload=payload,
            block_type='expert',
            expert_name=expert_name,
            layer_id=layer_id,
            # Evolution fields
            version=target_version,
            parent_hash=parent_hash,
            evolution_type=evolution_type,
            dimension_changes=dimension_changes,
            compatibility_range=[target_version],
            evolution_metadata={
                "evolution_timestamp": time.time(),
                "evolution_trigger": "manual",
                "performance_delta": None  # To be filled by PoL validation
            }
        )
        
        return block.compute_hash()
    
    def _serialize_expert_weights(self, weights: Dict[str, torch.Tensor]) -> bytes:
        """Serialize expert weights for storage"""
        # TODO: Use TensorBlock format for zero-copy loading
        # For now, use pickle
        import pickle
        return pickle.dumps(weights)
    
    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    
    def _extract_expert_number(self, expert_name: str) -> Optional[int]:
        """Extract expert number from expert name (e.g., 'layer0.expert2' -> 2)"""
        try:
            if '.' in expert_name and 'expert' in expert_name:
                expert_part = expert_name.split('.')[-1]  # Get 'expert2' part
                if expert_part.startswith('expert'):
                    return int(expert_part[6:])  # Remove 'expert' prefix
        except (ValueError, IndexError):
            pass
        return None
    
    def _load_spec_by_version(self, version: str) -> Optional[MetaSpecV2]:
        """Load specification by version from blockchain"""
        # TODO: Implement version-specific meta block retrieval
        meta_blocks = self.meta_chain.get_blocks_by_type('meta')
        
        for block in meta_blocks:
            if block.header.version == version:
                try:
                    spec_data = self.loader.load_block(block)
                    if isinstance(spec_data, MetaSpecV2):
                        return spec_data
                    elif isinstance(spec_data, dict):
                        return MetaSpecV2.from_dict(spec_data)
                except Exception as e:
                    print(f"⚠️ Failed to load spec from block: {e}")
        
        return None
    
    def _load_spec_by_generation(self, generation: int) -> Optional[MetaSpecV2]:
        """Load specification by generation number"""
        # TODO: Implement generation-based retrieval
        meta_blocks = self.meta_chain.get_blocks_by_type('meta')
        
        for block in meta_blocks:
            try:
                spec_data = self.loader.load_block(block)
                if isinstance(spec_data, dict) and spec_data.get('generation') == generation:
                    return MetaSpecV2.from_dict(spec_data)
                elif hasattr(spec_data, 'generation') and spec_data.generation == generation:
                    return spec_data
            except Exception:
                continue
        
        return None
    
    def _get_latest_spec(self) -> Optional[MetaSpecV2]:
        """Get the latest meta specification"""
        latest_block = self.meta_chain.storage.get_latest_block()
        if latest_block and latest_block.header.block_type == 'meta':
            try:
                spec_data = self.loader.load_block(latest_block)
                if isinstance(spec_data, MetaSpecV2):
                    return spec_data
                elif isinstance(spec_data, dict):
                    return MetaSpecV2.from_dict(spec_data)
            except Exception as e:
                print(f"⚠️ Failed to load latest spec: {e}")
        
        return None
    
    # -------------------------------------------------------------------------
    # Inference Integration
    # -------------------------------------------------------------------------
    
    def evolutionary_generate(
        self,
        prompt: str,
        target_version: Optional[str] = None,
        max_new_tokens: int = 64,
        top_k_experts: int = 2,
        stability_mode: str = "stable"  # "stable", "latest", "pinned"
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate text using evolutionary model reconstruction"""
        
        start_time = time.time()
        
        # Determine target version based on stability mode
        if stability_mode == "latest":
            target_version = None  # Use latest
        elif stability_mode == "pinned" and target_version:
            pass  # Use specified version
        else:  # "stable" mode
            target_version = self._get_stable_version()
        
        # Reconstruct model for target version
        model_arch, contexts = self.reconstruct_evolving_model(target_version)
        
        # Extract selected experts and routing plan
        selected_experts = []
        expert_usage = {}
        
        execution_plan = contexts.get("execution_plan", {})
        
        # Simulate expert selection based on evolved architecture
        for layer_info in execution_plan.get("layer_sequence", []):
            layer_id = layer_info["layer_id"]
            layer_experts = model_arch["experts_by_layer"].get(layer_id, {})
            
            # Select top-k experts for this layer
            available_experts = list(layer_experts.keys())[:top_k_experts]
            selected_experts.extend(available_experts)
            
            # Record usage
            for expert_name in available_experts:
                expert_usage[expert_name] = time.time() - start_time
        
        # Generate response using evolved model
        response = self._generate_with_evolved_model(
            prompt, model_arch, contexts, selected_experts, max_new_tokens
        )
        
        # Update usage tracking
        for expert_name, usage_time in expert_usage.items():
            self.usage_tracker.record_usage(
                expert_name=expert_name,
                response_time=usage_time,
                quality_score=0.85  # TODO: Implement actual quality assessment
            )
        
        total_time = time.time() - start_time
        
        return response, {
            "model_version": model_arch["version"],
            "selected_experts": selected_experts,
            "expert_usage": expert_usage,
            "total_time": total_time,
            "evolution_info": {
                "generation": getattr(self._current_spec, 'generation', 0),
                "total_experts": sum(len(experts) for experts in model_arch["experts_by_layer"].values()),
                "architecture_type": model_arch["architecture_type"]
            }
        }
    
    def _get_stable_version(self) -> Optional[str]:
        """Get the latest stable version (highest minor in current major)"""
        # TODO: Implement stable version detection logic
        if self._current_spec:
            return self._current_spec.version
        return None
    
    def _generate_with_evolved_model(
        self,
        prompt: str,
        model_arch: Dict[str, Any],
        contexts: Dict[str, Any],
        selected_experts: List[str],
        max_new_tokens: int
    ) -> str:
        """Generate text using the evolved model architecture"""
        
        model_version = model_arch["version"]
        expert_count = len(selected_experts)
        
        # Enhanced response based on evolution
        evolution_info = f"Evolution Gen-{getattr(self._current_spec, 'generation', 0)}"
        expert_summary = f"{expert_count} evolved experts"
        
        responses = [
            f"🧬 {evolution_info}: {prompt}에 대해 {expert_summary}가 협력하여 응답합니다.",
            f"Evolutionary Blyan v{model_version} processed your request using {expert_count} evolved experts.",
            f"진화한 {expert_summary}를 통해 더욱 향상된 응답을 제공합니다: {prompt}"
        ]
        
        import random
        base_response = random.choice(responses)
        
        # Add evolution-specific enhancements
        if self._current_spec and self._current_spec.generation > 0:
            enhancement = f" [진화 세대 {self._current_spec.generation}: {model_arch['architecture_type']} 아키텍처]"
            base_response += enhancement
        
        return base_response
    
    # -------------------------------------------------------------------------
    # Context Managers
    # -------------------------------------------------------------------------
    
    @contextmanager
    def evolution_context(self, target_version: str):
        """Context manager for temporary evolution state"""
        original_spec = self._current_spec
        original_version = self._current_version
        
        try:
            # Switch to target version
            model_arch, contexts = self.reconstruct_evolving_model(target_version)
            yield model_arch, contexts
        finally:
            # Restore original state
            self._current_spec = original_spec
            self._current_version = original_version
    
    # -------------------------------------------------------------------------
    # Status and Monitoring
    # -------------------------------------------------------------------------
    
    def get_evolution_status(self) -> Dict[str, Any]:
        """Get current evolution status"""
        status = {
            "current_version": self._current_version,
            "current_generation": getattr(self._current_spec, 'generation', 0) if self._current_spec else 0,
            "evolution_history": self._evolution_history,
            "cached_specs": list(self._spec_cache.keys()),
            "cached_models": list(self._model_cache.keys()),
            "total_experts": 0,
            "experts_by_layer": {}
        }
        
        if self._current_spec:
            status.update({
                "model_id": self._current_spec.model_id,
                "architecture": self._current_spec.architecture,
                "base_layers": self._current_spec.base_num_layers,
                "expandable_layers": self._current_spec.expandable_layers,
                "runtime_requirements": self._current_spec.runtime_req.to_dict() if hasattr(self._current_spec.runtime_req, 'to_dict') else str(self._current_spec.runtime_req),
                "pol_threshold": self._current_spec.pol_threshold
            })
            
            # Count experts per layer
            total_experts = 0
            for layer_id, config in self._current_spec.dynamic_experts.items():
                status["experts_by_layer"][layer_id] = config.current_experts
                total_experts += config.current_experts
            
            status["total_experts"] = total_experts
        
        return status
    
    def get_compatibility_matrix(self) -> Dict[str, List[str]]:
        """Get version compatibility matrix"""
        matrix = {}
        
        for version, spec in self._spec_cache.items():
            compatible_versions = []
            for other_version in self._spec_cache.keys():
                if spec.is_compatible_with(other_version):
                    compatible_versions.append(other_version)
            matrix[version] = compatible_versions
        
        return matrix
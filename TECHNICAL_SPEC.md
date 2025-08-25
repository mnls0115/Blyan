# Technical Specification
# Blyan Network - Distributed MoE Blockchain System

## 1. System Overview

### 1.1 Architecture Summary
Blyan Network implements a distributed dense model AI system (Qwen3-8B) using a dual-chain blockchain architecture with DAG structure for parallel layer evolution. The system uses Proof-of-Learning (PoL) validation for verifying model improvements through loss comparison and fraud detection.

### 1.2 Core Components
- **Blockchain Layer**: Dual-chain architecture (Meta chain A + Parameter chain B)
- **AI Layer**: Qwen3-8B dense model with 32 transformer layers
- **P2P Network**: Distributed inference with pipeline parallelism
- **Storage Layer**: Hybrid on-chain weights with zero-copy GPU loading
- **Economic Layer**: BLY token with automatic hourly reward distribution

### 1.3 Technology Stack
- **Language**: Python 3.10+
- **Framework**: FastAPI, PyTorch 2.0+
- **Blockchain**: Custom DAG implementation
- **Database**: PostgreSQL 14+, Redis 7+
- **Infrastructure**: Docker, Kubernetes
- **Monitoring**: Prometheus, Grafana

## 2. Blockchain Architecture

### 2.1 Two-Chain Design

#### Transaction Chain (PoS+BFT)
```python
class TransactionChain:
    consensus: str = "Tendermint BFT"
    finality: str = "<2 seconds"
    validators: int = 100-300
    tolerance: str = "<1/3 Byzantine"
    
    features:
        - Token transfers with nonce protection
        - EIP-1559 fee mechanism
        - Deterministic finality
        - Slashing conditions
```

#### PoL Chain (Proof-of-Learning)
```python
class PoLChain:
    purpose: str = "AI model improvements"
    validation: str = "Loss comparison on validation datasets"
    fraud_detection: str = "Statistical outlier analysis"
    rewards: str = "Automatic BLY distribution"
    
    features:
        - Layer weight storage (append-only)
        - Delta validation with fraud detection
        - Confidence scoring based on improvement
        - Hourly reward distribution cycles
```

### 2.2 DAG Structure

#### Block Types
```python
class BlockType(Enum):
    META = "meta"        # Model architecture
    LAYER = "layer"      # Layer weights (dense model)
    DELTA = "delta"      # Weight updates
    DATA = "dataset"     # Training data metadata
    MIGRATION = "migration"  # Layer dimension changes
```

#### DAG Properties
- **Append-Only**: New blocks are always appended, never replaced
- **Dependency Tracking**: `depends_on` field for parent relationships
- **Cycle Prevention**: Topological sorting validation
- **No Merging**: Parallel branches exist independently without intelligent merging

### 2.3 Consensus Mechanisms

#### PoS Validator Selection
```python
def select_validators(stake_pool: Dict[str, int]) -> List[str]:
    """
    VRF-based validator selection weighted by stake
    """
    total_stake = sum(stake_pool.values())
    selection_threshold = 2/3 * total_stake
    
    validators = []
    accumulated_stake = 0
    
    for validator in sorted_by_vrf(stake_pool):
        validators.append(validator)
        accumulated_stake += stake_pool[validator]
        if accumulated_stake >= selection_threshold:
            break
    
    return validators
```

#### PoL Quality Validation
```python
def validate_improvement(
    delta: TensorDict,
    original_weights: Dict,
    validation_set: Dataset
) -> PoLScore:
    """
    Comprehensive delta evaluation with fraud detection
    """
    # Apply delta and evaluate
    candidate_weights = apply_delta(original_weights, delta)
    loss_before = evaluate_weights(original_weights, validation_set)
    loss_after = evaluate_weights(candidate_weights, validation_set)
    
    # Calculate improvement and detect fraud
    improvement_score = (loss_before - loss_after) / loss_before
    fraud_probability = detect_fraud_patterns(delta, improvement_score)
    confidence = calculate_confidence(improvement_score, fraud_probability)
    
    return PoLScore(
        is_valid=(improvement_score >= 0.005 and fraud_probability < 0.7),
        improvement_score=improvement_score,
        fraud_probability=fraud_probability,
        confidence_score=confidence
    )
```

## 3. AI Model Architecture

### 3.1 Qwen3-8B Dense Model Specifications
```yaml
model:
  name: Qwen/Qwen3-8B
  total_parameters: 8B
  architecture: dense_transformer
  
  dimensions:
    layers: 32
    hidden_size: 4096
    intermediate_size: 11008
    num_attention_heads: 32
    num_key_value_heads: 32
    vocab_size: 151936
    
  quantization:
    precision: FP8
    memory_required: ~8GB (FP8)
    optimization: NF4/MXFP4 for 16GB GPUs
    
  pipeline_parallelism:
    strategy: layer_partitioning
    stages: auto_balanced
    communication: gRPC/HTTP
```

### 3.2 Layer Management & Pipeline Parallelism

#### Layer Partitioning Strategy
```python
class LayerPartitioner:
    def partition_model(self, num_gpus: int) -> List[LayerGroup]:
        """Partition dense model layers across GPUs"""
        total_layers = 32  # Qwen3-8B has 32 layers
        layers_per_gpu = total_layers // num_gpus
        
        partitions = []
        for gpu_id in range(num_gpus):
            start_layer = gpu_id * layers_per_gpu
            end_layer = start_layer + layers_per_gpu
            if gpu_id == num_gpus - 1:
                end_layer = total_layers  # Last GPU gets remaining layers
            
            partitions.append({
                "gpu_id": gpu_id,
                "layers": list(range(start_layer, end_layer)),
                "memory_required": self.calculate_memory(end_layer - start_layer)
            })
        
        return partitions
```

#### Delta Evolution
```python
class DeltaEvolution:
    def evolve_layer(
        self,
        layer_id: int,
        training_data: Dataset,
        current_version: str
    ) -> Dict:
        """Layer-wise improvement with deltas"""
        # 1. Load current layer weights
        layer_weights = self.blockchain.get_layer(layer_id, current_version)
        
        # 2. Train with LoRA/QLoRA for efficiency
        delta = train_with_lora(layer_weights, training_data)
        
        # 3. Compress delta (INT8 + sparse)
        compressed_delta = compress_delta(delta)
        
        # 4. Validate improvement
        if validate_improvement(compressed_delta):
            # 5. Submit to blockchain
            new_version = increment_version(current_version)
            block_hash = self.blockchain.add_delta_block(
                layer_id=layer_id,
                version=new_version,
                delta=compressed_delta,
                depends_on=current_version
            )
            return {"success": True, "block": block_hash, "size": len(compressed_delta)}
        
        return {"success": False, "reason": "No improvement"}
```

### 3.3 Distributed Inference

#### Node Registration
```python
class GPUNode:
    node_id: str
    host: str
    port: int
    available_experts: List[str]
    capabilities: Dict[str, Any]
    
    performance_metrics:
        latency_ms: float
        throughput_tps: float
        success_rate: float
        uptime_hours: float
```

#### Load Balancing
```python
class LoadBalancer:
    def route_request(self, required_experts: List[str]) -> GPUNode:
        """Route to optimal node based on expert availability"""
        candidates = []
        
        for node in self.active_nodes:
            if all(exp in node.available_experts for exp in required_experts):
                score = self.calculate_node_score(node)
                candidates.append((node, score))
        
        # Select best node considering latency, load, and reliability
        best_node = max(candidates, key=lambda x: x[1])
        return best_node[0]
```

## 4. Storage Architecture

### 4.1 Hybrid Storage Model

#### On-Chain Storage
- **Meta Blocks**: Model architecture (1KB)
- **Block Headers**: Hashes and metadata (100B)
- **Merkle Roots**: Aggregated proofs (32B)
- **State Transitions**: Consensus records (500B)

#### Off-Chain Storage
- **Expert Weights**: IPFS with on-chain CID (52MB each)
- **Training Data**: S3/IPFS with validation hashes
- **Inference Logs**: PostgreSQL with retention policy
- **Node Registry**: Redis with TTL expiration

### 4.2 TensorBlock System

#### Zero-Copy Implementation
```python
class ZeroCopyTileLoader:
    """Zero-copy loading from blockchain to GPU memory"""
    
    def load_tile(self, tile_hash: str, device: str = 'cuda') -> torch.Tensor:
        """
        Load tensor with minimal copying:
        - Memory-map blockchain file (no copy)
        - Create tensor view from mmap (no copy)
        - Pin memory for GPU transfer
        - Transfer to GPU (single copy)
        
        Traditional: blockchain → deserialize → tensor → GPU (3 copies)
        Zero-copy: mmap → torch.frombuffer → pin_memory → GPU (1 copy)
        """
        mmap_view = self._get_mmap_view(tile_hash)
        tensor_view = torch.from_numpy(
            np.frombuffer(mmap_view, dtype=np.float16)
        )
        if device != 'cpu':
            tensor_view = tensor_view.pin_memory()
            return tensor_view.to(device, non_blocking=True)
        return tensor_view
```

## 5. Network Protocol

### 5.1 P2P Communication

#### Message Types
```protobuf
enum MessageType {
    INFERENCE_REQUEST = 0;
    INFERENCE_RESPONSE = 1;
    EXPERT_TRANSFER = 2;
    HEARTBEAT = 3;
    VALIDATION_REQUEST = 4;
    DELTA_SUBMISSION = 5;
}

message P2PMessage {
    MessageType type = 1;
    bytes payload = 2;
    string sender_id = 3;
    int64 timestamp = 4;
    bytes signature = 5;
}
```

#### Transport Layer
- **Protocol**: gRPC over HTTP/2
- **Encryption**: TLS 1.3 mandatory
- **Authentication**: mTLS for node-to-node
- **Compression**: gzip for large payloads
- **Chunking**: 1MB chunks for expert weights

### 5.2 API Specifications

#### Core Endpoints
```yaml
endpoints:
  # Inference
  POST /chat:
    request:
      prompt: string
      max_tokens: int
      temperature: float
      use_moe: bool
    response:
      text: string
      tokens_used: int
      experts_used: List[str]
      cost_bly: float
  
  # Mining
  POST /mine:
    request:
      address: string
      block_data: bytes
      nonce: int
      candidate_loss: float
    response:
      block_hash: string
      reward_bly: float
  
  # Node Management
  POST /p2p/register:
    request:
      node_id: string
      host: string
      port: int
      available_experts: List[str]
    response:
      registered: bool
      auth_token: string
```

## 6. Security Model

### 6.1 Cryptographic Primitives
- **Hashing**: SHA3-256 for blocks
- **Signatures**: ECDSA with secp256k1
- **Encryption**: AES-256-GCM for data
- **Key Derivation**: PBKDF2 with 100k iterations

### 6.2 Attack Mitigation

#### Model Poisoning Prevention
```python
class IntegrityVerification:
    def verify_expert(self, expert_id: str, weights: Tensor) -> bool:
        """Multi-layer verification"""
        # 1. Weight hash verification
        expected_hash = self.blockchain.get_expert_hash(expert_id)
        actual_hash = sha3_256(weights.numpy().tobytes())
        
        # 2. Activation beacon test
        test_input = self.get_beacon_input(expert_id)
        expected_output = self.get_beacon_output(expert_id)
        actual_output = expert.forward(test_input)
        
        # 3. Statistical validation
        weight_stats = calculate_statistics(weights)
        expected_stats = self.get_expected_stats(expert_id)
        
        return (
            actual_hash == expected_hash and
            torch.allclose(actual_output, expected_output) and
            validate_statistics(weight_stats, expected_stats)
        )
```

### 6.3 Node Authentication

#### Hardware Binding
```python
class HardwareBinding:
    """GPU UUID-based authentication"""
    
    def bind_node(self, node_id: str) -> Dict:
        gpu_uuids = get_gpu_uuids()
        
        binding = {
            "node_id": node_id,
            "gpu_uuids": gpu_uuids,
            "timestamp": time.time(),
            "cpu_id": get_cpu_id(),
            "mac_addresses": get_mac_addresses()
        }
        
        signature = sign_with_node_key(binding)
        self.store_binding(node_id, binding, signature)
        
        return binding
```

## 7. Performance Specifications

### 7.1 System Requirements

#### Minimum Node Requirements
```yaml
hardware:
  gpu:
    memory: 16GB VRAM
    compute: 7.0+ (Volta or newer)
  cpu:
    cores: 8
    ram: 32GB
  storage:
    ssd: 500GB
    bandwidth: 500MB/s
  network:
    bandwidth: 100Mbps
    latency: <100ms to peers
```

#### Production Deployment
```yaml
infrastructure:
  main_node:
    provider: DigitalOcean
    specs: 4vCPU, 8GB RAM
    location: NYC3
    cost: $48/month
  
  gpu_nodes:
    provider: RunPod
    specs: RTX 4090, 24GB VRAM
    locations: Global
    cost: $0.44/hour
  
  storage:
    primary: PostgreSQL (100GB)
    cache: Redis (16GB)
    archive: IPFS + S3
```

### 7.2 Performance Targets

#### Latency Requirements
- **Inference**: <1000ms p99
- **Block Finality**: <2000ms
- **Expert Loading**: <500ms
- **API Response**: <100ms

#### Throughput Targets
- **Transactions**: 1000+ TPS
- **Inference**: 100+ QPS per node
- **Training**: 1GB/hour per node
- **Validation**: 1000+ validations/hour

### 7.3 Scalability Design

#### Horizontal Scaling
```python
class ScalabilityManager:
    def scale_inference(self, load: float) -> int:
        """Auto-scale inference nodes"""
        current_nodes = len(self.active_nodes)
        target_nodes = ceil(load / self.capacity_per_node)
        
        if target_nodes > current_nodes:
            self.spawn_nodes(target_nodes - current_nodes)
        elif target_nodes < current_nodes * 0.7:
            self.shutdown_nodes(current_nodes - target_nodes)
        
        return len(self.active_nodes)
```

## 8. Data Structures

### 8.1 Core Data Models

#### Block Structure
```python
@dataclass
class Block:
    index: int
    timestamp: float
    block_type: BlockType
    data: bytes
    metadata: Dict[str, Any]
    previous_hash: str
    hash: str
    depends_on: List[str]  # DAG dependencies
    signature: bytes
    
    def calculate_hash(self) -> str:
        content = f"{self.index}{self.timestamp}{self.data}{self.previous_hash}"
        return sha3_256(content.encode()).hexdigest()
```

#### Expert Registry
```python
@dataclass
class ExpertRecord:
    expert_id: str
    layer: int
    version: str
    block_hash: str
    weight_cid: str  # IPFS content ID
    size_bytes: int
    quality_score: float
    usage_count: int
    last_updated: datetime
    node_assignments: List[str]
```

### 8.2 Economic Models

#### Reward Calculation
```python
@dataclass
class RewardPolicy:
    # Inference rewards
    per_1k_tokens_bly: float = 1.0
    quality_multiplier: Tuple[float, float] = (0.5, 1.5)
    
    # Learning rewards
    per_1pct_improvement_bly: float = 500.0
    difficulty_factor: Tuple[float, float] = (1.0, 3.0)
    
    # Budget allocation
    daily_budget_bly: float = 273_972
    inference_allocation: float = 0.45
    learning_allocation: float = 0.35
    validation_allocation: float = 0.10
    dataset_allocation: float = 0.10
```

## 9. Development Standards

### 9.1 Code Organization
```
blyan/
├── api/              # REST API endpoints
├── backend/
│   ├── core/        # Blockchain core
│   ├── model/       # AI model management
│   ├── p2p/         # Network layer
│   ├── learning/    # Training systems
│   └── security/    # Security modules
├── blockchain/       # Chain implementations
├── frontend/        # Web interface
├── scripts/         # Utilities
└── tests/          # Test suites
```

### 9.2 Testing Requirements
- **Unit Tests**: >80% code coverage
- **Integration Tests**: All API endpoints
- **Load Tests**: 1000+ concurrent users
- **Security Tests**: Penetration testing
- **Chaos Tests**: Network partition handling

### 9.3 Documentation Standards
- **API Documentation**: OpenAPI 3.0 spec
- **Code Comments**: Docstrings for all public methods
- **Architecture Diagrams**: C4 model
- **Runbooks**: Operational procedures
- **Change Logs**: Semantic versioning

## 10. Deployment Architecture

### 10.1 Container Strategy
```dockerfile
# GPU Node Container
FROM nvidia/cuda:12.3-runtime-ubuntu22.04
RUN pip install torch transformers fastapi
COPY backend/ /app/backend/
CMD ["python", "-m", "backend.p2p.gpu_node"]
```

### 10.2 Orchestration
```yaml
# Kubernetes Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: blyan-gpu-node
spec:
  replicas: 10
  template:
    spec:
      containers:
      - name: gpu-node
        image: blyan/gpu-node:latest
        resources:
          limits:
            nvidia.com/gpu: 1
```

### 10.3 Monitoring Stack
- **Metrics**: Prometheus + Grafana
- **Logging**: ELK Stack
- **Tracing**: Jaeger
- **Alerting**: PagerDuty integration

## 11. Migration Path

### 11.1 From Prototype to Production
1. **Database Migration**: File → PostgreSQL
2. **Consensus Upgrade**: Simple → BFT
3. **Storage Migration**: Local → Distributed
4. **Network Hardening**: HTTP → gRPC+TLS

### 11.2 Version Compatibility
- **Protocol Version**: v2.0
- **Backward Compatibility**: 1 major version
- **Deprecation Period**: 3 months
- **Migration Tools**: Automated scripts

## Appendices

### A. Error Codes
```python
class ErrorCode(Enum):
    INVALID_BLOCK = 1001
    INSUFFICIENT_BALANCE = 2001
    EXPERT_NOT_FOUND = 3001
    NODE_OFFLINE = 4001
    VALIDATION_FAILED = 5001
```

### B. Configuration Parameters
```yaml
# config/system.yaml
system:
  chain_id: "blyan-mainnet"
  block_time: 2.0
  max_block_size: 1MB
  expert_cache_size: 10GB
  inference_timeout: 30s
```

### C. API Rate Limits
- **Public**: 100 requests/minute
- **Authenticated**: 1000 requests/minute
- **Enterprise**: Unlimited with SLA

---
*Last Updated: January 2025*
*Version: 2.0.0*
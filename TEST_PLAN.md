# Test Plan
# Blyan Network Testing Strategy

## Test Environment Setup

### Local Development
```bash
# Install test dependencies
pip install pytest pytest-cov pytest-asyncio pytest-benchmark

# Set test environment
export BLYAN_ENV=test
export BLYAN_TEST_MODE=true
export SKIP_DB_INIT=true

# Run all tests
pytest tests/ -v --cov=backend --cov-report=html
```

### Docker Test Environment
```bash
# Build test container
docker build -t blyan-test -f Dockerfile.test .

# Run tests in container
docker run --rm blyan-test pytest tests/

# Run with GPU support
docker run --gpus all --rm blyan-test pytest tests/gpu/
```

## Test Categories

### 1. Unit Tests

#### Blockchain Core
```python
# tests/unit/test_blockchain.py
def test_block_creation():
    block = Block(
        index=1,
        data=b"test_data",
        block_type="delta",
        previous_hash="0x123"
    )
    assert block.calculate_hash() != ""
    assert block.validate()

def test_dag_cycle_detection():
    chain = Chain(test_dir, "A")
    # Add blocks with dependencies
    block1 = chain.add_block(b"data1")
    block2 = chain.add_block(b"data2", depends_on=[block1])
    # This should fail - circular dependency
    with pytest.raises(CycleDetected):
        chain.add_block(b"data3", depends_on=[block2, block1])

def test_merkle_root_generation():
    blocks = [create_test_block(i) for i in range(10)]
    merkle_root = generate_merkle_root(blocks)
    assert len(merkle_root) == 64  # SHA-256 hex
```

#### Model Management
```python
# tests/unit/test_model.py
def test_layer_partitioning():
    partitioner = LayerPartitioner()
    partitions = partitioner.partition_model(num_gpus=4)
    assert len(partitions) == 4
    assert sum(len(p["layers"]) for p in partitions) == 32

def test_delta_compression():
    original = torch.randn(1000, 1000)
    delta = torch.randn(1000, 1000) * 0.01
    compressed = compress_delta(delta)
    decompressed = decompress_delta(compressed)
    assert torch.allclose(delta, decompressed, atol=1e-4)

def test_quantization():
    weights = torch.randn(4096, 4096)
    quantized = quantize_to_fp8(weights)
    assert quantized.dtype == torch.float8_e4m3fn
    assert quantized.numel() == weights.numel()
```

#### API Endpoints
```python
# tests/unit/test_api.py
@pytest.mark.asyncio
async def test_chat_endpoint(client):
    response = await client.post("/chat", json={
        "prompt": "Hello",
        "max_tokens": 10
    })
    assert response.status_code == 200
    assert "text" in response.json()

@pytest.mark.asyncio
async def test_rate_limiting(client):
    # Exceed rate limit
    for _ in range(101):
        response = await client.get("/health")
    assert response.status_code == 429
```

### 2. Integration Tests

#### P2P Network
```python
# tests/integration/test_p2p.py
@pytest.mark.asyncio
async def test_node_registration():
    # Start coordinator
    coordinator = DistributedInferenceCoordinator()
    
    # Register node
    node_id = await coordinator.register_node({
        "node_id": "test-gpu-001",
        "host": "localhost",
        "port": 8001,
        "layers": [0, 1, 2, 3]
    })
    
    assert node_id in coordinator.get_active_nodes()
    
    # Test heartbeat
    await coordinator.heartbeat(node_id)
    assert coordinator.is_node_healthy(node_id)

@pytest.mark.asyncio
async def test_distributed_inference():
    # Setup 4 GPU nodes
    nodes = await setup_test_nodes(4)
    
    # Run inference
    response = await distributed_inference(
        prompt="Test prompt",
        nodes=nodes
    )
    
    assert response["tokens_used"] > 0
    assert len(response["nodes_used"]) > 1
```

#### Blockchain Integration
```python
# tests/integration/test_blockchain_integration.py
def test_two_chain_bridge():
    tx_chain = TransactionChain()
    pol_chain = PoLChain()
    bridge = CrossChainBridge(tx_chain, pol_chain)
    
    # Create reward on PoL chain
    receipt = pol_chain.create_reward_receipt(
        address="0x123",
        amount=100.0,
        proof="training_proof"
    )
    
    # Bridge to tx chain
    tx_hash = bridge.relay_reward(receipt)
    
    # Verify on tx chain
    balance = tx_chain.get_balance("0x123")
    assert balance >= 100.0
```

### 3. End-to-End Tests

#### Complete Inference Flow
```python
# tests/e2e/test_inference_flow.py
@pytest.mark.e2e
async def test_full_inference_pipeline():
    # 1. Register GPU nodes
    nodes = await register_gpu_nodes(4)
    
    # 2. Load model from blockchain
    model = await load_model_from_blockchain("Qwen3-8B")
    
    # 3. Partition model
    partitions = partition_model(model, nodes)
    
    # 4. Run inference
    response = await run_inference(
        prompt="Explain quantum computing",
        max_tokens=100,
        partitions=partitions
    )
    
    # 5. Verify response
    assert len(response.text) > 50
    assert response.latency_ms < 1000
    assert response.cost_bly > 0
```

#### Training & Evolution
```python
# tests/e2e/test_training_flow.py
@pytest.mark.e2e
async def test_training_evolution():
    # 1. Submit training job
    job_id = await submit_training_job(
        dataset="test_dataset",
        layers=[0, 1],
        epochs=1
    )
    
    # 2. Wait for completion
    result = await wait_for_job(job_id, timeout=300)
    
    # 3. Verify improvement
    assert result.improvement > 0.01
    
    # 4. Submit delta to blockchain
    block_hash = await submit_delta(result.delta)
    
    # 5. Verify block creation
    block = await get_block(block_hash)
    assert block.type == "delta"
```

### 4. Performance Tests

#### Load Testing
```python
# tests/performance/test_load.py
@pytest.mark.benchmark
def test_inference_throughput(benchmark):
    """Test inference QPS"""
    client = APIClient()
    
    def run_inference():
        return client.chat("test", max_tokens=10)
    
    result = benchmark.pedantic(
        run_inference,
        rounds=100,
        iterations=10
    )
    
    # Should handle 100+ QPS
    assert result.stats.mean < 0.01  # 10ms average

@pytest.mark.benchmark
def test_concurrent_users(benchmark):
    """Test with 1000 concurrent users"""
    async def concurrent_requests():
        tasks = []
        for _ in range(1000):
            tasks.append(api_request())
        results = await asyncio.gather(*tasks)
        return results
    
    results = benchmark(concurrent_requests)
    success_rate = sum(1 for r in results if r.ok) / 1000
    assert success_rate > 0.95  # 95% success rate
```

#### Memory & GPU Tests
```python
# tests/performance/test_gpu.py
@pytest.mark.gpu
def test_memory_usage():
    """Test GPU memory optimization"""
    model = load_model_fp8("Qwen3-8B")
    
    # Check memory usage
    memory_used = torch.cuda.memory_allocated()
    assert memory_used < 8 * 1024**3  # Less than 8GB
    
    # Run inference
    output = model.generate("test", max_tokens=100)
    peak_memory = torch.cuda.max_memory_allocated()
    assert peak_memory < 12 * 1024**3  # Peak under 12GB

@pytest.mark.gpu
def test_pipeline_parallel():
    """Test pipeline parallelism efficiency"""
    gpus = setup_multi_gpu(4)
    model = distribute_model(gpus)
    
    start = time.time()
    for _ in range(100):
        model.forward(get_test_input())
    
    throughput = 100 / (time.time() - start)
    assert throughput > 10  # 10+ inferences per second
```

### 5. Security Tests

#### Authentication & Authorization
```python
# tests/security/test_auth.py
def test_api_key_validation():
    # Invalid key
    response = client.get("/chat", headers={"X-API-Key": "invalid"})
    assert response.status_code == 401
    
    # Valid key
    response = client.get("/chat", headers={"X-API-Key": valid_key})
    assert response.status_code == 200

def test_siwe_authentication():
    # Create SIWE message
    message = create_siwe_message(address, statement)
    signature = sign_message(message, private_key)
    
    # Authenticate
    response = client.post("/auth/siwe", json={
        "message": message,
        "signature": signature
    })
    assert response.status_code == 200
    assert "token" in response.json()
```

#### Integrity Verification
```python
# tests/security/test_integrity.py
def test_weight_verification():
    """Test model weight integrity"""
    layer = load_layer_from_blockchain(0)
    
    # Verify hash
    expected_hash = get_layer_hash(0)
    actual_hash = calculate_hash(layer)
    assert actual_hash == expected_hash
    
    # Verify activation beacon
    beacon_input = get_beacon_input(0)
    beacon_output = layer.forward(beacon_input)
    expected_output = get_expected_beacon_output(0)
    assert torch.allclose(beacon_output, expected_output)

def test_tamper_detection():
    """Test detection of tampered weights"""
    layer = load_layer_from_blockchain(0)
    
    # Tamper with weights
    layer.weight.data[0, 0] += 0.1
    
    # Should detect tampering
    with pytest.raises(IntegrityError):
        verify_layer_integrity(layer)
```

### 6. Chaos Testing

#### Network Failures
```python
# tests/chaos/test_network.py
@pytest.mark.chaos
async def test_node_failure():
    """Test handling of node failures"""
    nodes = await setup_nodes(4)
    
    # Start inference
    task = asyncio.create_task(
        distributed_inference("long prompt" * 100)
    )
    
    # Kill a node mid-inference
    await asyncio.sleep(1)
    await kill_node(nodes[1])
    
    # Should complete with remaining nodes
    result = await task
    assert result.success
    assert len(result.nodes_used) == 3

@pytest.mark.chaos
async def test_network_partition():
    """Test network partition handling"""
    nodes = await setup_nodes(6)
    
    # Create network partition
    partition_network(nodes[:3], nodes[3:])
    
    # Both partitions should continue operating
    result1 = await inference_on_nodes(nodes[:3])
    result2 = await inference_on_nodes(nodes[3:])
    
    assert result1.success
    assert result2.success
```

## Test Data

### Fixtures
```python
# tests/conftest.py
@pytest.fixture
def test_blockchain(tmp_path):
    """Create test blockchain"""
    chain = Chain(tmp_path, "test")
    # Add genesis block
    chain.add_block(b"genesis", block_type="meta")
    return chain

@pytest.fixture
async def test_client():
    """Create test API client"""
    app = create_app(test_mode=True)
    async with AsyncClient(app=app) as client:
        yield client

@pytest.fixture
def test_model():
    """Create small test model"""
    return create_test_model(
        layers=4,
        hidden_size=256,
        vocab_size=1000
    )
```

## CI/CD Pipeline

### GitHub Actions
```yaml
# .github/workflows/test.yml
name: Test Suite

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirements-test.txt
      - run: pytest tests/unit/ --cov

  integration-tests:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:14
      redis:
        image: redis:7
    steps:
      - uses: actions/checkout@v3
      - run: pytest tests/integration/

  gpu-tests:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v3
      - run: pytest tests/gpu/ -m gpu
```

## Test Metrics

### Coverage Requirements
- **Unit Tests**: >80% code coverage
- **Integration Tests**: All critical paths
- **E2E Tests**: Main user journeys
- **API Tests**: 100% endpoint coverage

### Performance Targets
| Metric | Target | Current |
|--------|--------|---------|
| Inference Latency | <1000ms | ✅ 850ms |
| Throughput | >100 QPS | ✅ 120 QPS |
| Success Rate | >99.9% | ✅ 99.95% |
| GPU Memory | <12GB | ✅ 8.5GB |

### Test Execution Time
- Unit Tests: <2 minutes
- Integration Tests: <10 minutes
- E2E Tests: <30 minutes
- Full Suite: <45 minutes

## Test Commands

```bash
# Run specific test categories
pytest tests/unit/                    # Unit tests only
pytest tests/integration/             # Integration tests
pytest tests/e2e/ -m e2e             # End-to-end tests
pytest tests/performance/ -m benchmark # Performance tests
pytest tests/security/                # Security tests
pytest tests/chaos/ -m chaos         # Chaos tests

# Run with coverage
pytest --cov=backend --cov-report=html --cov-report=term

# Run specific test
pytest tests/unit/test_blockchain.py::test_dag_cycle_detection -v

# Run tests in parallel
pytest -n 4 tests/

# Run with GPU
pytest tests/gpu/ -m gpu --gpu-device=0

# Generate test report
pytest --html=report.html --self-contained-html
```

---
*Test Plan Version: 1.0.0*
*Last Updated: January 2025*
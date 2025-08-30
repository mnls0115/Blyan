# Troubleshooting Guide

## GPU Node Issues

### Heartbeat 422 Error
**Problem**: GPU node receives HTTP 422 when sending heartbeats.

**Solution**: The heartbeat endpoint now properly handles Bearer token authentication:
```bash
# Correct format with Bearer token
curl -X POST https://blyan.com/api/gpu/heartbeat \
  -H "Authorization: Bearer $BLYAN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '"node-id"'
```

**Accepted body formats**:
- Raw JSON string: `"node-id"` (recommended for GPU nodes)
- Object format: `{"node_id": "node-id"}` (backwards compatible)

### CUDA Out of Memory (OOM)
**Problem**: GPU runs out of memory during model loading, showing errors like:
```
CUDA out of memory. Tried to allocate X GiB... Process has 42.51 GiB in use
```

**Causes**:
1. Overlapping model loads (warmup + chat request)
2. Insufficient GPU memory for BF16 model (requires ~16GB)
3. Memory fragmentation

**Solutions**:

1. **Prevent overlapping loads**: The model manager now uses a concurrency lock to ensure only one model load happens at a time.

2. **Optimize VRAM usage**:
```bash
# Set environment variables before starting node
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export GPU_DIRECT_CHUNK_SIZE=1
export GPU_LOAD_WORKERS=1
```

3. **Monitor GPU memory**:
```bash
# Check available VRAM
nvidia-smi

# Clear GPU cache if needed
python -c "import torch; torch.cuda.empty_cache()"
```

4. **For tight VRAM (40GB GPUs)**:
```bash
# Use single-threaded loading
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

### Node Shows as Inactive
**Problem**: GPU node registered but shows as inactive in `/api/gpu/status`.

**Solutions**:
1. Ensure heartbeats are being sent every 15 seconds
2. Check API key is correct: `echo $BLYAN_API_KEY`
3. Verify node can reach main API: `curl https://blyan.com/health`
4. Check logs for heartbeat errors: `grep "heartbeat" logs.txt`

### Registration Fails
**Problem**: GPU node fails to register with main node.

**Common causes**:
1. Missing or invalid `BLYAN_API_KEY`
2. Network connectivity issues
3. Main node P2P not initialized (this is OK - node runs standalone)

**Debug steps**:
```bash
# Test API connectivity
curl https://blyan.com/health

# Test with your API key
curl -X POST https://blyan.com/api/gpu/register \
  -H "Authorization: Bearer $BLYAN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "node_id": "test-node",
    "api_url": "https://your-node.com",
    "capabilities": {"gpu_memory_gb": 40, "layers": [0,1,2,3]}
  }'
```

## Model Loading Issues

### BF16 Precision Requirements
**Problem**: Model fails to load with dtype errors.

**Solution**: Ensure GPU supports BF16 (Ampere or newer):
```python
import torch
print(torch.cuda.get_device_capability())  # Should be (8,0) or higher
```

### Blockchain Weights Not Found
**Problem**: "Cannot proceed without blockchain weights" error.

**Solution**:
1. Check blockchain has genesis block: `ls data/A/`
2. Verify param_index exists: `cat data/param_index.json`
3. Initialize if needed:
```python
from backend.core.chain import Chain
chain = Chain('./data', 'A')
chain.add_block(b'{"model":"Qwen3-8B"}', block_type='meta')
```

## Network Issues

### 503 Service Unavailable
**Problem**: Frontend shows "No GPU nodes available".

**Causes**:
1. No GPU nodes registered
2. All nodes inactive (missed heartbeats)
3. Redis connection issues

**Solutions**:
1. Check node status: `curl https://blyan.com/api/gpu/status`
2. Register a node (see Registration section)
3. Check Redis: `redis-cli ping`

### Connection Timeouts
**Problem**: Requests timeout or take too long.

**Solutions**:
1. Check network latency: `ping blyan.com`
2. Use appropriate timeouts:
```python
import httpx
client = httpx.AsyncClient(timeout=30.0)  # 30 second timeout
```
3. For large responses, use streaming endpoints

## Performance Optimization

### Slow Inference
**Problem**: Chat responses take too long.

**Solutions**:
1. Enable GPU-direct loading: `export GPU_DIRECT=true`
2. Use pipeline parallelism for multi-GPU setups
3. Enable speculative decoding if available
4. Monitor GPU utilization: `nvidia-smi dmon -s u`

### High Memory Usage
**Problem**: System uses excessive RAM/VRAM.

**Solutions**:
1. Limit cache size: `export TRANSFORMERS_CACHE=/tmp/hf_cache`
2. Clear unused tensors: `torch.cuda.empty_cache()`
3. Use gradient checkpointing for training
4. Reduce batch size for inference

## Debugging Tips

### Enable Debug Logging
```bash
export LOG_LEVEL=DEBUG
export PYTHONUNBUFFERED=1
python run_gpu_node.py 2>&1 | tee debug.log
```

### Check System Status
```bash
# GPU status
nvidia-smi

# Network status
netstat -an | grep 8000

# Process status
ps aux | grep python

# Disk usage
df -h

# Memory usage
free -h
```

### Common Log Patterns
- `💓 Heartbeat sent successfully` - Node is healthy
- `💔 Heartbeat failed` - Check API key and network
- `🔒 Loading model EXCLUSIVELY from blockchain` - Correct behavior
- `⚠️ DEVELOPMENT MODE` - Not for production use
- `CUDA out of memory` - See VRAM optimization section

## Getting Help

If issues persist:
1. Check logs for specific error messages
2. Verify all environment variables are set correctly
3. Ensure you're using the latest code version
4. Create an issue with:
   - Error messages
   - Environment details (OS, GPU, Python version)
   - Steps to reproduce
   - Relevant log excerpts
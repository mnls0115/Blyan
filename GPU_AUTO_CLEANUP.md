# GPU Node Auto-Cleanup Implementation

## Features Added

### 1. Automatic Stale Node Cleanup
- **Background task** runs every 60 seconds
- **Two-stage process**:
  1. Nodes without heartbeat for 30s → marked as "inactive"
  2. Nodes inactive for >1 hour → automatically removed
- **Complete cleanup** includes:
  - Remove from layer assignments
  - Delete node data, heartbeat, and metrics
  - Remove from active nodes set
  - Publish cleanup event for monitoring

### 2. Configurable Timeout
- **Default**: 1 hour before removing inactive nodes
- **Environment variable**: `GPU_NODE_STALE_TIMEOUT_HOURS`
- **Constructor parameter**: `stale_timeout_hours`

Example:
```python
# Set to 2 hours
manager = GPUNodeManagerRedis(stale_timeout_hours=2.0)

# Or via environment
export GPU_NODE_STALE_TIMEOUT_HOURS=2.0
```

### 3. Manual Cleanup API Endpoint
```bash
# Normal cleanup (respects timeout)
POST /api/gpu/cleanup

# Force cleanup (removes all inactive nodes)
POST /api/gpu/cleanup?force=true
```

### 4. Cleanup Method
```python
# Programmatic cleanup
result = await gpu_node_manager.cleanup_stale_nodes(force=False)

# Returns:
{
    "success": true,
    "nodes_removed": [
        {"node_id": "gpu_node_123", "inactive_hours": 2.5}
    ],
    "nodes_marked_inactive": ["gpu_node_456"],
    "total_removed": 1,
    "total_marked_inactive": 1,
    "timestamp": 1756543200
}
```

## How It Works

### Heartbeat Lifecycle
1. **Active** (0-30s): Node sending heartbeats, fully operational
2. **Inactive** (30s-1hr): No heartbeat, marked inactive but kept for recovery
3. **Removed** (>1hr): Automatically cleaned up, all data deleted

### Redis Keys Used
- `gpu:node:{node_id}` - Node configuration and state
- `gpu:heartbeat:{node_id}` - TTL key (30s) for liveness
- `gpu:metrics:{node_id}` - Performance metrics
- `gpu:layer:{layer_id}` - Set of nodes serving each layer

### Cleanup Process
1. Every 60 seconds, scan all nodes
2. Check heartbeat key existence
3. If no heartbeat:
   - Calculate time since last heartbeat
   - If >1 hour: remove completely
   - If <1 hour: mark as inactive
4. Pipeline all deletions for atomicity
5. Publish event for monitoring

## Benefits

1. **Automatic cleanup**: No manual intervention needed
2. **Prevents clutter**: Old nodes don't accumulate
3. **Resource efficiency**: Redis memory freed automatically
4. **Recovery window**: 1-hour grace period for temporary issues
5. **Configurable**: Adjust timeout based on your needs
6. **Observable**: Events published for monitoring

## Configuration Examples

### Quick cleanup (15 minutes)
```bash
export GPU_NODE_STALE_TIMEOUT_HOURS=0.25
```

### Conservative cleanup (4 hours)
```bash
export GPU_NODE_STALE_TIMEOUT_HOURS=4.0
```

### Disable auto-cleanup (24 hours)
```bash
export GPU_NODE_STALE_TIMEOUT_HOURS=24.0
```

## Monitoring

Watch cleanup events:
```bash
redis-cli subscribe gpu:events
```

Check node status:
```bash
curl https://blyan.com/api/gpu/status
```

## Testing

```python
# Register a test node
await manager.register_node("test-1", "http://test", "key")

# Stop heartbeats and wait
await asyncio.sleep(3700)  # Wait >1 hour

# Node should be auto-removed
status = await manager.get_node_status()
assert "test-1" not in [n["node_id"] for n in status["nodes"]]
```

## Summary

The GPU node manager now automatically disconnects and removes nodes that haven't responded for the configured timeout period (default 1 hour). This ensures:
- Clean registry without stale entries
- Efficient resource usage
- Automatic recovery from node failures
- No manual cleanup required
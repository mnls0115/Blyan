# GPU Node Architecture

## Key Understanding: Blockchain Location

### ❌ INCORRECT Assumption:
- Main node (DigitalOcean) stores blockchain
- GPU nodes sync from main node

### ✅ CORRECT Architecture:
- **Main Node (DigitalOcean)**: Service coordinator only - NO blockchain
- **GPU Nodes**: Each maintains its OWN independent blockchain

## Node Types

### 1. Main Node (Service Node)
- Location: DigitalOcean (165.227.221.225)
- Role: API gateway, request routing, coordination
- Blockchain: **NONE** - just a coordinator
- Function: Routes inference requests to GPU nodes

### 2. GPU Nodes
- Location: Distributed (RunPod, local, etc.)
- Role: Model hosting, inference, training
- Blockchain: **LOCAL** - each node has its own
- Function: Store model experts in local blockchain

## Blockchain Distribution

```
Main Node (DigitalOcean)
    |
    ├── No blockchain storage
    ├── Routes requests only
    └── Coordinates GPU nodes

GPU Node 1                  GPU Node 2
    |                           |
    ├── Local Chain A           ├── Local Chain A
    ├── Local Chain B           ├── Local Chain B
    └── 6,144 experts          └── 6,144 experts
```

## Implications

1. **No Central Blockchain**: Each GPU node is independent
2. **No Sync Needed**: GPU nodes don't sync from main node
3. **Local Storage**: Each GPU node stores full model locally
4. **Future Enhancement**: GPU-to-GPU sync could be added later

## Current Implementation

The code has been updated to reflect this architecture:
- Removed sync from main node
- Allow empty blockchains (normal for new nodes)
- Each GPU node uploads its own model copy

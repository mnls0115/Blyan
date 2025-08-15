# Anchored Bridge Protocol

## Proof Flow Architecture

### RewardReceipt Generation (PoL Chain)
```
RewardReceipt {
    proof_id: bytes32,      // Unique identifier
    earner: address,        // Recipient address
    amount: uint256,        // BLY tokens earned
    epoch: uint64,          // PoL epoch number
    meta_hash: bytes32      // Reference to model/inference metadata
}
```

### Merkle Tree Construction
```
Epoch N:
    Root
    ├── H(Receipt1 || Receipt2)
    │   ├── Receipt1
    │   └── Receipt2
    └── H(Receipt3 || Receipt4)
        ├── Receipt3
        └── Receipt4
```

### Claim Process
1. **Generate Proof**: PoL node provides Merkle proof for receipt
2. **Submit to Tx-Chain**: User calls Token.claim(receipt, proof)
3. **Verification**: Contract verifies:
   - Merkle proof validity
   - Epoch finalization status
   - No prior claim (claimed[proof_id] == false)
4. **Mint**: BLY tokens minted to earner address

### Relayer Operations
```
while true:
    // PoL → Tx anchoring
    if (block.height % N == 0):
        epoch = pol.getFinalizedEpoch()
        root = pol.getEpochRoot(epoch)
        tx.AnchoringPoL.commit(epoch, root, sigs, proof)
    
    // Tx → PoL anchoring
    if pol.shouldAnchor():
        header = tx.getFinalizedHeader()
        pol.AnchoringTx.commit(header, sigs, proof)
    
    sleep(RELAY_INTERVAL)
```

## Security Guarantees
- **Finality-only**: Only finalized data crosses chains
- **Double-claim Prevention**: claimed mapping + proof_id uniqueness
- **Byzantine Tolerance**: 2/3+ signatures required
- **Replay Protection**: Epoch numbers strictly increasing
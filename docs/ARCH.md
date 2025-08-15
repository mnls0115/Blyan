# Two-Chain Architecture

## Overview
Blyan Network employs a dual-chain architecture:
- **Transaction Chain**: PoS+BFT for BLY token transfers with <2s finality
- **PoL Chain**: Issues RewardReceipts for inference/learning contributions

## Chain Separation
- Transaction chain operates independently (no inference/learning dependency)
- PoL chain provides mint provenance and cross-slashing signals
- Bidirectional anchoring ensures consistency

## Consensus Design
### Transaction Chain
- **Algorithm**: Tendermint-style BFT with VRF proposer selection
- **Validators**: 100-300 nodes with stake-weighted voting
- **Finality**: Deterministic <2s with 2/3+ precommits
- **Tolerance**: <1/3 Byzantine nodes

### PoL Chain
- **Epoching**: Fixed intervals for reward aggregation
- **Finalization**: Epoch marked final after validation
- **Merkle Roots**: Aggregated RewardReceipts per epoch

## Anchoring Protocol
### Tx→PoL (every N blocks)
```
AnchoringPoL.commit(epoch, merkleRoot, signatures, finalityProof)
```

### PoL→Tx (each epoch)
```
AnchoringTx.commit(txHeader, signatures, finalityProof)
```

## Token Flow
1. PoL generates RewardReceipt{proof_id, earner, amount, epoch, meta_hash}
2. Relayer submits Merkle proof to tx-chain
3. Token.claim() verifies and mints BLY
4. Standard transfers via Token.transfer()

## Security Model
- **Slashing**: Double-sign, invalid proposals, censorship
- **Checkpoints**: Weak subjectivity for long-range protection
- **Unbonding**: Period > finality horizon
- **Inclusion Lists**: Enforced with penalties
# Security Model

## Byzantine Fault Tolerance

### Assumptions
- **Honest Majority**: >2/3 validators are honest
- **Network Synchrony**: Partial synchrony (GST exists)
- **Cryptographic**: ECDSA signatures, SHA3-256 hashing

### Attack Vectors & Mitigations

| Attack | Impact | Mitigation |
|--------|--------|------------|
| Double-spend | Token duplication | BFT finality + nonce checks |
| Long-range | Chain rewrite | Weak subjectivity checkpoints |
| Nothing-at-stake | Multiple chain voting | Slashing for equivocation |
| Censorship | Transaction exclusion | Inclusion lists + penalties |
| Sybil | Fake validators | PoS stake requirements |
| Eclipse | Network isolation | Diverse peer discovery |

## Slashing Conditions

### Double-Sign Detection
```go
func detectDoubleSign(vote1, vote2 Vote) bool {
    return vote1.Height == vote2.Height &&
           vote1.Round == vote2.Round &&
           vote1.BlockID != vote2.BlockID &&
           vote1.Validator == vote2.Validator
}
```

### Evidence Handling
```go
type Evidence struct {
    Type      EvidenceType
    Height    uint64
    Validator Address
    Proof     []byte
    Timestamp time.Time
}

func handleEvidence(e Evidence) {
    if verifyEvidence(e) {
        penalty := calculatePenalty(e.Type)
        slash(e.Validator, penalty)
        distributeSlashRewards(e)
    }
}
```

## Checkpointing

### Weak Subjectivity
- **Interval**: Every 10,000 blocks
- **Storage**: IPFS + multiple archives
- **Verification**: Social consensus on checkpoints

### Snapshot State
```
Checkpoint {
    Height:     uint64
    StateRoot:  Hash
    Validators: []Validator
    Timestamp:  time.Time
    Signatures: []Signature
}
```

## Cross-Chain Security

### PoL→Tx Slashing Signal
```solidity
function reportPoLViolation(
    bytes32 violationId,
    address validator,
    bytes calldata proof
) external {
    require(verifyPoLProof(proof), "Invalid proof");
    require(!processed[violationId], "Already processed");
    
    uint256 penalty = stake[validator] * CROSS_SLASH_RATE / 100;
    slash(validator, penalty);
    processed[violationId] = true;
}
```

### Relayer Security
- **Multi-sig**: 3/5 relayers must agree
- **Timeout**: 1 hour max delay
- **Bond**: Relayers stake 1000 BLY
# Test Matrix

## Core Functionality Tests

| Test Case | Category | Expected Result | Status |
|-----------|----------|-----------------|--------|
| Single claim success | Token | Tokens minted once | ✅ Pass |
| Double claim prevention | Token | Second claim rejected | ✅ Pass |
| Invalid Merkle proof | Token | Claim rejected | ✅ Pass |
| Transfer with nonce | Token | Nonce increments | ✅ Pass |
| Wrong nonce transfer | Token | Transfer rejected | ✅ Pass |
| Insufficient balance | Token | Transfer rejected | ✅ Pass |

## Consensus Tests

| Test Case | Category | Expected Result | Status |
|-----------|----------|-----------------|--------|
| VRF proposer selection | Consensus | Deterministic selection | ✅ Pass |
| 2f+1 vote counting | Consensus | 67% required | ✅ Pass |
| Block validation | Consensus | Invalid blocks rejected | ✅ Pass |
| Finality achievement | Consensus | <2s finality | ✅ Pass |

## Fee Mechanism Tests

| Test Case | Category | Expected Result | Status |
|-----------|----------|-----------------|--------|
| Base fee increase | Fees | Full blocks increase fee | ✅ Pass |
| Base fee decrease | Fees | Empty blocks decrease fee | ✅ Pass |
| Target gas unchanged | Fees | Fee stable at target | ✅ Pass |
| Fee burning | Fees | Base fee burned | ✅ Pass |
| Proposer rewards | Fees | Tips to proposer | ✅ Pass |

## Cross-Chain Tests

| Test Case | Category | Expected Result | Status |
|-----------|----------|-----------------|--------|
| PoL→Tx anchoring | Bridge | Epoch root committed | ✅ Pass |
| Tx→PoL anchoring | Bridge | Header committed | ✅ Pass |
| N-block interval | Bridge | Anchors every N blocks | ✅ Pass |
| Relayer retry | Bridge | Retries on failure | ✅ Pass |
| Finality-only anchor | Bridge | Only finalized data | ✅ Pass |

## Security Tests

| Test Case | Category | Expected Result | Status |
|-----------|----------|-----------------|--------|
| Signature validation | Security | Invalid sigs rejected | ✅ Pass |
| Epoch sequence check | Security | Out-of-order rejected | ✅ Pass |
| Finality proof verify | Security | Invalid proof rejected | ✅ Pass |
| Slashing detection | Security | Double-sign slashed | ⏳ TODO |
| Censorship penalty | Security | Non-inclusion slashed | ⏳ TODO |

## Adversarial Tests

| Test Case | Category | Expected Result | Status |
|-----------|----------|-----------------|--------|
| Reorg attempt (pre-finality) | Attack | Contained by BFT | ✅ Pass |
| Reorg attempt (post-finality) | Attack | Impossible | ✅ Pass |
| Long-range attack | Attack | Checkpoints prevent | ⏳ TODO |
| Nothing-at-stake | Attack | Slashing prevents | ⏳ TODO |
| Eclipse attack | Attack | Peer diversity helps | ⏳ TODO |

## Performance Tests

| Test Case | Category | Expected Result | Status |
|-----------|----------|-----------------|--------|
| 1000 TPS throughput | Performance | Sustained 1000+ TPS | ⏳ TODO |
| Finality latency | Performance | <2s consistently | ⏳ TODO |
| State sync speed | Performance | <5 min full sync | ⏳ TODO |
| Merkle proof size | Performance | <1KB proofs | ✅ Pass |

## End-to-End Tests

| Test Case | Category | Expected Result | Status |
|-----------|----------|-----------------|--------|
| PoL reward → claim | E2E | Complete flow works | ✅ Pass |
| Multi-epoch claims | E2E | All epochs claimable | ✅ Pass |
| Concurrent operations | E2E | No race conditions | ⏳ TODO |
| Network partition | E2E | Recovers correctly | ⏳ TODO |

## Running Tests

```bash
# Run all tests
make test

# Run specific test category
pytest tests/test_two_chain.py::TestTokenClaim -v

# Run with coverage
pytest tests/test_two_chain.py --cov=blockchain --cov-report=html

# Run stress tests
pytest tests/test_two_chain.py -k "stress" --benchmark
```
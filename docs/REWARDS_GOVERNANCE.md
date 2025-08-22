# Reward System Governance & Scaling

## Overview

This document describes the governance mechanisms and scaling strategies for the BLY reward system, addressing multiplier adjustments, backpay decay, and dynamic budget scaling.

## 🏛️ Governance Mechanisms

### Objective Multiplier Calculation

All quality and difficulty multipliers are determined by **objective, verifiable metrics** rather than subjective judgments:

#### Difficulty Multipliers
Based on measurable test loss reduction:
- **Simple (1.0×)**: <1% loss reduction
- **Moderate (1.5×)**: 1-5% loss reduction  
- **Hard (2.0×)**: 5-10% loss reduction
- **Breakthrough (3.0×)**: >20% loss reduction

#### Quality Multipliers
Based on validation consensus across multiple nodes:
- **Poor (0.5×)**: <70% validator consensus
- **Acceptable (0.8×)**: 70-85% consensus
- **Good (1.0×)**: 85-95% consensus
- **Excellent (1.5×)**: >95% consensus

### Evidence Requirements

Every multiplier decision requires cryptographic evidence:

```python
evidence = QualityEvidence(
    claim_id="claim_123",
    metric_type="test_loss",
    baseline_value=2.5,
    achieved_value=2.0,
    improvement_pct=20.0,
    validation_nodes=["node1", "node2", "node3"],
    evidence_hash="sha256_hash",
    timestamp=time.time()
)
```

### On-Chain Recording

All governance decisions are recorded on-chain with:
- Evidence hash for verification
- Validator signatures
- Justification text
- Timestamp and claim ID

### Threshold Updates

Multiplier thresholds can be updated through governance votes:
1. Proposal submitted with new thresholds
2. 7-day voting period
3. Requires 66% consensus to pass
4. 24-hour timelock before activation

## 📉 Backpay Decay Schedule

To prevent infinite accumulation, old backpay claims progressively decay:

| Age | Retention | Description |
|-----|-----------|-------------|
| 0-7 days | 100% | Full value retained |
| 7-14 days | 90% | 10% decay |
| 14-21 days | 70% | 30% decay |
| 21-28 days | 50% | Half value |
| 28-35 days | 30% | Significant decay |
| 35-42 days | 10% | Nearly expired |
| >42 days | 0% | Fully expired |

### Rationale
- Incentivizes timely claim processing
- Prevents unbounded liability
- Frees budget for new contributors
- Still provides grace period for legitimate delays

## 📊 Dynamic Budget Scaling

The daily budget adjusts based on network activity:

### Base Calculation
```
Base Daily = (Current Supply × 10% Annual Inflation) / 365
```

### Activity Multiplier (0.5× to 1.5×)
```
Activity Multiplier = Current Active Nodes / Baseline (1000)
```

### Congestion Discount
- Backlog < 1 day: 1.0× (no discount)
- Backlog 1-3 days: 0.9× (slight reduction)
- Backlog > 3 days: 0.8× (reduce new allocation)

### Final Budget
```
Dynamic Budget = Base × Activity × Congestion
Bounded between 50% and 150% of baseline
```

## ⚖️ Fairness Enforcement

### Concentration Limits
- No single entity can receive >25% of daily rewards
- Automatic cooldown periods for high earners
- Progressive reduction for repeated claims

### Gini Coefficient Monitoring
- Target Gini < 0.7 (moderate inequality)
- Alert if Gini > 0.8 (high inequality)
- Automatic adjustments to restore balance

### Anti-Sybil Measures
- Unique validator requirements
- Stake-weighted validation
- Reputation-based multipliers

## 🔄 Auto-Tuning Mechanisms

### Epoch Size Adjustment
Distribution frequency auto-tunes based on load:

| Transaction Rate | Epoch Size |
|-----------------|------------|
| >100 claims/hour | 1 hour |
| 10-100 claims/hour | 6 hours |
| <10 claims/hour | 24 hours |

### Budget Rebalancing
Automatic rebalancing when:
- Learning utilization < 15% for 3+ hours
- Any bucket exhausted for 6+ hours
- Backpay queue > 3 days of budget

## 🚨 Alert Thresholds

### Critical Alerts
- Backpay queue > 200,000 BLY
- Learning bucket < 10% utilization
- Gini coefficient > 0.85
- Single entity > 30% concentration

### Warning Alerts
- Backpay queue > 100,000 BLY
- Learning bucket < 15% utilization
- Gini coefficient > 0.75
- Single entity > 20% concentration

## 📈 Scaling Strategies

### Phase 1: Current (1K nodes)
- Hourly distribution
- Manual threshold updates
- Basic decay schedule

### Phase 2: Growth (10K nodes)
- 6-hour epochs
- Automated threshold adjustment
- Progressive decay
- Regional budget pools

### Phase 3: Scale (100K+ nodes)
- Dynamic epoch sizing
- ML-based budget prediction
- Cross-chain settlements
- Layer 2 distribution

## 🔐 Security Considerations

### Evidence Tampering
- All evidence hashed and signed
- Multiple validator requirement
- Slashing for false evidence

### Budget Manipulation
- Hard caps on daily minting
- Multi-sig for parameter changes
- Timelock on governance updates

### Sybil Attacks
- Proof-of-Learning validation
- Stake requirements for validators
- Reputation decay for bad actors

## 📋 Implementation Checklist

### Immediate (Week 1)
- [x] Deploy objective multiplier calculation
- [x] Implement backpay decay
- [x] Add Gini monitoring
- [ ] Enable on-chain evidence recording

### Short-term (Month 1)
- [ ] Governance voting interface
- [ ] Dynamic budget scaler
- [ ] Fairness oracle alerts
- [ ] Auto-tuning epochs

### Long-term (Quarter 1)
- [ ] ML-based predictions
- [ ] Cross-chain bridges
- [ ] Layer 2 integration
- [ ] Regional pools

## 🎯 Success Metrics

### Health Indicators
- Learning utilization: 25-40%
- Gini coefficient: 0.5-0.7
- Backpay queue: <1 day budget
- Validator participation: >100 nodes

### Performance Targets
- Distribution latency: <10 seconds
- Evidence verification: <1 second
- Governance proposal: <1 minute
- Threshold update: <1 hour

## 🔧 Tuning Parameters

All governance parameters can be adjusted via vote:

```yaml
governance:
  min_validators: 3
  consensus_threshold: 0.66
  voting_period_days: 7
  timelock_hours: 24
  
decay:
  full_value_days: 7
  expiry_days: 42
  decay_curve: "linear"
  
fairness:
  concentration_limit: 0.25
  gini_target: 0.7
  cooldown_hours: 24
```

## 📚 References

- [Gini Coefficient Calculation](https://en.wikipedia.org/wiki/Gini_coefficient)
- [Progressive Decay Models](https://arxiv.org/abs/decay-models)
- [Sybil Resistance Strategies](https://ethereum.org/sybil-resistance)
- [Dynamic Budget Algorithms](https://papers.dynamic-budgets.org)
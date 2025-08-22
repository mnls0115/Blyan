# BLY Rewards System Documentation

## Overview

The Blyan Network rewards system is **entirely work-based**, not hardware-based. All rewards are calculated based on actual work performed: tokens processed, improvements validated, tasks completed, and data contributed.

## 🎯 Core Principles

1. **Work-Based Rewards Only**: No GPU-specific multipliers or hardware advantages
2. **Dynamic Budget Balancing**: Ensures both inference and learning are funded
3. **Transparent Calculations**: All formulas are public and verifiable
4. **Idempotent Distribution**: Prevents double-pays with replay protection
5. **Quality Incentives**: Higher quality work earns more rewards

## 💰 Reward Rates (Work-Based Units)

### Inference Rewards
- **Base Rate**: ~1 BLY per 1,000 tokens processed
- **Quality Multiplier**: 0.5× to 1.5× based on output quality
- **Example**: Processing 100k tokens at quality 1.1 = ~110 BLY

### Learning Rewards  
- **Base Rate**: ~500 BLY per 1% validated improvement
- **Difficulty Factor**: 1.0× to 3.0× for complexity
- **Applicability Factor**: 0.8× to 1.2× for broad usefulness
- **Example**: 2% improvement with difficulty 2.0 = ~2,000 BLY

### Validation Rewards
- **Base Rate**: 10 BLY per validation task
- **Complexity Bonus**: Additional multiplier for complex validations
- **Example**: Completing 50 validation tasks = 500 BLY

### Dataset Contribution Rewards
- **Base Rate**: 100 BLY per GB of quality data
- **Quality Score**: 0× to 1× based on data quality
- **Diversity Bonus**: Up to 2× for unique/diverse data
- **Example**: 10 GB at 0.9 quality = 900 BLY

## 📊 Budget Allocation System

### Daily Budget Distribution
Total daily budget: ~273,972 BLY (10% annual inflation of 1B supply)

| Category | Target | Floor | Ceiling |
|----------|--------|-------|---------|
| Inference | 45% | 30% | 60% |
| Learning | 35% | 25% | 50% |
| Validation | 10% | 5% | 15% |
| Dataset | 10% | 5% | 15% |

### Dynamic Balancing Features
- **Hourly Rebalancing**: Budgets refresh every hour
- **Rollover**: Unused budget rolls over for up to 24 hours
- **Floors**: Minimum guaranteed allocation per category
- **Ceilings**: Maximum allowed to prevent monopolization
- **Backpay Queue**: Unpaid rewards queued when budget exhausted

## 🔧 API Endpoints

### Estimate Rewards
```bash
GET /rewards/estimate?tokens=100000&quality=1.1
```

Response:
```json
{
  "inference": {
    "per_1k_tokens_bly": 1.0,
    "estimate_bly": 110.0,
    "budget_available": true
  },
  "notes": ["Estimates include active balancing"]
}
```

### Check Budget Status
```bash
GET /rewards/buckets/status
```

### Interactive Calculator
```bash
GET /rewards/calculator?scenario=inference_hour&tokens_per_hour=100000
```

## 📈 Earnings Examples

### Inference Node (8 hours/day)
- Processes: 800,000 tokens/day
- Quality: 1.0 average
- **Daily Earnings**: ~800 BLY

### Learning Contributor (1 session/day)
- Achieves: 3% improvement
- Difficulty: 1.5 (moderate)
- **Daily Earnings**: ~2,250 BLY

### Validation Node (continuous)
- Completes: 200 tasks/day
- **Daily Earnings**: ~2,000 BLY

### Dataset Provider (weekly)
- Contributes: 50 GB quality data
- Quality: 0.9, Diversity: 1.1
- **Weekly Earnings**: ~4,950 BLY

## 🛡️ Anti-Gaming Measures

1. **Statistical Validation**: All improvements must pass confidence thresholds
2. **Verification Nodes**: Multiple nodes verify claims independently
3. **Challenge Period**: 24-hour window for disputing rewards
4. **Rate Limiting**: Maximum claims per hour per node
5. **Diversity Requirements**: Rewards require diverse validation sources

## 📊 Monitoring & Metrics

### Prometheus Metrics
- `rewards_inference_bly_total`: Total inference rewards
- `rewards_learning_bly_total`: Total learning rewards
- `rewards_bucket_utilization`: Budget utilization by type
- `rewards_backpay_queue_size`: Pending reward claims

### Grafana Dashboards
- Budget utilization over time
- Reward distribution pie charts
- Backpay queue monitoring
- Per-unit reward rates

### Health Checks
```bash
GET /rewards/health
```

Returns system health and any active issues.

## 🔄 Distribution Process

### Hourly Automatic Distribution
1. Claims submitted throughout the hour
2. Idempotency check prevents duplicates
3. Budget allocation from appropriate bucket
4. Backpay queue for exhausted budgets
5. Distribution at hour boundary
6. Permanent ledger recording

### Idempotency Guarantees
- Each claim has unique hash
- 48-hour idempotency window
- Replay protection for recovery
- Audit trail for all distributions

## 💡 Best Practices

### For Inference Nodes
- Focus on quality over quantity
- Maintain consistent uptime
- Monitor your quality scores

### For Learning Contributors
- Target high-difficulty improvements
- Ensure broad applicability
- Provide thorough validation

### For Validators
- Maintain high accuracy
- Process diverse task types
- Stay online during peak hours

### For Dataset Providers
- Ensure high quality scores
- Contribute diverse data
- Follow licensing requirements

## 🚀 Getting Started

### 1. Check Current Rates
```python
from backend.rewards.policy import RewardConfig

config = RewardConfig.from_yaml()
print(f"Per 1k tokens: {config.per_1k_tokens_bly} BLY")
print(f"Per 1% improvement: {config.per_1pct_improvement_bly} BLY")
```

### 2. Estimate Your Earnings
```python
from backend.rewards.policy import calc_inference_bly

# Estimate for 1 million tokens at quality 1.2
reward = calc_inference_bly(1_000_000, quality=1.2)
print(f"Estimated reward: {reward} BLY")
```

### 3. Submit Claims
```python
from backend.rewards.distributor import AutomaticRewardDistributor

distributor = AutomaticRewardDistributor()
claim_id = await distributor.submit_claim(
    'inference',
    'your_wallet_address',
    {'tokens': 100000, 'quality': 1.1}
)
```

## 📝 Configuration

All reward parameters are configured in `config/reward_policy.yaml`:

```yaml
rates:
  per_1k_tokens_bly: 1.0
  per_1pct_improvement_bly: 500.0
  per_validation_task_bly: 10.0
  per_gb_dataset_bly: 100.0

bucket_split:
  inference: 0.45
  learning: 0.35
  validation: 0.10
  dataset: 0.10
```

## 🔍 Troubleshooting

### Low Rewards
- Check quality scores
- Verify budget availability
- Review backpay queue status

### Delayed Payments
- Distributions occur hourly
- Check backpay queue if budget exhausted
- Verify claim was accepted

### Quality Issues
- Ensure work meets validation criteria
- Check confidence thresholds
- Review validator feedback

## 📚 Additional Resources

- [Tokenomics Whitepaper](../TOKENOMICS.md)
- [API Reference](./API_REFERENCE.md)
- [Validation Guide](./VALIDATION_GUIDE.md)
- [Dataset Standards](./DATASET_STANDARDS.md)
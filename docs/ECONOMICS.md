# Economic Model

## Fee Structure (EIP-1559 Style)

### Base Fee Algorithm
```
newBaseFee = oldBaseFee * (1 + α * (gasUsed - gasTarget) / gasTarget)
α = 0.125 (max 12.5% change per block)
gasTarget = blockGasLimit / 2
minBaseFee = 1 gwei
```

### Fee Distribution
- **Base Fee**: 100% burned (deflationary pressure)
- **Priority Fee (Tip)**: 100% to block proposer
- **Total Fee**: baseFee + priorityFee

## Token Supply Dynamics

### Minting
- **Source**: PoL RewardReceipts only
- **Rate**: Variable based on network activity
- **Cap**: 10% annual inflation maximum

### Burning
- **Base fees**: All transaction base fees burned
- **Slashing**: 50% burned, 50% to reporter
- **Failed claims**: Gas fees burned (no mint)

## Validator Economics

### Staking Requirements
- **Minimum**: 10,000 BLY
- **Maximum**: 1,000,000 BLY (anti-centralization)
- **Unbonding**: 21 days

### Rewards
- **Block rewards**: Priority fees only
- **PoL bonus**: Validators running inference nodes get 2x weight
- **Commission**: 5-20% on delegator rewards

### Slashing Penalties
| Violation | Penalty | Distribution |
|-----------|---------|--------------|
| Double-sign | 5% stake | 50% burn, 50% reporter |
| Downtime | 0.01% per hour | 100% burn |
| Censorship | 1% stake | 100% burn |
| Invalid proposal | 0.1% stake | 100% burn |

## Inclusion List Economics
- **Size**: Max 10 transactions per proposer
- **Priority**: Must include within 2 blocks
- **Penalty**: 1% stake for non-inclusion
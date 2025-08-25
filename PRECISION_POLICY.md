# Precision Policy for Blyan

## Core Principle: BF16 for All Learning and Inference

All models involved in learning and inference MUST use **BF16 (bfloat16)** precision to ensure numerical consistency across the distributed network.

## Requirements

### Hardware Requirements
- **GPU**: NVIDIA Ampere or newer (Compute Capability 8.0+)
- **Examples**: RTX 30xx, RTX 40xx, A100, H100
- **NOT Supported**: Older GPUs (V100, RTX 20xx, GTX series)

### Precision by Component

| Component | Precision | Reason |
|-----------|-----------|---------|
| **Student Models** | BF16 | Numerical consistency in distributed learning |
| **Inference Models** | BF16 | All nodes must produce identical outputs |
| **Learning/Training** | BF16 | Gradient consistency across nodes |
| **Blockchain Weights** | BF16 | Storage and reconstruction consistency |
| **Teacher Models** | INT8* | Exception - see below |

## ⚠️ EXCEPTION: Teacher Models

Teacher models are the **ONLY** exception to the BF16 rule.

### Why INT8 for Teacher Models?
1. **Frozen Checkpoints**: Teacher models are N-1 generation, never updated
2. **Validation Only**: Used solely for quality gating, not learning
3. **Performance**: 4x faster inference, 4x less memory
4. **Inclusivity**: Allows older GPUs to participate as validators
5. **No Learning Impact**: Does not affect distributed learning consistency

### Teacher Model Guidelines
- Teacher models MUST be clearly marked as INT8 in code
- Validation thresholds should account for INT8 vs BF16 differences
- Older GPUs can ONLY run teacher models, not student/inference models

## Implementation

### Correct Implementation
```python
# Student/Inference/Learning models - BF16 ONLY
if torch.cuda.get_device_capability()[0] < 8:
    raise RuntimeError("GPU does not support BF16. Minimum Ampere (CC 8.0) required.")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # NO FALLBACKS
    device_map="auto"
)
```

### Teacher Model Exception
```python
# Teacher models - INT8 allowed
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,  # OK for teacher only
    llm_int8_threshold=6.0
)

teacher_model = AutoModelForCausalLM.from_pretrained(
    "teacher_model_v17",
    quantization_config=quantization_config  # INT8 for validation
)
```

## Enforcement

1. **No Fallbacks**: Code must fail fast if BF16 not supported
2. **No Mixed Precision**: All computations in BF16 (except teacher)
3. **No CPU Mode**: BF16 requires GPU support
4. **Clear Errors**: Explicitly state GPU requirements in error messages

## Benefits

- **Numerical Consistency**: All nodes compute with identical precision
- **Deterministic Results**: Same inputs produce same outputs everywhere
- **Simplified Debugging**: No precision-related discrepancies
- **Future Proof**: BF16 is the standard for modern AI workloads

## Migration Path

For nodes with older GPUs:
1. Can still participate as **validators** using INT8 teacher models
2. Cannot participate in learning or inference
3. Should plan GPU upgrade to Ampere or newer for full participation

---

*Last Updated: 2024*
*Policy Version: 1.0*
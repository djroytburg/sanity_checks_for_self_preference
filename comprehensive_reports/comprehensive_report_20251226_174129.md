# Comprehensive Self-Preference Analysis Report

**Generated**: 2025-12-26 17:41:29 EST

---

## Dataset: ALPACA_EVAL

### Judge: `Llama-3.1-8B-Instruct`

### Reference: `Llama-3.1-70B-Instruct`

**Null Test Decision** (across all 13 proxies):
- Proxies with p ≥ 0.05: 7/13
- Passing proxies: Qwen2.5-32B-Instruct, Qwen2.5-72B-Instruct, Qwen2.5-14B-Instruct ...
  - Proxy `Qwen2.5-32B-Instruct`: t-test p = 6.5879e-01, one-sided p = 3.2940e-01, KS p = 7.0988e-01, Sample Size = 82
  - Proxy `Qwen2.5-72B-Instruct`: t-test p = 4.2492e-03, one-sided p = 2.1246e-03, KS p = 1.2449e-01, Sample Size = 81
  - Proxy `Qwen2.5-14B-Instruct`: t-test p = 2.2706e-01, one-sided p = 8.8647e-01, KS p = 6.4785e-01, Sample Size = 74
  - Proxy `glm-4-plus`: t-test p = 6.1745e-01, one-sided p = 6.9128e-01, KS p = 9.2565e-01, Sample Size = 41
  - Proxy `qwen-plus`: t-test p = 1.2597e-01, one-sided p = 9.3701e-01, KS p = 4.1355e-01, Sample Size = 31
  - Proxy `claude-3.5-haiku`: t-test p = 5.9199e-01, one-sided p = 7.0401e-01, KS p = 8.0806e-01, Sample Size = 19
  - Proxy `QwQ-32B`: t-test p = 9.4590e-01, one-sided p = 4.7295e-01, KS p = 4.1752e-01, Sample Size = 10
- **Decision**: ✅ **FAIL TO REJECT** H₀ (α = 0.05)
  - *At least one proxy shows no significant difference between J and K*

#### Self-Recognition vs Self-Preference Correlation
**Overall Correlation** (n=319):
- Pearson r = 0.7212, p = 1.8101e-52
- Spearman ρ = 0.6839, p = 2.6056e-45

**Stratified by LSP/ILSP**:
- **LSP** (n=70):
  - Pearson r = 0.7913, p = 3.5636e-16
  - Spearman ρ = 0.7592, p = 2.5958e-14
- **ILSP** (n=249):
  - Pearson r = 0.5519, p = 2.9833e-21
  - Spearman ρ = 0.5671, p = 1.3624e-22

**Correlation Validity Check**:
- Overall Pearson r (0.7212) ≥ ILSP r (0.5519): ✅
- Overall Spearman ρ (0.6839) ≥ ILSP ρ (0.5671): ✅
- **Decision**: VALID ✅
  - *Overall correlation is at least as strong as ILSP-only correlation*

---

### Reference: `Llama-3.1-8B`

**Null Test Decision** (across all 3 proxies):
- Proxies with p ≥ 0.05: 2/3
- Passing proxies: Qwen2.5-7B, Llama-3.1-70B
  - Proxy `Qwen2.5-7B`: t-test p = 8.0570e-03, one-sided p = 4.0285e-03, KS p = 1.5317e-01, Sample Size = 19
  - Proxy `Llama-3.1-70B`: t-test p = 1.1015e-03, one-sided p = 5.5075e-04, KS p = 1.1238e-01, Sample Size = 17
- **Decision**: ✅ **FAIL TO REJECT** H₀ (α = 0.05)
  - *At least one proxy shows no significant difference between J and K*

#### Self-Recognition vs Self-Preference Correlation
**Overall Correlation** (n=334):
- Pearson r = 0.6660, p = 3.5691e-44
- Spearman ρ = 0.6675, p = 1.9520e-44

**Stratified by LSP/ILSP**:
- **LSP** (n=297):
  - Pearson r = 0.5996, p = 2.2247e-30
  - Spearman ρ = 0.6198, p = 6.6274e-33
- **ILSP** (n=37):
  - Pearson r = 0.5256, p = 8.3552e-04
  - Spearman ρ = 0.5453, p = 4.8291e-04

**Correlation Validity Check**:
- Overall Pearson r (0.6660) ≥ ILSP r (0.5256): ✅
- Overall Spearman ρ (0.6675) ≥ ILSP ρ (0.5453): ✅
- **Decision**: VALID ✅
  - *Overall correlation is at least as strong as ILSP-only correlation*

---

### Reference: `Qwen2.5-7B-Instruct`

**Null Test Decision** (across all 4 proxies):
- Proxies with p ≥ 0.05: 2/4
- Passing proxies: Llama-3.1-70B-Instruct, Qwen2.5-72B-Instruct
  - Proxy `Llama-3.1-70B-Instruct`: t-test p = 3.9583e-01, one-sided p = 8.0208e-01, KS p = 7.3029e-01, Sample Size = 105
  - Proxy `Qwen2.5-72B-Instruct`: t-test p = 4.2881e-01, one-sided p = 7.8560e-01, KS p = 5.7416e-01, Sample Size = 52
- **Decision**: ✅ **FAIL TO REJECT** H₀ (α = 0.05)
  - *At least one proxy shows no significant difference between J and K*

#### Self-Recognition vs Self-Preference Correlation
**Overall Correlation** (n=355):
- Pearson r = 0.7563, p = 4.9102e-67
- Spearman ρ = 0.7450, p = 4.8527e-64

**Stratified by LSP/ILSP**:
- **LSP** (n=157):
  - Pearson r = 0.6044, p = 5.2024e-17
  - Spearman ρ = 0.5930, p = 2.7741e-16
- **ILSP** (n=198):
  - Pearson r = 0.5508, p = 4.1809e-17
  - Spearman ρ = 0.5140, p = 9.5641e-15

**Correlation Validity Check**:
- Overall Pearson r (0.7563) ≥ ILSP r (0.5508): ✅
- Overall Spearman ρ (0.7450) ≥ ILSP ρ (0.5140): ✅
- **Decision**: VALID ✅
  - *Overall correlation is at least as strong as ILSP-only correlation*

---

### Reference: `gemma-2-9b-it`

**Null Test Decision** (across all 1 proxies):
- Proxies with p ≥ 0.05: 0/1
- **Decision**: ❌ **REJECT** H₀ (α = 0.05)
  - *All proxies show significant difference between J and K*

#### Self-Recognition vs Self-Preference Correlation
**Overall Correlation** (n=184):
- Pearson r = 0.6589, p = 2.8156e-24
- Spearman ρ = 0.6509, p = 1.4936e-23

**Stratified by LSP/ILSP**:
- **LSP** (n=31):
  - Pearson r = 0.4383, p = 1.3644e-02
  - Spearman ρ = 0.5485, p = 1.3993e-03
- **ILSP** (n=153):
  - Pearson r = 0.5776, p = 5.4184e-15
  - Spearman ρ = 0.5669, p = 2.1983e-14

**Correlation Validity Check**:
- Overall Pearson r (0.6589) ≥ ILSP r (0.5776): ✅
- Overall Spearman ρ (0.6509) ≥ ILSP ρ (0.5669): ✅
- **Decision**: VALID ✅
  - *Overall correlation is at least as strong as ILSP-only correlation*

---

### Judge: `Qwen2.5-0.5B-Instruct`

### Reference: `Llama-3.1-70B-Instruct`

**Null Test Decision** (across all 11 proxies):
- Proxies with p ≥ 0.05: 10/11
- Passing proxies: Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, DeepSeek-R1-Distill-Qwen-32B ...
  - Proxy `Llama-3.1-8B-Instruct`: t-test p = 1.5918e-04, one-sided p = 9.9992e-01, KS p = 1.7173e-04, Sample Size = 237
  - Proxy `Qwen2.5-7B-Instruct`: t-test p = 2.6492e-02, one-sided p = 9.8675e-01, KS p = 2.8138e-01, Sample Size = 225
  - Proxy `DeepSeek-R1-Distill-Qwen-32B`: t-test p = 7.6292e-02, one-sided p = 9.6185e-01, KS p = 2.8053e-01, Sample Size = 165
  - Proxy `Qwen2.5-72B-Instruct`: t-test p = 4.7814e-01, one-sided p = 7.6093e-01, KS p = 5.7456e-01, Sample Size = 138
  - Proxy `Qwen2.5-14B-Instruct`: t-test p = 7.7020e-01, one-sided p = 3.8510e-01, KS p = 9.5053e-01, Sample Size = 118
  - Proxy `Qwen2.5-32B-Instruct`: t-test p = 1.7775e-02, one-sided p = 9.9111e-01, KS p = 6.3457e-02, Sample Size = 116
  - Proxy `glm-4-plus`: t-test p = 1.4169e-01, one-sided p = 9.2916e-01, KS p = 4.2788e-01, Sample Size = 65
  - Proxy `qwen-plus`: t-test p = 6.8284e-01, one-sided p = 3.4142e-01, KS p = 3.7070e-01, Sample Size = 48
  - Proxy `claude-3.5-haiku`: t-test p = 2.0290e-01, one-sided p = 8.9855e-01, KS p = 7.3274e-01, Sample Size = 26
  - Proxy `QwQ-32B`: t-test p = 6.0432e-01, one-sided p = 3.0216e-01, KS p = 6.7814e-01, Sample Size = 15
- **Decision**: ✅ **FAIL TO REJECT** H₀ (α = 0.05)
  - *At least one proxy shows no significant difference between J and K*

#### Self-Recognition vs Self-Preference Correlation
**Overall Correlation** (n=430):
- Pearson r = 0.5779, p = 1.1192e-39
- Spearman ρ = 0.5802, p = 4.7538e-40

**Stratified by LSP/ILSP**:
- **LSP** (n=8):
  - Pearson r = 0.8704, p = 4.9314e-03
  - Spearman ρ = 0.7665, p = 2.6520e-02
- **ILSP** (n=422):
  - Pearson r = 0.5708, p = 7.3888e-38
  - Spearman ρ = 0.5705, p = 8.4165e-38

**Correlation Validity Check**:
- Overall Pearson r (0.5779) ≥ ILSP r (0.5708): ✅
- Overall Spearman ρ (0.5802) ≥ ILSP ρ (0.5705): ✅
- **Decision**: VALID ✅
  - *Overall correlation is at least as strong as ILSP-only correlation*

---

### Judge: `Qwen2.5-1.5B-Instruct`

### Reference: `Llama-3.1-70B-Instruct`

**Null Test Decision** (across all 12 proxies):
- Proxies with p ≥ 0.05: 12/12
- Passing proxies: Llama-3.1-70B, Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct ...
  - Proxy `Llama-3.1-70B`: t-test p = 2.9308e-01, one-sided p = 1.4654e-01, KS p = 1.6888e-01, Sample Size = 233
  - Proxy `Qwen2.5-3B-Instruct`: t-test p = 7.9899e-01, one-sided p = 3.9950e-01, KS p = 9.2565e-02, Sample Size = 220
  - Proxy `Qwen2.5-7B-Instruct`: t-test p = 9.1374e-01, one-sided p = 4.5687e-01, KS p = 3.1637e-02, Sample Size = 176
  - Proxy `Llama-3.1-8B-Instruct`: t-test p = 3.6763e-01, one-sided p = 8.1619e-01, KS p = 7.3770e-04, Sample Size = 174
  - Proxy `DeepSeek-R1-Distill-Qwen-32B`: t-test p = 6.3536e-01, one-sided p = 6.8232e-01, KS p = 6.8915e-02, Sample Size = 131
  - Proxy `Qwen2.5-72B-Instruct`: t-test p = 2.2395e-01, one-sided p = 1.1197e-01, KS p = 2.3350e-02, Sample Size = 109
  - Proxy `Qwen2.5-14B-Instruct`: t-test p = 6.5746e-03, one-sided p = 9.9671e-01, KS p = 2.2199e-03, Sample Size = 100
  - Proxy `Qwen2.5-32B-Instruct`: t-test p = 2.4584e-03, one-sided p = 9.9877e-01, KS p = 4.7914e-06, Sample Size = 91
  - Proxy `glm-4-plus`: t-test p = 5.4242e-02, one-sided p = 9.7288e-01, KS p = 4.6788e-03, Sample Size = 54
  - Proxy `qwen-plus`: t-test p = 5.7018e-02, one-sided p = 9.7149e-01, KS p = 6.7430e-04, Sample Size = 46
  - Proxy `claude-3.5-haiku`: t-test p = 3.6477e-02, one-sided p = 9.8176e-01, KS p = 1.7453e-01, Sample Size = 20
  - Proxy `QwQ-32B`: t-test p = 8.8476e-01, one-sided p = 4.4238e-01, KS p = 6.7814e-01, Sample Size = 15
- **Decision**: ✅ **FAIL TO REJECT** H₀ (α = 0.05)
  - *At least one proxy shows no significant difference between J and K*

#### Self-Recognition vs Self-Preference Correlation
**Overall Correlation** (n=353):
- Pearson r = 0.7020, p = 1.1203e-53
- Spearman ρ = 0.7316, p = 2.3290e-60

**Stratified by LSP/ILSP**:
- **LSP** (n=49):
  - Pearson r = 0.7663, p = 1.3943e-10
  - Spearman ρ = 0.7754, p = 6.0900e-11
- **ILSP** (n=304):
  - Pearson r = 0.6848, p = 2.1268e-43
  - Spearman ρ = 0.7114, p = 3.5560e-48

**Correlation Validity Check**:
- Overall Pearson r (0.7020) ≥ ILSP r (0.6848): ✅
- Overall Spearman ρ (0.7316) ≥ ILSP ρ (0.7114): ✅
- **Decision**: VALID ✅
  - *Overall correlation is at least as strong as ILSP-only correlation*

---

### Judge: `Qwen2.5-14B-Instruct`

### Reference: `Llama-3.1-70B-Instruct`

**Null Test Decision** (across all 13 proxies):
- Proxies with p ≥ 0.05: 8/13
- Passing proxies: DeepSeek-R1-Distill-Qwen-32B, Llama-3.1-8B-Instruct, Qwen2.5-72B-Instruct ...
  - Proxy `DeepSeek-R1-Distill-Qwen-32B`: t-test p = 4.8645e-01, one-sided p = 2.4323e-01, KS p = 9.2862e-01, Sample Size = 82
  - Proxy `Llama-3.1-8B-Instruct`: t-test p = 5.3740e-03, one-sided p = 2.6870e-03, KS p = 6.2663e-02, Sample Size = 74
  - Proxy `Qwen2.5-72B-Instruct`: t-test p = 6.1092e-01, one-sided p = 3.0546e-01, KS p = 3.7220e-01, Sample Size = 72
  - Proxy `Qwen2.5-32B-Instruct`: t-test p = 7.9791e-02, one-sided p = 9.6010e-01, KS p = 2.0411e-01, Sample Size = 63
  - Proxy `glm-4-plus`: t-test p = 3.6155e-01, one-sided p = 8.1922e-01, KS p = 5.1001e-01, Sample Size = 36
  - Proxy `qwen-plus`: t-test p = 1.6245e-02, one-sided p = 9.9188e-01, KS p = 2.8992e-01, Sample Size = 33
  - Proxy `claude-3.5-haiku`: t-test p = 4.0647e-01, one-sided p = 7.9677e-01, KS p = 5.0259e-01, Sample Size = 18
  - Proxy `QwQ-32B`: t-test p = 5.6466e-01, one-sided p = 2.8233e-01, KS p = 8.6898e-01, Sample Size = 12
- **Decision**: ✅ **FAIL TO REJECT** H₀ (α = 0.05)
  - *At least one proxy shows no significant difference between J and K*

#### Self-Recognition vs Self-Preference Correlation
**Overall Correlation** (n=358):
- Pearson r = 0.6196, p = 2.4249e-39
- Spearman ρ = 0.6920, p = 2.5026e-52

**Stratified by LSP/ILSP**:
- **LSP** (n=239):
  - Pearson r = 0.2651, p = 3.2956e-05
  - Spearman ρ = 0.4403, p = 9.4302e-13
- **ILSP** (n=119):
  - Pearson r = 0.5125, p = 2.5502e-09
  - Spearman ρ = 0.5468, p = 1.2488e-10

**Correlation Validity Check**:
- Overall Pearson r (0.6196) ≥ ILSP r (0.5125): ✅
- Overall Spearman ρ (0.6920) ≥ ILSP ρ (0.5468): ✅
- **Decision**: VALID ✅
  - *Overall correlation is at least as strong as ILSP-only correlation*

---

### Judge: `Qwen2.5-32B-Instruct`

### Reference: `Llama-3.1-70B-Instruct`

**Null Test Decision** (across all 13 proxies):
- Proxies with p ≥ 0.05: 7/13
- Passing proxies: Llama-3.1-8B-Instruct, Qwen2.5-72B-Instruct, Qwen2.5-14B-Instruct ...
  - Proxy `Llama-3.1-8B-Instruct`: t-test p = 5.0407e-04, one-sided p = 2.5204e-04, KS p = 5.8714e-02, Sample Size = 82
  - Proxy `Qwen2.5-72B-Instruct`: t-test p = 2.0336e-02, one-sided p = 1.0168e-02, KS p = 6.2663e-02, Sample Size = 74
  - Proxy `Qwen2.5-14B-Instruct`: t-test p = 5.6363e-01, one-sided p = 2.8182e-01, KS p = 8.3572e-01, Sample Size = 63
  - Proxy `glm-4-plus`: t-test p = 2.0573e-02, one-sided p = 9.8971e-01, KS p = 4.0156e-02, Sample Size = 37
  - Proxy `qwen-plus`: t-test p = 7.4078e-01, one-sided p = 6.2961e-01, KS p = 9.9884e-01, Sample Size = 30
  - Proxy `claude-3.5-haiku`: t-test p = 1.4150e-01, one-sided p = 9.2925e-01, KS p = 5.3793e-01, Sample Size = 19
  - Proxy `QwQ-32B`: t-test p = 2.5160e-01, one-sided p = 1.2580e-01, KS p = 7.3011e-01, Sample Size = 9
- **Decision**: ✅ **FAIL TO REJECT** H₀ (α = 0.05)
  - *At least one proxy shows no significant difference between J and K*

#### Self-Recognition vs Self-Preference Correlation
*No results available*

---

### Judge: `Qwen2.5-3B-Instruct`

### Reference: `Llama-3.1-70B-Instruct`

**Null Test Decision** (across all 12 proxies):
- Proxies with p ≥ 0.05: 11/12
- Passing proxies: Llama-3.1-70B, Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct ...
  - Proxy `Llama-3.1-70B`: t-test p = 8.2272e-01, one-sided p = 4.1136e-01, KS p = 5.9210e-01, Sample Size = 215
  - Proxy `Qwen2.5-7B-Instruct`: t-test p = 4.8511e-02, one-sided p = 9.7574e-01, KS p = 9.6461e-02, Sample Size = 190
  - Proxy `Llama-3.1-8B-Instruct`: t-test p = 7.4163e-01, one-sided p = 3.7081e-01, KS p = 7.6744e-01, Sample Size = 162
  - Proxy `DeepSeek-R1-Distill-Qwen-32B`: t-test p = 6.2730e-01, one-sided p = 6.8635e-01, KS p = 4.1655e-01, Sample Size = 144
  - Proxy `Qwen2.5-72B-Instruct`: t-test p = 2.1484e-02, one-sided p = 9.8926e-01, KS p = 7.0049e-03, Sample Size = 120
  - Proxy `Qwen2.5-14B-Instruct`: t-test p = 8.5226e-03, one-sided p = 9.9574e-01, KS p = 5.1956e-02, Sample Size = 99
  - Proxy `Qwen2.5-32B-Instruct`: t-test p = 5.4757e-03, one-sided p = 9.9726e-01, KS p = 4.2142e-03, Sample Size = 94
  - Proxy `glm-4-plus`: t-test p = 5.5483e-02, one-sided p = 9.7226e-01, KS p = 5.1665e-03, Sample Size = 49
  - Proxy `qwen-plus`: t-test p = 1.7176e-03, one-sided p = 9.9914e-01, KS p = 3.5169e-02, Sample Size = 42
  - Proxy `claude-3.5-haiku`: t-test p = 2.0439e-01, one-sided p = 8.9781e-01, KS p = 5.7134e-01, Sample Size = 20
  - Proxy `QwQ-32B`: t-test p = 7.3988e-01, one-sided p = 6.3006e-01, KS p = 9.3833e-01, Sample Size = 15
- **Decision**: ✅ **FAIL TO REJECT** H₀ (α = 0.05)
  - *At least one proxy shows no significant difference between J and K*

#### Self-Recognition vs Self-Preference Correlation
**Overall Correlation** (n=380):
- Pearson r = 0.3845, p = 7.8156e-15
- Spearman ρ = 0.5295, p = 7.6744e-29

**Stratified by LSP/ILSP**:
- **LSP** (n=99):
  - Pearson r = 0.2534, p = 1.1370e-02
  - Spearman ρ = 0.3590, p = 2.6278e-04
- **ILSP** (n=281):
  - Pearson r = 0.3711, p = 1.3229e-10
  - Spearman ρ = 0.5144, p = 2.2106e-20

**Correlation Validity Check**:
- Overall Pearson r (0.3845) ≥ ILSP r (0.3711): ✅
- Overall Spearman ρ (0.5295) ≥ ILSP ρ (0.5144): ✅
- **Decision**: VALID ✅
  - *Overall correlation is at least as strong as ILSP-only correlation*

---

### Judge: `Qwen2.5-7B-Instruct`

### Reference: `DeepSeek-R1-Distill-Qwen-7B`

**Null Test Decision** (across all 1 proxies):
- Proxies with p ≥ 0.05: 1/1
- Passing proxies: Qwen2.5-7B
  - Proxy `Qwen2.5-7B`: t-test p = 1.3994e-01, one-sided p = 6.9972e-02, KS p = 3.0351e-02, Sample Size = 47
- **Decision**: ✅ **FAIL TO REJECT** H₀ (α = 0.05)
  - *At least one proxy shows no significant difference between J and K*

#### Self-Recognition vs Self-Preference Correlation
**Overall Correlation** (n=258):
- Pearson r = 0.7874, p = 1.0484e-55
- Spearman ρ = 0.8879, p = 2.5788e-88

**Stratified by LSP/ILSP**:
- **LSP** (n=211):
  - Pearson r = 0.7542, p = 4.8908e-40
  - Spearman ρ = 0.8778, p = 1.0115e-68
- **ILSP** (n=47):
  - Pearson r = 0.3580, p = 1.3479e-02
  - Spearman ρ = 0.5558, p = 4.9976e-05

**Correlation Validity Check**:
- Overall Pearson r (0.7874) ≥ ILSP r (0.3580): ✅
- Overall Spearman ρ (0.8879) ≥ ILSP ρ (0.5558): ✅
- **Decision**: VALID ✅
  - *Overall correlation is at least as strong as ILSP-only correlation*

---

### Reference: `Llama-3.1-70B-Instruct`

**Null Test Decision** (across all 13 proxies):
- Proxies with p ≥ 0.05: 11/13
- Passing proxies: Qwen2.5-3B-Instruct, Llama-3.1-70B, Llama-3.1-8B-Instruct ...
  - Proxy `Qwen2.5-3B-Instruct`: t-test p = 6.4794e-02, one-sided p = 3.2397e-02, KS p = 1.5666e-01, Sample Size = 190
  - Proxy `Llama-3.1-70B`: t-test p = 1.0938e-02, one-sided p = 5.4692e-03, KS p = 5.0499e-02, Sample Size = 170
  - Proxy `Llama-3.1-8B-Instruct`: t-test p = 2.7144e-01, one-sided p = 8.6428e-01, KS p = 4.5407e-01, Sample Size = 133
  - Proxy `DeepSeek-R1-Distill-Qwen-32B`: t-test p = 2.1679e-01, one-sided p = 8.9160e-01, KS p = 4.3450e-01, Sample Size = 129
  - Proxy `Qwen2.5-72B-Instruct`: t-test p = 1.1715e-02, one-sided p = 9.9414e-01, KS p = 2.9707e-01, Sample Size = 118
  - Proxy `Qwen2.5-14B-Instruct`: t-test p = 1.1632e-05, one-sided p = 9.9999e-01, KS p = 4.5015e-03, Sample Size = 95
  - Proxy `Qwen2.5-32B-Instruct`: t-test p = 1.5436e-04, one-sided p = 9.9992e-01, KS p = 1.1605e-02, Sample Size = 86
  - Proxy `glm-4-plus`: t-test p = 3.2961e-03, one-sided p = 9.9835e-01, KS p = 2.1792e-01, Sample Size = 45
  - Proxy `qwen-plus`: t-test p = 2.1463e-01, one-sided p = 8.9269e-01, KS p = 5.2789e-01, Sample Size = 37
  - Proxy `claude-3.5-haiku`: t-test p = 9.6571e-01, one-sided p = 5.1715e-01, KS p = 9.2052e-01, Sample Size = 14
  - Proxy `QwQ-32B`: t-test p = 4.2037e-01, one-sided p = 2.1018e-01, KS p = 9.2052e-01, Sample Size = 14
- **Decision**: ✅ **FAIL TO REJECT** H₀ (α = 0.05)
  - *At least one proxy shows no significant difference between J and K*

#### Self-Recognition vs Self-Preference Correlation
**Overall Correlation** (n=369):
- Pearson r = 0.6708, p = 1.4422e-49
- Spearman ρ = 0.7197, p = 3.9923e-60

**Stratified by LSP/ILSP**:
- **LSP** (n=133):
  - Pearson r = 0.6092, p = 7.2106e-15
  - Spearman ρ = 0.5511, p = 6.2740e-12
- **ILSP** (n=236):
  - Pearson r = 0.6142, p = 7.2501e-26
  - Spearman ρ = 0.6761, p = 6.9806e-33

**Correlation Validity Check**:
- Overall Pearson r (0.6708) ≥ ILSP r (0.6142): ✅
- Overall Spearman ρ (0.7197) ≥ ILSP ρ (0.6761): ✅
- **Decision**: VALID ✅
  - *Overall correlation is at least as strong as ILSP-only correlation*

---

### Reference: `Llama-3.1-8B-Instruct`

**Null Test Decision** (across all 4 proxies):
- Proxies with p ≥ 0.05: 3/4
- Passing proxies: gemma-2-9b-it, Llama-3.1-Tulu-3-8B, Llama-3.1-70B-Instruct
  - Proxy `gemma-2-9b-it`: t-test p = 5.7803e-02, one-sided p = 2.8902e-02, KS p = 2.2157e-01, Sample Size = 89
  - Proxy `Llama-3.1-Tulu-3-8B`: t-test p = 9.4232e-01, one-sided p = 4.7116e-01, KS p = 4.5062e-01, Sample Size = 43
  - Proxy `Llama-3.1-70B-Instruct`: t-test p = 8.7053e-01, one-sided p = 5.6474e-01, KS p = 8.6323e-01, Sample Size = 34
- **Decision**: ✅ **FAIL TO REJECT** H₀ (α = 0.05)
  - *At least one proxy shows no significant difference between J and K*

#### Self-Recognition vs Self-Preference Correlation
**Overall Correlation** (n=343):
- Pearson r = 0.7159, p = 3.8078e-55
- Spearman ρ = 0.7370, p = 5.7752e-60

**Stratified by LSP/ILSP**:
- **LSP** (n=191):
  - Pearson r = 0.7006, p = 1.6437e-29
  - Spearman ρ = 0.6638, p = 1.2505e-25
- **ILSP** (n=152):
  - Pearson r = 0.6194, p = 1.8053e-17
  - Spearman ρ = 0.6846, p = 2.3579e-22

**Correlation Validity Check**:
- Overall Pearson r (0.7159) ≥ ILSP r (0.6194): ✅
- Overall Spearman ρ (0.7370) ≥ ILSP ρ (0.6846): ✅
- **Decision**: VALID ✅
  - *Overall correlation is at least as strong as ILSP-only correlation*

---

### Reference: `Qwen2.5-72B-Instruct`

**Null Test Decision** (across all 2 proxies):
- Proxies with p ≥ 0.05: 1/2
- Passing proxies: Llama-3.1-70B-Instruct
  - Proxy `Llama-3.1-70B-Instruct`: t-test p = 1.1799e-01, one-sided p = 5.8994e-02, KS p = 1.5301e-01, Sample Size = 126
- **Decision**: ✅ **FAIL TO REJECT** H₀ (α = 0.05)
  - *At least one proxy shows no significant difference between J and K*

#### Self-Recognition vs Self-Preference Correlation
**Overall Correlation** (n=253):
- Pearson r = 0.6098, p = 3.7331e-27
- Spearman ρ = 0.7271, p = 6.9418e-43

**Stratified by LSP/ILSP**:
- **LSP** (n=40):
  - Pearson r = 0.5761, p = 1.0017e-04
  - Spearman ρ = 0.6792, p = 1.4496e-06
- **ILSP** (n=213):
  - Pearson r = 0.5423, p = 1.1031e-17
  - Spearman ρ = 0.6856, p = 6.5110e-31

**Correlation Validity Check**:
- Overall Pearson r (0.6098) ≥ ILSP r (0.5423): ✅
- Overall Spearman ρ (0.7271) ≥ ILSP ρ (0.6856): ✅
- **Decision**: VALID ✅
  - *Overall correlation is at least as strong as ILSP-only correlation*

---

### Reference: `Qwen2.5-7B`

**Null Test Decision** (across all 3 proxies):
- Proxies with p ≥ 0.05: 2/3
- Passing proxies: Llama-3.1-8B, Qwen2.5-72B
  - Proxy `Llama-3.1-8B`: t-test p = 2.3598e-02, one-sided p = 1.1799e-02, KS p = 1.0591e-01, Sample Size = 49
  - Proxy `Qwen2.5-72B`: t-test p = 9.1515e-01, one-sided p = 4.5758e-01, KS p = 8.0293e-01, Sample Size = 43
- **Decision**: ✅ **FAIL TO REJECT** H₀ (α = 0.05)
  - *At least one proxy shows no significant difference between J and K*

#### Self-Recognition vs Self-Preference Correlation
**Overall Correlation** (n=293):
- Pearson r = 0.7589, p = 3.7245e-56
- Spearman ρ = 0.8118, p = 5.8755e-70

**Stratified by LSP/ILSP**:
- **LSP** (n=214):
  - Pearson r = 0.7125, p = 1.8601e-34
  - Spearman ρ = 0.7583, p = 2.8870e-41
- **ILSP** (n=79):
  - Pearson r = 0.6212, p = 1.0103e-09
  - Spearman ρ = 0.6501, p = 8.9459e-11

**Correlation Validity Check**:
- Overall Pearson r (0.7589) ≥ ILSP r (0.6212): ✅
- Overall Spearman ρ (0.8118) ≥ ILSP ρ (0.6501): ✅
- **Decision**: VALID ✅
  - *Overall correlation is at least as strong as ILSP-only correlation*

---

### Judge: `gemma-2-9b-it`

### Reference: `Llama-3.1-8B-Instruct`

**Null Test Decision** (across all 4 proxies):
- Proxies with p ≥ 0.05: 1/4
- Passing proxies: Llama-3.1-Tulu-3-8B
  - Proxy `Llama-3.1-Tulu-3-8B`: t-test p = 8.7849e-02, one-sided p = 4.3924e-02, KS p = 4.2018e-01, Sample Size = 41
- **Decision**: ✅ **FAIL TO REJECT** H₀ (α = 0.05)
  - *At least one proxy shows no significant difference between J and K*

#### Self-Recognition vs Self-Preference Correlation
**Overall Correlation** (n=308):
- Pearson r = 0.2037, p = 3.2030e-04
- Spearman ρ = 0.5930, p = 1.2195e-30

**Stratified by LSP/ILSP**:
- **LSP** (n=162):
  - Pearson r = 0.3182, p = 3.6882e-05
  - Spearman ρ = 0.3720, p = 1.0920e-06
- **ILSP** (n=146):
  - Pearson r = 0.1683, p = 4.2350e-02
  - Spearman ρ = 0.4386, p = 3.0719e-08

**Correlation Validity Check**:
- Overall Pearson r (0.2037) ≥ ILSP r (0.1683): ✅
- Overall Spearman ρ (0.5930) ≥ ILSP ρ (0.4386): ✅
- **Decision**: VALID ✅
  - *Overall correlation is at least as strong as ILSP-only correlation*

---

### Reference: `gemma-2-9b`

**Null Test Decision** (across all 1 proxies):
- Proxies with p ≥ 0.05: 0/1
- **Decision**: ❌ **REJECT** H₀ (α = 0.05)
  - *All proxies show significant difference between J and K*

#### Self-Recognition vs Self-Preference Correlation
**Overall Correlation** (n=154):
- Pearson r = 0.2466, p = 2.0492e-03
- Spearman ρ = 0.6526, p = 4.7654e-20

**Stratified by LSP/ILSP**:
- **LSP** (n=125):
  - Pearson r = 0.2117, p = 1.7783e-02
  - Spearman ρ = 0.5851, p = 7.7345e-13
- **ILSP** (n=29):
  - Pearson r = 0.1297, p = 5.0265e-01
  - Spearman ρ = 0.4180, p = 2.4033e-02

**Correlation Validity Check**:
- Overall Pearson r (0.2466) ≥ ILSP r (0.1297): ✅
- Overall Spearman ρ (0.6526) ≥ ILSP ρ (0.4180): ✅
- **Decision**: VALID ✅
  - *Overall correlation is at least as strong as ILSP-only correlation*

---


## Summary Statistics for ALPACA_EVAL

- **Total Judge-Reference Pairs**: 16
- **Judge Swap Null NOT Rejected** (✅): 14/16
- **Correlation VALID** (✅): 15/16

---


## Dataset: TRANSLATION

### Judge: `Llama-3.1-8B-Instruct`

### Reference: `Llama-3.1-70B-Instruct`

**Null Test Decision** (across all 2 proxies):
- Proxies with p ≥ 0.05: 2/2
- Passing proxies: Llama-3.1-70B, Qwen2.5-72B-Instruct
  - Proxy `Llama-3.1-70B`: t-test p = 6.2984e-04, one-sided p = 3.1492e-04, KS p = 5.0880e-02, Sample Size = 157
  - Proxy `Qwen2.5-72B-Instruct`: t-test p = 8.9593e-02, one-sided p = 9.5520e-01, KS p = 2.6384e-01, Sample Size = 71
- **Decision**: ✅ **FAIL TO REJECT** H₀ (α = 0.05)
  - *At least one proxy shows no significant difference between J and K*

#### Self-Recognition vs Self-Preference Correlation
**Overall Correlation** (n=227):
- Pearson r = 0.6301, p = 1.6261e-26
- Spearman ρ = 0.6826, p = 1.7540e-32

**Stratified by LSP/ILSP**:
- **LSP** (n=47):
  - Pearson r = 0.6088, p = 5.6208e-06
  - Spearman ρ = 0.5439, p = 7.7667e-05
- **ILSP** (n=180):
  - Pearson r = 0.4254, p = 2.6238e-09
  - Spearman ρ = 0.5464, p = 2.1127e-15

**Correlation Validity Check**:
- Overall Pearson r (0.6301) ≥ ILSP r (0.4254): ✅
- Overall Spearman ρ (0.6826) ≥ ILSP ρ (0.5464): ✅
- **Decision**: VALID ✅
  - *Overall correlation is at least as strong as ILSP-only correlation*

---

### Reference: `Llama-3.1-8B`

**Null Test Decision** (across all 3 proxies):
- Proxies with p ≥ 0.05: 3/3
- Passing proxies: Qwen2.5-7B, Llama-3.1-70B, gemma-2-9b
  - Proxy `Qwen2.5-7B`: t-test p = 1.0250e-01, one-sided p = 5.1248e-02, KS p = 3.7273e-01, Sample Size = 38
  - Proxy `Llama-3.1-70B`: t-test p = 4.4350e-01, one-sided p = 7.7825e-01, KS p = 5.3793e-01, Sample Size = 19
  - Proxy `gemma-2-9b`: t-test p = 3.3090e-02, one-sided p = 1.6545e-02, KS p = 5.6018e-02, Sample Size = 18
- **Decision**: ✅ **FAIL TO REJECT** H₀ (α = 0.05)
  - *At least one proxy shows no significant difference between J and K*

#### Self-Recognition vs Self-Preference Correlation
**Overall Correlation** (n=248):
- Pearson r = 0.6668, p = 2.9238e-33
- Spearman ρ = 0.6442, p = 1.8106e-30

**Stratified by LSP/ILSP**:
- **LSP** (n=202):
  - Pearson r = 0.5250, p = 1.0500e-15
  - Spearman ρ = 0.4968, p = 5.5005e-14
- **ILSP** (n=46):
  - Pearson r = 0.3947, p = 6.6326e-03
  - Spearman ρ = 0.3543, p = 1.5716e-02

**Correlation Validity Check**:
- Overall Pearson r (0.6668) ≥ ILSP r (0.3947): ✅
- Overall Spearman ρ (0.6442) ≥ ILSP ρ (0.3543): ✅
- **Decision**: VALID ✅
  - *Overall correlation is at least as strong as ILSP-only correlation*

---

### Reference: `Qwen2.5-7B-Instruct`

**Null Test Decision** (across all 2 proxies):
- Proxies with p ≥ 0.05: 1/2
- Passing proxies: Qwen2.5-72B-Instruct
  - Proxy `Qwen2.5-72B-Instruct`: t-test p = 4.5827e-01, one-sided p = 7.7087e-01, KS p = 1.0591e-01, Sample Size = 49
- **Decision**: ✅ **FAIL TO REJECT** H₀ (α = 0.05)
  - *At least one proxy shows no significant difference between J and K*

#### Self-Recognition vs Self-Preference Correlation
**Overall Correlation** (n=272):
- Pearson r = 0.7099, p = 5.2671e-43
- Spearman ρ = 0.6824, p = 1.2663e-38

**Stratified by LSP/ILSP**:
- **LSP** (n=103):
  - Pearson r = 0.5418, p = 3.4211e-09
  - Spearman ρ = 0.5259, p = 1.1697e-08
- **ILSP** (n=169):
  - Pearson r = 0.5284, p = 1.5336e-13
  - Spearman ρ = 0.4668, p = 1.5841e-10

**Correlation Validity Check**:
- Overall Pearson r (0.7099) ≥ ILSP r (0.5284): ✅
- Overall Spearman ρ (0.6824) ≥ ILSP ρ (0.4668): ✅
- **Decision**: VALID ✅
  - *Overall correlation is at least as strong as ILSP-only correlation*

---

### Reference: `gemma-2-9b-it`

**Null Test Decision** (across all 1 proxies):
- Proxies with p ≥ 0.05: 0/1
- **Decision**: ❌ **REJECT** H₀ (α = 0.05)
  - *All proxies show significant difference between J and K*

#### Self-Recognition vs Self-Preference Correlation
**Overall Correlation** (n=188):
- Pearson r = 0.5948, p = 2.2427e-19
- Spearman ρ = 0.5768, p = 4.5885e-18

**Stratified by LSP/ILSP**:
- **LSP** (n=13):
  - Pearson r = 0.4080, p = 1.6641e-01
  - Spearman ρ = 0.1538, p = 6.1580e-01
- **ILSP** (n=175):
  - Pearson r = 0.5056, p = 9.5683e-13
  - Spearman ρ = 0.5182, p = 2.0547e-13

**Correlation Validity Check**:
- Overall Pearson r (0.5948) ≥ ILSP r (0.5056): ✅
- Overall Spearman ρ (0.5768) ≥ ILSP ρ (0.5182): ✅
- **Decision**: VALID ✅
  - *Overall correlation is at least as strong as ILSP-only correlation*

---

### Judge: `Qwen2.5-7B-Instruct`

### Reference: `Llama-3.1-8B-Instruct`

**Null Test Decision** (across all 3 proxies):
- Proxies with p ≥ 0.05: 3/3
- Passing proxies: Llama-3.1-8B, gemma-2-9b-it, Llama-3.1-70B-Instruct
  - Proxy `Llama-3.1-8B`: t-test p = 1.0968e-01, one-sided p = 5.4838e-02, KS p = 4.7733e-01, Sample Size = 85
  - Proxy `gemma-2-9b-it`: t-test p = 5.1205e-01, one-sided p = 2.5602e-01, KS p = 4.9164e-01, Sample Size = 35
  - Proxy `Llama-3.1-70B-Instruct`: t-test p = 9.2031e-01, one-sided p = 4.6015e-01, KS p = 9.3566e-01, Sample Size = 27
- **Decision**: ✅ **FAIL TO REJECT** H₀ (α = 0.05)
  - *At least one proxy shows no significant difference between J and K*

#### Self-Recognition vs Self-Preference Correlation
**Overall Correlation** (n=280):
- Pearson r = 0.5099, p = 6.2142e-20
- Spearman ρ = 0.6141, p = 2.0678e-30

**Stratified by LSP/ILSP**:
- **LSP** (n=187):
  - Pearson r = 0.3503, p = 8.8861e-07
  - Spearman ρ = 0.4457, p = 1.6337e-10
- **ILSP** (n=93):
  - Pearson r = 0.3240, p = 1.5344e-03
  - Spearman ρ = 0.4262, p = 2.0522e-05

**Correlation Validity Check**:
- Overall Pearson r (0.5099) ≥ ILSP r (0.3240): ✅
- Overall Spearman ρ (0.6141) ≥ ILSP ρ (0.4262): ✅
- **Decision**: VALID ✅
  - *Overall correlation is at least as strong as ILSP-only correlation*

---

### Reference: `Qwen2.5-72B-Instruct`

**Null Test Decision** (across all 2 proxies):
- Proxies with p ≥ 0.05: 1/2
- Passing proxies: Llama-3.1-70B-Instruct
  - Proxy `Llama-3.1-70B-Instruct`: t-test p = 9.4611e-01, one-sided p = 5.2695e-01, KS p = 9.9554e-01, Sample Size = 142
- **Decision**: ✅ **FAIL TO REJECT** H₀ (α = 0.05)
  - *At least one proxy shows no significant difference between J and K*

#### Self-Recognition vs Self-Preference Correlation
**Overall Correlation** (n=240):
- Pearson r = 0.3675, p = 4.3503e-09
- Spearman ρ = 0.4348, p = 1.7278e-12

**Stratified by LSP/ILSP**:
- **LSP** (n=28):
  - Pearson r = 0.4529, p = 1.5516e-02
  - Spearman ρ = 0.4481, p = 1.6779e-02
- **ILSP** (n=212):
  - Pearson r = 0.2245, p = 9.9793e-04
  - Spearman ρ = 0.3006, p = 8.4009e-06

**Correlation Validity Check**:
- Overall Pearson r (0.3675) ≥ ILSP r (0.2245): ✅
- Overall Spearman ρ (0.4348) ≥ ILSP ρ (0.3006): ✅
- **Decision**: VALID ✅
  - *Overall correlation is at least as strong as ILSP-only correlation*

---

### Reference: `Qwen2.5-7B`

**Null Test Decision** (across all 2 proxies):
- Proxies with p ≥ 0.05: 2/2
- Passing proxies: Llama-3.1-8B, Qwen2.5-72B
  - Proxy `Llama-3.1-8B`: t-test p = 2.5480e-01, one-sided p = 1.2740e-01, KS p = 8.9706e-01, Sample Size = 54
  - Proxy `Qwen2.5-72B`: t-test p = 2.0633e-01, one-sided p = 8.9683e-01, KS p = 4.4904e-01, Sample Size = 24
- **Decision**: ✅ **FAIL TO REJECT** H₀ (α = 0.05)
  - *At least one proxy shows no significant difference between J and K*

#### Self-Recognition vs Self-Preference Correlation
**Overall Correlation** (n=268):
- Pearson r = 0.5646, p = 5.7437e-24
- Spearman ρ = 0.6601, p = 6.5250e-35

**Stratified by LSP/ILSP**:
- **LSP** (n=209):
  - Pearson r = 0.4602, p = 2.3767e-12
  - Spearman ρ = 0.5268, p = 2.5430e-16
- **ILSP** (n=59):
  - Pearson r = 0.0923, p = 4.8688e-01
  - Spearman ρ = 0.1823, p = 1.6700e-01

**Correlation Validity Check**:
- Overall Pearson r (0.5646) ≥ ILSP r (0.0923): ✅
- Overall Spearman ρ (0.6601) ≥ ILSP ρ (0.1823): ✅
- **Decision**: VALID ✅
  - *Overall correlation is at least as strong as ILSP-only correlation*

---

### Judge: `gemma-2-9b-it`

### Reference: `Llama-3.1-8B-Instruct`

**Null Test Decision** (across all 3 proxies):
- Proxies with p ≥ 0.05: 2/3
- Passing proxies: Qwen2.5-7B-Instruct, Llama-3.1-70B-Instruct
  - Proxy `Qwen2.5-7B-Instruct`: t-test p = 9.3263e-01, one-sided p = 4.6631e-01, KS p = 8.7449e-01, Sample Size = 35
  - Proxy `Llama-3.1-70B-Instruct`: t-test p = 5.9916e-01, one-sided p = 2.9958e-01, KS p = 9.9238e-01, Sample Size = 23
- **Decision**: ✅ **FAIL TO REJECT** H₀ (α = 0.05)
  - *At least one proxy shows no significant difference between J and K*

#### Self-Recognition vs Self-Preference Correlation
**Overall Correlation** (n=291):
- Pearson r = 0.5509, p = 1.7146e-24
- Spearman ρ = 0.6274, p = 3.0296e-33

**Stratified by LSP/ILSP**:
- **LSP** (n=236):
  - Pearson r = 0.3798, p = 1.6331e-09
  - Spearman ρ = 0.4822, p = 3.8300e-15
- **ILSP** (n=55):
  - Pearson r = 0.2642, p = 5.1295e-02
  - Spearman ρ = 0.2912, p = 3.0990e-02

**Correlation Validity Check**:
- Overall Pearson r (0.5509) ≥ ILSP r (0.2642): ✅
- Overall Spearman ρ (0.6274) ≥ ILSP ρ (0.2912): ✅
- **Decision**: VALID ✅
  - *Overall correlation is at least as strong as ILSP-only correlation*

---

### Reference: `gemma-2-9b`

**Null Test Decision** (across all 1 proxies):
- Proxies with p ≥ 0.05: 1/1
- Passing proxies: Llama-3.1-8B
  - Proxy `Llama-3.1-8B`: t-test p = 1.5345e-02, one-sided p = 7.6726e-03, KS p = 1.1230e-01, Sample Size = 42
- **Decision**: ✅ **FAIL TO REJECT** H₀ (α = 0.05)
  - *At least one proxy shows no significant difference between J and K*

#### Self-Recognition vs Self-Preference Correlation
**Overall Correlation** (n=103):
- Pearson r = 0.6248, p = 1.7554e-12
- Spearman ρ = 0.7433, p = 2.4404e-19

**Stratified by LSP/ILSP**:
- **LSP** (n=61):
  - Pearson r = 0.1977, p = 1.2674e-01
  - Spearman ρ = 0.4469, p = 3.0531e-04
- **ILSP** (n=42):
  - Pearson r = 0.5110, p = 5.4343e-04
  - Spearman ρ = 0.5921, p = 3.6167e-05

**Correlation Validity Check**:
- Overall Pearson r (0.6248) ≥ ILSP r (0.5110): ✅
- Overall Spearman ρ (0.7433) ≥ ILSP ρ (0.5921): ✅
- **Decision**: VALID ✅
  - *Overall correlation is at least as strong as ILSP-only correlation*

---


## Summary Statistics for TRANSLATION

- **Total Judge-Reference Pairs**: 9
- **Judge Swap Null NOT Rejected** (✅): 8/9
- **Correlation VALID** (✅): 9/9

---


## Dataset: TRUTHFULNESS

### Judge: `Llama-3.1-8B-Instruct`

### Reference: `Llama-3.1-70B-Instruct`

**Null Test Decision** (across all 2 proxies):
- Proxies with p ≥ 0.05: 0/2
- **Decision**: ❌ **REJECT** H₀ (α = 0.05)
  - *All proxies show significant difference between J and K*

#### Self-Recognition vs Self-Preference Correlation
**Overall Correlation** (n=186):
- Pearson r = 0.8021, p = 4.5618e-43
- Spearman ρ = 0.7647, p = 5.9046e-37

**Stratified by LSP/ILSP**:
- **LSP** (n=41):
  - Pearson r = 0.6822, p = 9.0316e-07
  - Spearman ρ = 0.6904, p = 5.8730e-07
- **ILSP** (n=145):
  - Pearson r = 0.6878, p = 1.2345e-21
  - Spearman ρ = 0.6191, p = 1.0389e-16

**Correlation Validity Check**:
- Overall Pearson r (0.8021) ≥ ILSP r (0.6878): ✅
- Overall Spearman ρ (0.7647) ≥ ILSP ρ (0.6191): ✅
- **Decision**: VALID ✅
  - *Overall correlation is at least as strong as ILSP-only correlation*

---

### Reference: `Llama-3.1-8B`

**Null Test Decision** (across all 3 proxies):
- Proxies with p ≥ 0.05: 2/3
- Passing proxies: Qwen2.5-7B, Llama-3.1-70B
  - Proxy `Qwen2.5-7B`: t-test p = 6.2147e-01, one-sided p = 3.1074e-01, KS p = 2.4058e-01, Sample Size = 23
  - Proxy `Llama-3.1-70B`: t-test p = 3.4310e-01, one-sided p = 1.7155e-01, KS p = 3.8555e-01, Sample Size = 15
- **Decision**: ✅ **FAIL TO REJECT** H₀ (α = 0.05)
  - *At least one proxy shows no significant difference between J and K*

#### Self-Recognition vs Self-Preference Correlation
**Overall Correlation** (n=282):
- Pearson r = 0.7905, p = 1.4763e-61
- Spearman ρ = 0.7810, p = 3.5304e-59

**Stratified by LSP/ILSP**:
- **LSP** (n=243):
  - Pearson r = 0.7203, p = 3.6956e-40
  - Spearman ρ = 0.7377, p = 5.2151e-43
- **ILSP** (n=39):
  - Pearson r = 0.7294, p = 1.3881e-07
  - Spearman ρ = 0.5778, p = 1.1738e-04

**Correlation Validity Check**:
- Overall Pearson r (0.7905) ≥ ILSP r (0.7294): ✅
- Overall Spearman ρ (0.7810) ≥ ILSP ρ (0.5778): ✅
- **Decision**: VALID ✅
  - *Overall correlation is at least as strong as ILSP-only correlation*

---

### Reference: `Qwen2.5-7B-Instruct`

**Null Test Decision** (across all 2 proxies):
- Proxies with p ≥ 0.05: 1/2
- Passing proxies: Qwen2.5-7B
  - Proxy `Qwen2.5-7B`: t-test p = 4.3452e-02, one-sided p = 2.1726e-02, KS p = 2.6367e-01, Sample Size = 111
- **Decision**: ✅ **FAIL TO REJECT** H₀ (α = 0.05)
  - *At least one proxy shows no significant difference between J and K*

#### Self-Recognition vs Self-Preference Correlation
**Overall Correlation** (n=208):
- Pearson r = 0.8350, p = 2.3785e-55
- Spearman ρ = 0.8426, p = 2.8927e-57

**Stratified by LSP/ILSP**:
- **LSP** (n=92):
  - Pearson r = 0.6686, p = 3.2809e-13
  - Spearman ρ = 0.6366, p = 9.1068e-12
- **ILSP** (n=116):
  - Pearson r = 0.7451, p = 8.7716e-22
  - Spearman ρ = 0.6997, p = 2.3856e-18

**Correlation Validity Check**:
- Overall Pearson r (0.8350) ≥ ILSP r (0.7451): ✅
- Overall Spearman ρ (0.8426) ≥ ILSP ρ (0.6997): ✅
- **Decision**: VALID ✅
  - *Overall correlation is at least as strong as ILSP-only correlation*

---

### Reference: `gemma-2-9b-it`

**Null Test Decision** (across all 1 proxies):
- Proxies with p ≥ 0.05: 0/1
- **Decision**: ❌ **REJECT** H₀ (α = 0.05)
  - *All proxies show significant difference between J and K*

#### Self-Recognition vs Self-Preference Correlation
**Overall Correlation** (n=176):
- Pearson r = 0.7990, p = 2.7955e-40
- Spearman ρ = 0.8204, p = 4.2450e-44

**Stratified by LSP/ILSP**:
- **LSP** (n=48):
  - Pearson r = 0.4375, p = 1.8752e-03
  - Spearman ρ = 0.5446, p = 6.2874e-05
- **ILSP** (n=128):
  - Pearson r = 0.7110, p = 5.2906e-21
  - Spearman ρ = 0.7161, p = 2.0951e-21

**Correlation Validity Check**:
- Overall Pearson r (0.7990) ≥ ILSP r (0.7110): ✅
- Overall Spearman ρ (0.8204) ≥ ILSP ρ (0.7161): ✅
- **Decision**: VALID ✅
  - *Overall correlation is at least as strong as ILSP-only correlation*

---

### Judge: `Qwen2.5-7B-Instruct`

### Reference: `Llama-3.1-8B-Instruct`

**Null Test Decision** (across all 3 proxies):
- Proxies with p ≥ 0.05: 1/3
- Passing proxies: Llama-3.1-70B-Instruct
  - Proxy `Llama-3.1-70B-Instruct`: t-test p = 4.6401e-01, one-sided p = 2.3201e-01, KS p = 8.2345e-01, Sample Size = 31
- **Decision**: ✅ **FAIL TO REJECT** H₀ (α = 0.05)
  - *At least one proxy shows no significant difference between J and K*

#### Self-Recognition vs Self-Preference Correlation
**Overall Correlation** (n=276):
- Pearson r = 0.7499, p = 4.3850e-51
- Spearman ρ = 0.8206, p = 1.5196e-68

**Stratified by LSP/ILSP**:
- **LSP** (n=141):
  - Pearson r = 0.6348, p = 2.8606e-17
  - Spearman ρ = 0.7664, p = 1.6619e-28
- **ILSP** (n=135):
  - Pearson r = 0.5619, p = 1.3366e-12
  - Spearman ρ = 0.6345, p = 1.4301e-16

**Correlation Validity Check**:
- Overall Pearson r (0.7499) ≥ ILSP r (0.5619): ✅
- Overall Spearman ρ (0.8206) ≥ ILSP ρ (0.6345): ✅
- **Decision**: VALID ✅
  - *Overall correlation is at least as strong as ILSP-only correlation*

---

### Reference: `Qwen2.5-72B-Instruct`

**Null Test Decision** (across all 2 proxies):
- Proxies with p ≥ 0.05: 0/2
- **Decision**: ❌ **REJECT** H₀ (α = 0.05)
  - *All proxies show significant difference between J and K*

#### Self-Recognition vs Self-Preference Correlation
**Overall Correlation** (n=179):
- Pearson r = 0.7116, p = 6.0987e-29
- Spearman ρ = 0.7658, p = 8.9908e-36

**Stratified by LSP/ILSP**:
- **LSP** (n=65):
  - Pearson r = 0.6428, p = 7.7653e-09
  - Spearman ρ = 0.6718, p = 9.0075e-10
- **ILSP** (n=114):
  - Pearson r = 0.6066, p = 8.4871e-13
  - Spearman ρ = 0.6396, p = 1.8540e-14

**Correlation Validity Check**:
- Overall Pearson r (0.7116) ≥ ILSP r (0.6066): ✅
- Overall Spearman ρ (0.7658) ≥ ILSP ρ (0.6396): ✅
- **Decision**: VALID ✅
  - *Overall correlation is at least as strong as ILSP-only correlation*

---

### Reference: `Qwen2.5-7B`

**Null Test Decision** (across all 2 proxies):
- Proxies with p ≥ 0.05: 1/2
- Passing proxies: Qwen2.5-72B
  - Proxy `Qwen2.5-72B`: t-test p = 5.1476e-01, one-sided p = 2.5738e-01, KS p = 7.5464e-02, Sample Size = 15
- **Decision**: ✅ **FAIL TO REJECT** H₀ (α = 0.05)
  - *At least one proxy shows no significant difference between J and K*

#### Self-Recognition vs Self-Preference Correlation
**Overall Correlation** (n=225):
- Pearson r = 0.6543, p = 6.9879e-29
- Spearman ρ = 0.7553, p = 8.3344e-43

**Stratified by LSP/ILSP**:
- **LSP** (n=181):
  - Pearson r = 0.5769, p = 1.8905e-17
  - Spearman ρ = 0.7149, p = 1.2864e-29
- **ILSP** (n=44):
  - Pearson r = 0.3928, p = 8.3538e-03
  - Spearman ρ = 0.4708, p = 1.2589e-03

**Correlation Validity Check**:
- Overall Pearson r (0.6543) ≥ ILSP r (0.3928): ✅
- Overall Spearman ρ (0.7553) ≥ ILSP ρ (0.4708): ✅
- **Decision**: VALID ✅
  - *Overall correlation is at least as strong as ILSP-only correlation*

---

### Judge: `gemma-2-9b-it`

### Reference: `Llama-3.1-8B-Instruct`

**Null Test Decision** (across all 3 proxies):
- Proxies with p ≥ 0.05: 2/3
- Passing proxies: Qwen2.5-7B-Instruct, Llama-3.1-70B-Instruct
  - Proxy `Qwen2.5-7B-Instruct`: t-test p = 4.3918e-01, one-sided p = 2.1959e-01, KS p = 7.3716e-01, Sample Size = 86
  - Proxy `Llama-3.1-70B-Instruct`: t-test p = 9.6779e-02, one-sided p = 4.8389e-02, KS p = 4.3374e-01, Sample Size = 32
- **Decision**: ✅ **FAIL TO REJECT** H₀ (α = 0.05)
  - *At least one proxy shows no significant difference between J and K*

#### Self-Recognition vs Self-Preference Correlation
**Overall Correlation** (n=270):
- Pearson r = 0.5947, p = 3.2306e-27
- Spearman ρ = 0.6330, p = 1.2159e-31

**Stratified by LSP/ILSP**:
- **LSP** (n=135):
  - Pearson r = 0.3888, p = 3.1661e-06
  - Spearman ρ = 0.4567, p = 2.5851e-08
- **ILSP** (n=135):
  - Pearson r = 0.2698, p = 1.5523e-03
  - Spearman ρ = 0.2726, p = 1.3815e-03

**Correlation Validity Check**:
- Overall Pearson r (0.5947) ≥ ILSP r (0.2698): ✅
- Overall Spearman ρ (0.6330) ≥ ILSP ρ (0.2726): ✅
- **Decision**: VALID ✅
  - *Overall correlation is at least as strong as ILSP-only correlation*

---

### Reference: `gemma-2-9b`

**Null Test Decision** (across all 1 proxies):
- Proxies with p ≥ 0.05: 0/1
- **Decision**: ❌ **REJECT** H₀ (α = 0.05)
  - *All proxies show significant difference between J and K*

#### Self-Recognition vs Self-Preference Correlation
**Overall Correlation** (n=138):
- Pearson r = 0.7360, p = 8.3865e-25
- Spearman ρ = 0.7652, p = 8.6938e-28

**Stratified by LSP/ILSP**:
- **LSP** (n=92):
  - Pearson r = 0.1037, p = 3.2527e-01
  - Spearman ρ = 0.5679, p = 3.5643e-09
- **ILSP** (n=46):
  - Pearson r = 0.6775, p = 2.3283e-07
  - Spearman ρ = 0.7354, p = 5.8393e-09

**Correlation Validity Check**:
- Overall Pearson r (0.7360) ≥ ILSP r (0.6775): ✅
- Overall Spearman ρ (0.7652) ≥ ILSP ρ (0.7354): ✅
- **Decision**: VALID ✅
  - *Overall correlation is at least as strong as ILSP-only correlation*

---


## Summary Statistics for TRUTHFULNESS

- **Total Judge-Reference Pairs**: 9
- **Judge Swap Null NOT Rejected** (✅): 5/9
- **Correlation VALID** (✅): 9/9

---


## Overall Summary Across All Datasets

- **Total Pairs Analyzed**: 34
- **Judge Swap Null NOT Rejected**: 27/34 (79.4%)
- **Correlation VALID**: 33/34 (97.1%)
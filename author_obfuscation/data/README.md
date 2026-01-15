---
license: mit
viewer: false
---
# Mitigating Self-Preference by Authorship Obfuscation (Dataset)

## Dataset Summary

This dataset supports the project **“Mitigating Self-Preference by Authorship Obfuscation.”**
It contains long-form reading-comprehension questions (sourced from **QuALITY**)

---

## Dataset Structure

### Data Fields

| Field                                  | Type              | Description                                                                    |
| -------------------------------------- | ----------------- | ------------------------------------------------------------------------------ |
| `pid`                                  | `string`          | Unique identifier for a question instance.                                     |
| `text`                                 | `string`          | Long passage on which the question is based.                                   |
| `questions`                            | `string`          | The question **including the options A–D** (verbatim as provided).             |
| `output_label`                         | `string`          | Gold label (one of `A`, `B`, `C`, `D`).                                        |
| `output`                               | `string`          | Gold answer text corresponding to `output_label`.                              |
| `{model}_output_label`                 | `string`          | Model’s predicted label (`A`–`D`). Example: `DeepSeek-V3_output_label`.        |
| `{model}_reason`                       | `string`          | Model’s free-text justification. Example: `DeepSeek-V3_reason`.                |
| `{model}_reason_perturb_llm_auto`      | `string`          | The same model’s reason with **2 words replaced with synonyms**.               |
| `{model1}_reason_paraphrased_{model2}` | `string`          | `{model1}`’s reason paraphrased by `{model2}`.                                 |

> **Notes**
>
> * `{model}` and `{model1}/{model2}` are literal model identifiers (e.g., `DeepSeek-V3`, `Qwen2.5-7B-Instruct`).

##  Preference and Recognition Data

This data files contains **model preference** and **self-recognition** results under various evaluation settings and quadrants. Each JSON file corresponds to a specific configuration (normal or 2-word perturbed; beneficial or harmful quadrant).

---

###  Preference Data

Each entry in the preference JSON files represents a single evaluation instance comparing two models.  
The fields are structured as follows:

| Field | Description |
|--------|--------------|
| `evaluator` | The **judge model** for the record. |
| `evaluatee` | The **competitor model** . |
| `pid` | A unique **problem identifier**. |
| `forward_comparison` | The evaluator’s preference in the **forward ordering (A,B)**. |
| `backward_comparison` | The evaluator’s preference in the **reverse ordering (B,A)**. |

**Files:**
- `clean_pref_quality_harmful.json` — Harmful quadrant, **normal** setting  
- `clean_pref_quality_ben.json` — Beneficial quadrant, **normal** setting  
- `clean_pref_2w_quality_harmful.json` — Harmful quadrant, **2-word perturbation** setting  
- `clean_pref_2w_quality_ben.json` — Beneficial quadrant, **2-word perturbation** setting  

---

### Recognition Data

The recognition JSON files measure how well models **recognize or detect themselves** in pairwise setups.  
The schema mirrors the preference data but with detection-specific fields.

| Field | Description |
|--------|--------------|
| `evaluator` | The **judge model**  for the record. |
| `evaluatee` | The **competitor model** . |
| `pid` | A unique **problem identifier**. |
| `forward_detection` | Recognition output in the **forward ordering (A,B)**. |
| `backward_detection` | Recognition output in the **reverse ordering (B,A)**. |

**Files:**
- `clean_self_recog_quality_ben.json` — Recognition, **beneficial quadrant**, normal setting  
- `clean_self_recog_quality_harmful.json` — Recognition, **harmful quadrant**, normal setting  
- `clean_self_recog_quality_2w_ben.json` — Recognition, **beneficial quadrant**, 2-word perturbation  
- `clean_self_recog_quality_2w_harmful.json` — Recognition, **harmful quadrant**, 2-word perturbation  


## Licensing

* **This dataset:** `CC BY 4.0`
* **QuALITY content:** you must also comply with **QuALITY’s** original license/terms.

---

## Citation

If you use this dataset, please cite this project.

**This project:**

```bibtex
@dataset{self_preference_obfuscation_2026,
  title   = {Mitigating Self-Preference by Authorship Obfuscation},
  author  = {Taslim Mahbub and Shi Feng},
  year    = {2026},
  url     = {}
}
```
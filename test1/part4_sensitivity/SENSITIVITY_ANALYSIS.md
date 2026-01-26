# Sensitivity Analysis: Out-of-Family Proxies

## Purpose

This analysis tests whether using only out-of-family proxies (i.e., excluding proxies from
the same model family as the judge) significantly changes our findings about self-preference.

The concern is that same-family models might share similar biases, which could inflate or
deflate self-preference measurements in ways that don't generalize.

## Methodology

1. **Full data**: All available (judge, proxy) pairs
2. **Out-of-family**: Only (judge, proxy) pairs where judge and proxy are from different
   model families (e.g., Llama judge with Qwen/Gemma proxies, but not Llama proxies)

For each condition, we compute the mean difference `J - avg(K)` where:
- J = judge's self-preference (position-averaged P(self) on ILSP examples)
- K = proxy's preference for judge's response (position-averaged)

## Results

### ILSP (Illegitimate Self-Preference)

| Condition | N | Mean Diff | Std |
|-----------|---|-----------|-----|
| All proxies | 11,684 | 0.0197 | 0.2327 |
| Out-of-family | 10,106 | 0.0227 | 0.2439 |
| **Difference** | | **+0.0031** | |

### LSP (Legitimate Self-Preference)

| Condition | N | Mean Diff | Std |
|-----------|---|-----------|-----|
| All proxies | 19,298 | -0.0006 | 0.2020 |
| Out-of-family | 17,681 | 0.0023 | 0.2203 |
| **Difference** | | **+0.0029** | |

## Interpretation

The difference in mean ILSP diff between all proxies and out-of-family only is
**0.0031** (larger
with all proxies).

This suggests that
same-family proxies do not substantially affect the self-preference measurement.

## Conclusion

**The results are robust to proxy family composition.** Using only out-of-family proxies does not substantially change the estimated self-preference effect.

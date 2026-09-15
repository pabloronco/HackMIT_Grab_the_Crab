# R8 training diagnostics (observation-only, no policy/reward/protocol change)

Requested alongside the R8 corrected rerun to decide whether a *separately preregistered* future ablation is scientifically justified. Not used to modify this rerun.

## r8_fixed_seed0_20260915_185240

### 1. Validation curve plateau check
- eval points: 60 (update 5 -> 300)
- last 3 rl_missed_fraction: [0.7394, 0.5947, 0.6714]
- tail linear slope (last 10 points): +0.001686 missed_fraction/update
- first-half mean 0.6375 -> second-half mean 0.6327 (improvement +0.0047)
- Read as plateaued (tail slope not meaningfully negative).

### 2. Reward term magnitude (dense vs. terminal)
Per-round dense terms (fires every round):
  - mean_uncertainty_reduction: mean=+0.0279, mean_abs=0.0347 (n=25730 rounds)
  - new_detections: mean=+0.0669, mean_abs=0.0669 (n=25730 rounds)
Per-episode SUMMED dense terms (comparable scale to the one terminal term):
  - mean_uncertainty_reduction: mean=+0.1498, mean_abs=0.1545 (n=4800 episodes)
  - new_detections: mean=+0.3586, mean_abs=0.3586 (n=4800 episodes)
Terminal term (missed_occupied_fraction, fires once/episode): mean=-1.3577, mean_abs=1.3577 (n=4800 episodes)
terminal_mean_abs / summed_dense_mean_abs = 2.646
- Not clearly drowned out by this ratio.

### 3. Action statistics
- episodes: 4800, rounds: 25730
  - effort=1: 11055 (43.0%)
  - effort=3: 5407 (21.0%)
  - effort=6: 9268 (36.0%)
- same-episode site revisit rate: 14.0%
- mean distinct sites visited per episode: 4.61

## r8_fixed_seed1_20260915_185246

### 1. Validation curve plateau check
- eval points: 60 (update 5 -> 300)
- last 3 rl_missed_fraction: [0.7579, 0.5947, 0.638]
- tail linear slope (last 10 points): +0.000580 missed_fraction/update
- first-half mean 0.6483 -> second-half mean 0.6518 (improvement -0.0035)
- Read as plateaued (tail slope not meaningfully negative).

### 2. Reward term magnitude (dense vs. terminal)
Per-round dense terms (fires every round):
  - mean_uncertainty_reduction: mean=+0.0288, mean_abs=0.0358 (n=25558 rounds)
  - new_detections: mean=+0.0666, mean_abs=0.0666 (n=25558 rounds)
Per-episode SUMMED dense terms (comparable scale to the one terminal term):
  - mean_uncertainty_reduction: mean=+0.1532, mean_abs=0.1585 (n=4800 episodes)
  - new_detections: mean=+0.3544, mean_abs=0.3544 (n=4800 episodes)
Terminal term (missed_occupied_fraction, fires once/episode): mean=-1.3528, mean_abs=1.3528 (n=4800 episodes)
terminal_mean_abs / summed_dense_mean_abs = 2.638
- Not clearly drowned out by this ratio.

### 3. Action statistics
- episodes: 4800, rounds: 25558
  - effort=1: 9737 (38.1%)
  - effort=3: 7044 (27.6%)
  - effort=6: 8777 (34.3%)
- same-episode site revisit rate: 14.3%
- mean distinct sites visited per episode: 4.56

## r8_fixed_seed2_20260915_185251

### 1. Validation curve plateau check
- eval points: 60 (update 5 -> 300)
- last 3 rl_missed_fraction: [0.6838, 0.5391, 0.6892]
- tail linear slope (last 10 points): +0.000635 missed_fraction/update
- first-half mean 0.6535 -> second-half mean 0.6359 (improvement +0.0176)
- Read as plateaued (tail slope not meaningfully negative).

### 2. Reward term magnitude (dense vs. terminal)
Per-round dense terms (fires every round):
  - mean_uncertainty_reduction: mean=+0.0264, mean_abs=0.0328 (n=26320 rounds)
  - new_detections: mean=+0.0704, mean_abs=0.0704 (n=26320 rounds)
Per-episode SUMMED dense terms (comparable scale to the one terminal term):
  - mean_uncertainty_reduction: mean=+0.1448, mean_abs=0.1493 (n=4800 episodes)
  - new_detections: mean=+0.3862, mean_abs=0.3862 (n=4800 episodes)
Terminal term (missed_occupied_fraction, fires once/episode): mean=-1.3381, mean_abs=1.3381 (n=4800 episodes)
terminal_mean_abs / summed_dense_mean_abs = 2.499
- Not clearly drowned out by this ratio.

### 3. Action statistics
- episodes: 4800, rounds: 26320
  - effort=1: 11492 (43.7%)
  - effort=3: 6233 (23.7%)
  - effort=6: 8595 (32.7%)
- same-episode site revisit rate: 13.0%
- mean distinct sites visited per episode: 4.77

## Overall: no flags triggered.

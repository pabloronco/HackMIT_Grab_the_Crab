# R8 paired case-wise comparison (bootstrap 95% CI on missed_occupied_fraction delta)

delta = baseline_missed_fraction - rl_missed_fraction; positive means RL missed less (RL better on that case). Reporting-only analysis, does not change policy/reward/protocol.

## frontier <-> gnn_rl

### Per-seed (bootstrap over 180 cases, one CI per RL training seed)

**ALL**
- seed 0 (n=180): mean_delta=-0.0051  95% CI=[-0.0188, +0.0089]  -> not distinguishable from 0
- seed 1 (n=180): mean_delta=-0.0002  95% CI=[-0.0110, +0.0113]  -> not distinguishable from 0
- seed 2 (n=180): mean_delta=-0.0233  95% CI=[-0.0416, -0.0054]  -> RL worse (CI excludes 0, negative)

**id_test**
- seed 0 (n=90): mean_delta=+0.0008  95% CI=[-0.0186, +0.0200]  -> not distinguishable from 0
- seed 1 (n=90): mean_delta=+0.0013  95% CI=[-0.0152, +0.0193]  -> not distinguishable from 0
- seed 2 (n=90): mean_delta=-0.0281  95% CI=[-0.0572, +0.0007]  -> not distinguishable from 0

**ood_model_test**
- seed 0 (n=30): mean_delta=-0.0332  95% CI=[-0.0589, -0.0114]  -> RL worse (CI excludes 0, negative)
- seed 1 (n=30): mean_delta=+0.0078  95% CI=[-0.0317, +0.0485]  -> not distinguishable from 0
- seed 2 (n=30): mean_delta=-0.0122  95% CI=[-0.0425, +0.0176]  -> not distinguishable from 0

**ood_q_high**
- seed 0 (n=30): mean_delta=+0.0058  95% CI=[-0.0242, +0.0458]  -> not distinguishable from 0
- seed 1 (n=30): mean_delta=-0.0126  95% CI=[-0.0296, +0.0000]  -> not distinguishable from 0
- seed 2 (n=30): mean_delta=-0.0098  95% CI=[-0.0458, +0.0250]  -> not distinguishable from 0

**ood_q_low**
- seed 0 (n=30): mean_delta=-0.0056  95% CI=[-0.0500, +0.0333]  -> not distinguishable from 0
- seed 1 (n=30): mean_delta=+0.0000  95% CI=[+0.0000, +0.0000]  -> not distinguishable from 0
- seed 2 (n=30): mean_delta=-0.0333  95% CI=[-0.0833, +0.0000]  -> not distinguishable from 0

### Pooled across 3 seeds (case-clustered bootstrap)
- **ALL** (n_cases=180, n_case_seed_obs=540): mean_delta=-0.0095  95% CI=[-0.0202, +0.0010]  -> not distinguishable from 0
- **id_test** (n_cases=90, n_case_seed_obs=270): mean_delta=-0.0087  95% CI=[-0.0250, +0.0077]  -> not distinguishable from 0
- **ood_model_test** (n_cases=30, n_case_seed_obs=90): mean_delta=-0.0125  95% CI=[-0.0391, +0.0131]  -> not distinguishable from 0
- **ood_q_high** (n_cases=30, n_case_seed_obs=90): mean_delta=-0.0055  95% CI=[-0.0226, +0.0129]  -> not distinguishable from 0
- **ood_q_low** (n_cases=30, n_case_seed_obs=90): mean_delta=-0.0130  95% CI=[-0.0407, +0.0074]  -> not distinguishable from 0

## information_gain <-> gnn_rl

### Per-seed (bootstrap over 180 cases, one CI per RL training seed)

**ALL**
- seed 0 (n=180): mean_delta=+0.0543  95% CI=[+0.0317, +0.0779]  -> RL better (CI excludes 0, positive)
- seed 1 (n=180): mean_delta=+0.0592  95% CI=[+0.0370, +0.0824]  -> RL better (CI excludes 0, positive)
- seed 2 (n=180): mean_delta=+0.0361  95% CI=[+0.0171, +0.0552]  -> RL better (CI excludes 0, positive)

**id_test**
- seed 0 (n=90): mean_delta=+0.0835  95% CI=[+0.0530, +0.1161]  -> RL better (CI excludes 0, positive)
- seed 1 (n=90): mean_delta=+0.0840  95% CI=[+0.0544, +0.1164]  -> RL better (CI excludes 0, positive)
- seed 2 (n=90): mean_delta=+0.0547  95% CI=[+0.0292, +0.0817]  -> RL better (CI excludes 0, positive)

**ood_model_test**
- seed 0 (n=30): mean_delta=-0.0334  95% CI=[-0.0751, +0.0044]  -> not distinguishable from 0
- seed 1 (n=30): mean_delta=+0.0076  95% CI=[-0.0388, +0.0516]  -> not distinguishable from 0
- seed 2 (n=30): mean_delta=-0.0124  95% CI=[-0.0540, +0.0258]  -> not distinguishable from 0

**ood_q_high**
- seed 0 (n=30): mean_delta=+0.0752  95% CI=[-0.0015, +0.1494]  -> not distinguishable from 0
- seed 1 (n=30): mean_delta=+0.0568  95% CI=[-0.0165, +0.1289]  -> not distinguishable from 0
- seed 2 (n=30): mean_delta=+0.0595  95% CI=[-0.0122, +0.1275]  -> not distinguishable from 0

**ood_q_low**
- seed 0 (n=30): mean_delta=+0.0333  95% CI=[-0.0026, +0.0795]  -> not distinguishable from 0
- seed 1 (n=30): mean_delta=+0.0388  95% CI=[-0.0019, +0.0910]  -> not distinguishable from 0
- seed 2 (n=30): mean_delta=+0.0055  95% CI=[-0.0111, +0.0221]  -> not distinguishable from 0

### Pooled across 3 seeds (case-clustered bootstrap)
- **ALL** (n_cases=180, n_case_seed_obs=540): mean_delta=+0.0499  95% CI=[+0.0305, +0.0698]  -> RL better (CI excludes 0, positive)
- **id_test** (n_cases=90, n_case_seed_obs=270): mean_delta=+0.0741  95% CI=[+0.0494, +0.1001]  -> RL better (CI excludes 0, positive)
- **ood_model_test** (n_cases=30, n_case_seed_obs=90): mean_delta=-0.0127  95% CI=[-0.0509, +0.0218]  -> not distinguishable from 0
- **ood_q_high** (n_cases=30, n_case_seed_obs=90): mean_delta=+0.0638  95% CI=[-0.0067, +0.1318]  -> not distinguishable from 0
- **ood_q_low** (n_cases=30, n_case_seed_obs=90): mean_delta=+0.0258  95% CI=[-0.0008, +0.0563]  -> not distinguishable from 0

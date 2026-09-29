# Backtest — Analyst Dashboard technical model (TASE)

Sample: 2013-03-21 → 2026-09-29 · split in/out of sample at 2020-01-01 · 62 stocks · benchmark ^TA125.TA · data: Yahoo Finance

Excess return = stock return minus TA-125 return over the horizon, entry at the next session's close, in %.
hit = % of observations with positive excess return. t = t-statistic of the mean. IC = mean weekly Spearman rank correlation between score and forward excess return.

**Caveats:** the universe is today's list of 62 stocks (survivorship bias), weekly observations overlap for 3m/6m horizons (t-stats overstated), no transaction costs.

## V19 (original)

### in sample

| horizon | IC | IC IR | WAIT mean / hit | BUY n · mean / hit | STRONG BUY n · mean / hit | EXIT mean / hit |
|---|---|---|---|---|---|---|
| 1m | 0.0205 | 0.134 | 0.9 / 52.8% | 142 · 1.12 / 52.8% | — | 0.73 / 51.7% |
| 3m | 0.0449 | 0.269 | 2.77 / 55.7% | 142 · 2.39 / 56.3% | — | 1.67 / 53.9% |
| 6m | 0.0374 | 0.244 | 5.69 / 57.2% | 142 · 1.13 / 54.9% | — | 3.48 / 54.9% |

Score buckets (3m excess return):

| score | n | mean | median | hit |
|---|---|---|---|---|
| <45 | 8298 | 2.29 | 1.4 | 54.7% |
| 45-55 | 6691 | 3.28 | 1.81 | 56.5% |
| 55-68 | 3406 | 2.92 | 1.83 | 56.4% |
| 68-82 | 449 | 2.6 | 1.56 | 56.3% |
| 82+ | 10 | 2.12 | -0.36 | 50.0% |

### out of sample

| horizon | IC | IC IR | WAIT mean / hit | BUY n · mean / hit | STRONG BUY n · mean / hit | EXIT mean / hit |
|---|---|---|---|---|---|---|
| 1m | 0.0028 | 0.018 | 0.03 / 47.0% | 190 · -1.81 / 38.9% | — | -0.01 / 47.3% |
| 3m | 0.0076 | 0.047 | 0.07 / 45.2% | 183 · -3.09 / 42.1% | — | -0.51 / 44.5% |
| 6m | -0.0077 | -0.051 | 0.36 / 43.4% | 168 · -6.81 / 33.3% | — | -0.35 / 43.2% |

Score buckets (3m excess return):

| score | n | mean | median | hit |
|---|---|---|---|---|
| <45 | 9654 | 0.35 | -1.7 | 45.4% |
| 45-55 | 7266 | -0.03 | -1.92 | 44.4% |
| 55-68 | 3433 | -0.51 | -1.75 | 45.5% |
| 68-82 | 314 | -1.49 | -0.4 | 49.0% |
| 82+ | 2 | 4.17 | 4.17 | 50.0% |

## V20

### in sample

| horizon | IC | IC IR | WAIT mean / hit | BUY n · mean / hit | STRONG BUY n · mean / hit | EXIT mean / hit |
|---|---|---|---|---|---|---|
| 1m | 0.0179 | 0.119 | 0.91 / 52.8% | 50 · -0.45 / 42.0% | — | 0.96 / 52.1% |
| 3m | 0.0352 | 0.215 | 2.77 / 55.7% | 50 · 0.3 / 56.0% | — | 1.96 / 54.4% |
| 6m | 0.0311 | 0.201 | 5.66 / 57.2% | 50 · 4.41 / 52.0% | — | 3.67 / 54.3% |

Score buckets (3m excess return):

| score | n | mean | median | hit |
|---|---|---|---|---|
| <45 | 8659 | 2.52 | 1.52 | 55.0% |
| 45-55 | 7082 | 3.04 | 1.79 | 56.3% |
| 55-68 | 2899 | 2.95 | 1.66 | 56.4% |
| 68-82 | 214 | 1.2 | 0.97 | 52.3% |

### out of sample

| horizon | IC | IC IR | WAIT mean / hit | BUY n · mean / hit | STRONG BUY n · mean / hit | EXIT mean / hit |
|---|---|---|---|---|---|---|
| 1m | 0.0057 | 0.036 | 0.01 / 46.9% | 109 · 0.19 / 50.5% | — | 0.07 / 48.0% |
| 3m | 0.0046 | 0.029 | 0.04 / 45.1% | 102 · 0.8 / 48.0% | — | -0.48 / 45.5% |
| 6m | -0.0105 | -0.07 | 0.31 / 43.3% | 97 · -2.5 / 36.1% | — | -0.31 / 43.6% |

Score buckets (3m excess return):

| score | n | mean | median | hit |
|---|---|---|---|---|
| <45 | 10186 | 0.4 | -1.71 | 45.4% |
| 45-55 | 7411 | 0.01 | -1.8 | 45.0% |
| 55-68 | 2896 | -1.05 | -1.85 | 44.5% |
| 68-82 | 175 | -1.26 | -1.49 | 44.0% |
| 82+ | 1 | 1.25 | 1.25 | 100.0% |

## V20 + ADX filter

### in sample

| horizon | IC | IC IR | WAIT mean / hit | BUY n · mean / hit | STRONG BUY n · mean / hit | EXIT mean / hit |
|---|---|---|---|---|---|---|
| 1m | 0.0179 | 0.119 | 0.91 / 52.8% | 36 · 0.08 / 47.2% | — | 0.96 / 52.1% |
| 3m | 0.0352 | 0.215 | 2.77 / 55.7% | 36 · 2.05 / 58.3% | — | 1.96 / 54.4% |
| 6m | 0.0311 | 0.201 | 5.66 / 57.2% | 36 · 6.05 / 52.8% | — | 3.67 / 54.3% |

Score buckets (3m excess return):

| score | n | mean | median | hit |
|---|---|---|---|---|
| <45 | 8659 | 2.52 | 1.52 | 55.0% |
| 45-55 | 7082 | 3.04 | 1.79 | 56.3% |
| 55-68 | 2899 | 2.95 | 1.66 | 56.4% |
| 68-82 | 214 | 1.2 | 0.97 | 52.3% |

### out of sample

| horizon | IC | IC IR | WAIT mean / hit | BUY n · mean / hit | STRONG BUY n · mean / hit | EXIT mean / hit |
|---|---|---|---|---|---|---|
| 1m | 0.0057 | 0.036 | 0.01 / 46.9% | 72 · 0.38 / 50.0% | — | 0.07 / 48.0% |
| 3m | 0.0046 | 0.029 | 0.05 / 45.1% | 67 · -0.76 / 43.3% | — | -0.48 / 45.5% |
| 6m | -0.0105 | -0.07 | 0.32 / 43.3% | 64 · -5.19 / 31.2% | — | -0.31 / 43.6% |

Score buckets (3m excess return):

| score | n | mean | median | hit |
|---|---|---|---|---|
| <45 | 10186 | 0.4 | -1.71 | 45.4% |
| 45-55 | 7411 | 0.01 | -1.8 | 45.0% |
| 55-68 | 2896 | -1.05 | -1.85 | 44.5% |
| 68-82 | 175 | -1.26 | -1.49 | 44.0% |
| 82+ | 1 | 1.25 | 1.25 | 100.0% |

## V20 + RS percentile

### in sample

| horizon | IC | IC IR | WAIT mean / hit | BUY n · mean / hit | STRONG BUY n · mean / hit | EXIT mean / hit |
|---|---|---|---|---|---|---|
| 1m | 0.0248 | 0.161 | 0.91 / 52.8% | 41 · -1.7 / 31.7% | 1 · 2.65 / 100.0% | 0.96 / 52.1% |
| 3m | 0.0403 | 0.239 | 2.78 / 55.7% | 41 · -2.75 / 51.2% | 1 · 0.16 / 100.0% | 1.96 / 54.4% |
| 6m | 0.0385 | 0.246 | 5.67 / 57.3% | 41 · -0.7 / 41.5% | 1 · -4.0 / 0.0% | 3.67 / 54.3% |

Score buckets (3m excess return):

| score | n | mean | median | hit |
|---|---|---|---|---|
| <45 | 8804 | 2.6 | 1.67 | 55.4% |
| 45-55 | 7052 | 2.92 | 1.69 | 56.0% |
| 55-68 | 2814 | 3.0 | 1.56 | 56.0% |
| 68-82 | 184 | 1.06 | 0.09 | 50.5% |

### out of sample

| horizon | IC | IC IR | WAIT mean / hit | BUY n · mean / hit | STRONG BUY n · mean / hit | EXIT mean / hit |
|---|---|---|---|---|---|---|
| 1m | 0.0112 | 0.07 | 0.01 / 46.9% | 134 · 0.47 / 50.0% | — | 0.07 / 48.0% |
| 3m | 0.0117 | 0.072 | 0.04 / 45.1% | 127 · 0.74 / 49.6% | — | -0.48 / 45.5% |
| 6m | -0.0045 | -0.03 | 0.31 / 43.3% | 119 · -2.49 / 37.8% | — | -0.31 / 43.6% |

Score buckets (3m excess return):

| score | n | mean | median | hit |
|---|---|---|---|---|
| <45 | 10143 | 0.35 | -1.74 | 45.4% |
| 45-55 | 7456 | -0.06 | -1.77 | 44.9% |
| 55-68 | 2882 | -0.72 | -1.91 | 44.7% |
| 68-82 | 188 | -1.03 | -1.11 | 44.7% |

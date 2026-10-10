# LP QA (nfp 2, A 6, vacuum): published coil sets through the three-term free-boundary solve

Target (prescribed boundary, fixed-boundary solve): two-term QS error 5.49e-07, iota 0.423 -> 0.416, A 6.000

| coil set | <B.n>/<B> on target (ours / Gil x1e-5) | QS actual free-boundary (ours) | QS Gil's table | iota axis->edge | A | B.n / p-bal / K after solve |
|---|---|---|---|---|---|---|
| wechsung_L18 | 83.0 / 91.1 | 1.47e-03 | 2.1e-03 | 0.424 -> 0.416 | 6.000 | 1.8e-04 / 1.2e-04 / 2.3e-04 |
| gil_L18 | 88.0 / 93.7 | 1.47e-03 | 1.9e-03 | 0.424 -> 0.416 | 6.000 | 2.4e-04 / 1.5e-04 / 2.9e-04 |
| wechsung_L20 | 29.6 / 32 | 1.61e-04 | 1.9e-04 | 0.423 -> 0.416 | 6.000 | 8.4e-05 / 5.3e-05 / 1.0e-04 |
| gil_L20 | 32.5 / 34 | 1.78e-04 | 2.6e-04 | 0.423 -> 0.416 | 6.000 | 9.0e-05 / 5.7e-05 / 1.1e-04 |
| wechsung_L24 | 4.2 / 4.35 | 3.07e-06 | 2.4e-06 | 0.423 -> 0.416 | 6.000 | 3.0e-05 / 1.8e-05 / 2.7e-05 |
| gil_L24 | 6.0 / 6.0 | 7.87e-06 | 1.1e-05 | 0.423 -> 0.416 | 6.000 | 3.2e-05 / 2.0e-05 / 3.2e-05 |
| gil_3coil_L18 | 81.7 / 85.5 | 9.90e-04 | 2.0e-03 | 0.423 -> 0.416 | 6.000 | 6.2e-05 / 3.8e-05 / 8.5e-05 |

QS = sum over s = 0.1..1.0 of the two-term quasisymmetry residual (simsopt convention); Gil's column is their Table 2 (QFM surface + VMEC).

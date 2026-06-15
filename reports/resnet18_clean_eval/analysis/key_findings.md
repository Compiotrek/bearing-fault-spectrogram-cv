# Key Findings

- **Best variant by accuracy:** `clean` (1.000).
- **Worst variant by accuracy:** `noisy_0db` (0.290).
- **Clean performance:** accuracy 1.000, macro F1 1.000.
- **Noisy performance trend:** 10db: 0.950, 5db: 0.611, 0db: 0.290. Lower SNR indicates stronger noise.

## Denoising Effect

- **10 dB:** denoising hurts accuracy by -0.088 (0.950 to 0.862).
- **5 dB:** denoising improves accuracy by +0.243 (0.611 to 0.854).
- **0 dB:** denoising improves accuracy by +0.399 (0.290 to 0.689).

## Weakest Recall

- `normal`: 0.027 on `noisy_0db`.
- `ball`: 0.137 on `denoised_0db`.
- `inner_race`: 0.966 on `noisy_5db`.
- `outer_race`: 0.000 on `noisy_0db`.
- **Overall weakest:** `outer_race` on `noisy_0db` with recall 0.000.

# CKS algorithm

For candidate `c` in document `d`:

`CKS(c,d) = w_T T(c,d) + w_D D(c) + w_X X(c) + w_F F(c) + w_P P(c)`.

- `T`: within-document min-max-scaled TF-IDF.
- `D`: `log(1 + df) / log(1 + N)`.
- `X`: normalized Shannon entropy over fitting-document occurrence counts.
- `F`: log-scaled document frequency in observed author keywords.
- `P`: `0.7 * structural_quality + 0.3 * keyword_length_prior`.

The Paper-1 frozen profile uses configuration `W050_S40` and threshold
`0.40`. All corpus-level resources were fitted on 305 development records.

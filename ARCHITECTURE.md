# CDR Sentinel — Architecture in Plain English

## One-sentence explanation

We clean RBI and NFHS data, match the same districts across both sources, calculate CDR from RBI credit/deposits, train a regression model using NFHS indicators only, and present the evaluation and district-level residuals in a dashboard.

## Pipeline

```text
RBI workbook ──> Clean quarter values ──> Sum credit/deposits ──> CDR target ─┐
                                                                              ├─> District matching
NFHS training data ──> Clean indicators ──> State-aware fuzzy matching ───────┘
                                           |
                                           v
                               Matched training dataset
                                           |
                                           v
                       Numeric NFHS features only (no RBI target inputs)
                                           |
                                           v
                         Median imputation inside ML pipeline
                                           |
                                           v
                    Compare median baseline / Ridge / Extra Trees
                                           |
                                           v
                         Holdout metrics + permutation importance
                                           |
                          ┌────────────────┴────────────────┐
                          v                                 v
                  Streamlit dashboard             Saved model pipeline
                                                            |
NFHS test data ──> Same feature columns ─────────────────────┘
                                                            |
                                                            v
                                  submission.csv (Predicted_CDR only)
```

## How to explain each part

1. **RBI cleaning:** The workbook has metadata and a two-row header. The loader locates district, deposit and credit columns, converts numeric strings and sums all provided quarters.
2. **Target calculation:** For each district, `CDR = (total credit / total deposits) × 100`. Rows with invalid or zero deposits do not get a valid target.
3. **NFHS preparation:** District/state labels are normalized for matching; candidate indicator columns are converted to numeric. Original labels are retained for review.
4. **Entity resolution:** Exact district matches are tried first. Fuzzy matching is state-aware where possible, uses a confidence threshold, and does not reuse an RBI district for multiple NFHS records. The decisions are saved in an audit CSV.
5. **Leakage prevention:** RBI credit, deposits and the calculated CDR are excluded from model features. The model only receives numeric NFHS columns.
6. **Model evaluation:** Median baseline, Ridge and Extra Trees use the same train/validation split. MAE, RMSE and R² are compared; the lowest-RMSE candidate is selected.
7. **Interpretation:** Permutation importance estimates which features the fitted model relies on for validation predictions. It cannot show that a feature causes CDR.
8. **Test submission:** The trained preprocessing/model pipeline predicts rows from `social_test.csv` in their original order. The output has exactly one column: `Predicted_CDR`.

## Interview answer (about 45 seconds)

“I treated this as both a data-engineering and machine-learning problem. First, I parsed the RBI workbook, cleaned its multi-row headers and numeric values, and aggregated quarterly credit and deposits to calculate district-level CDR. The NFHS data used different district naming conventions, so I normalised names and used state-aware fuzzy matching, then saved a matching audit for quality checks. To avoid target leakage, I used only numeric NFHS indicators as predictors and kept the RBI-derived CDR as the target. I compared a baseline, Ridge regression and Extra Trees on a held-out split, reviewed MAE, RMSE and R², and used permutation importance to inspect model behaviour. Finally, I built a Streamlit dashboard and a one-column submission generator.”

## Honest limitations to mention

- Fuzzy matching needs manual review; a high string-similarity score is not a guarantee that two districts are identical.
- NFHS and RBI observations may represent different time periods.
- The random holdout does not fully test performance on unseen regions; spatial/grouped validation would be a valuable next step.
- Residuals and feature importance are exploratory signals, not causal claims or proof of underserved credit demand.

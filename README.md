# Credit Default Classification

## Overview

This project applies supervised classification methods to predict loan default risk using a real-world credit dataset. The analysis covers exploratory data analysis (EDA), linear probability modeling, logistic regression, and decision tree classification.

---

## Dataset

- **Full dataset size:** 28,630 observations × 12 variables
- **Working subset:** 7,500 randomly sampled rows with 8 selected columns (seed: `338011`)

### Variables

| Variable | Description |
|---|---|
| `age` | Borrower age |
| `income` | Annual income |
| `home_ownership` | Home ownership status (`RENT`, `OWN`, `MORTGAGE`, `OTHER`) |
| `emp_length` | Employment length (in years) |
| `loan_intent` | Purpose of the loan |
| `loan_grade` | Loan grade (A–G) |
| `loan_amnt` | Loan amount |
| `loan_int_rate` | Interest rate |
| `loan_status` | **Target variable** — `0` = repaid (non-default), `1` = default |
| `loan_percent_income` | Loan amount as a percentage of income |
| `default_before` | Prior default history (`Y`/`N`) |
| `cred_hist_length` | Credit history length (years) |

**Class distribution (full dataset):**  
- Non-default (0): 22,427 (78.3%)  
- Default (1): 6,203 (21.7%)

---

## Methods

### 1. Exploratory Data Analysis
- Summary statistics and structure inspection
- Missing value check (none found)
- Class imbalance analysis for `loan_status`
- Frequency tables for categorical variables (`home_ownership`, `loan_intent`, `loan_grade`)
- Group-level summaries comparing defaulters vs. non-defaulters
- Visualizations: scatter plots, box plots, and pairwise correlation matrix (`ggpairs`)

### 2. Linear Probability Models (LPM)
Three OLS models were estimated with `loan_status` as the dependent variable:

- **Model 1 (`lm_1`):** Full model — `loan_percent_income`, `cred_hist_length`, `loan_amnt`, `age`, `income`, `home_ownership`, `loan_grade`
- **Model 2 (`lm_2`):** Log-transformed income — `log(income)`
- **Model 3 (`lm_3`):** Polynomial income + loan amount and home ownership — `income + income² + loan_amnt + home_ownership`

### 3. Logistic Regression
A logit model (`glm` with `binomial(link = "logit")`) was estimated using the same predictor set as `lm_1`.

### 4. Decision Tree
A classification tree (`rpart`, `method = "class"`) was built on the same predictor set. Variable importance was extracted and visualized using `rpart.plot`.

---

## Key Findings

- Borrowers in the **MORTGAGE** category repaid at over 6× the rate of defaulters; **OWN** borrowers repaid at over 13× the rate.
- `loan_percent_income` and `loan_grade` emerged as strong predictors of default across models.
- The decision tree's variable importance ranking highlights which features drive classification splits.

---

## Dependencies

```r
library(tidyverse)
library(GGally)
library(rpart)
library(rpart.plot)
library(cvms)
library(tibble)
```


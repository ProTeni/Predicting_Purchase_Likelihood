# Predicting Purchase Likelihood — A Classification Pipeline (and an Honest Result)

**A full binary-classification workflow on marketing engagement data — and a candid write-up of why the model didn't beat chance, and what that tells us.**

> This project runs the complete classification pipeline end to end. The headline finding is
> not a high score — it's a diagnosis: the available features don't carry enough signal to
> predict purchase likelihood, and no amount of tuning changes that. Knowing *why* a model
> fails is as valuable as making one succeed.

---

## Business problem
MarketGenix wants to target leads most likely to convert, using Google Analytics engagement
data (likes, shares, comments, clicks, time on platform, ad engagement, purchase history).
The task: predict **Purchase Likelihood** (binary) so marketing can prioritise high-potential leads.

## What I did
A deliberately thorough pipeline, to rule out technique as the limiting factor:
- **Cleaning:** dropped identifiers, regex-cleaned malformed values, fixed dtypes, checked nulls
- **Encoding:** one-hot encoded `Engagement_Level` with `drop_first=True` (avoids the dummy trap)
- **EDA:** correlation heatmap, class-balance check (data was roughly balanced), distribution plots
- **Transforms:** log-transformed skewed features
- **Scaler choice, justified:** ran an **IQR outlier analysis** to choose `RobustScaler` rather than
  picking one by default
- **Four modelling passes:** Logistic Regression → Logistic + `GridSearchCV` → `RidgeClassifierCV`
  → Ridge + **PCA** (dimensionality reduction)
- **Metric choice, justified:** prioritised **recall** (a missed lead costs more than a wasted contact)

## The honest result
Best configuration (Ridge + PCA): **Recall 0.72, F1 0.56, Accuracy 0.48.**

On *balanced* binary data, 0.48 accuracy is below random. What's actually happening: the model
defaults to predicting the positive class, which inflates recall while the model isn't really
*discriminating* between buyers and non-buyers. The high recall is hollow.

**Diagnosis:** the engagement features available here simply don't carry predictive signal for
purchase intent. This is a data limitation, not a modelling one — extensive tuning, regularisation,
and dimensionality reduction all failed to move accuracy off chance.

## Why this project is here
Recognising a model that hasn't worked — and being able to explain *why* — matters more than
reporting a number that looks good in isolation. This repo documents that judgement.

## What would actually improve it
Not more tuning. **Richer behavioural features** — session depth, recency/frequency of visits,
funnel stage, product-page dwell time — the kinds of signals that actually relate to purchase
intent. Better inputs, not a better algorithm.

## Tools
Python (pandas, NumPy, scikit-learn, seaborn, matplotlib) · LogisticRegression, RidgeClassifierCV,
GridSearchCV, PCA, RobustScaler

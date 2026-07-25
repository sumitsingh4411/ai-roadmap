---
title: "First ML Project"
stage: 3
order: 17
minutes: 90
difficulty: intermediate
prerequisites: ["feature-engineering", "trees-ensembles", "data-visualization", "real-datasets"]
tags: ["machine-learning", "project", "end-to-end"]
summary: "A full guided ML pipeline end to end: problem framing, EDA, cleaning, features, baseline, iteration, evaluation, and writing up results."
---

# First ML Project

## Why this matters

Every lesson in Stage 3 taught one piece of the machine learning workflow
in isolation — a model, a metric, a technique. Real projects don't arrive
in pieces. They arrive as one messy question ("which customers are about
to leave?") and it's on you to turn that into framed problem, cleaned
data, engineered features, a trained model, an honest evaluation, and a
result someone else can actually use. This lesson is that whole pipeline,
start to finish, on one dataset, using nothing you haven't already learned
in this roadmap — the payoff for everything since Stage 1.

## The concept

**Problem framing comes before any code.** Before touching data, write
down, in plain language: what are you predicting, for whom, and what
decision will the prediction actually change? "Predict which customers
will cancel their subscription next month, so the retention team can offer
them a discount before they leave" is a framed problem — it names the
label (`churned`), the unit of prediction (one customer), and the reason
the prediction needs to exist at all. A model with 95% accuracy that
answers a question nobody asked is a wasted project; five minutes of
framing up front prevents that.

**EDA (exploratory data analysis) comes before cleaning.** You cannot
decide how to fix a dataset's problems until you know what they are.
EDA — looking at shape, dtypes, missing values, class balance, and how the
label relates to a few candidate features — is reconnaissance, not
decoration. Everything you noticed in the real-datasets and
data-visualization lessons (`.head()`, `.info()`, `.isna().sum()`,
`groupby`, a histogram) gets used here, on a dataset you've genuinely never
seen before.

**Cleaning fixes what EDA found — deliberately, not automatically.**
Every fix should trace back to something EDA actually surfaced: a column
with the wrong dtype, a chunk of missing values, duplicate rows, an
outlier. The real-datasets lesson's warning matters even more here: an
outlier-detection rule like IQR flags *candidates*, not verdicts — you
still have to look at what got flagged before deciding whether to fix it,
drop it, or leave it alone.

**Features turn cleaned columns into what a model can learn from.**
Categorical columns get encoded, dates get mined for structure, everything
gets assembled into the `X` matrix and `y` vector from the ML Fundamentals
lesson, and the train/test split happens *before* any model sees the data.

**A baseline exists so "improvement" means something.** The single most
important number in this whole pipeline might be the very first one: a
dead-simple model — often as simple as "always predict the most common
class" — establishes a floor. Every model you try after that either beats
the baseline by a meaningful margin, or it hasn't actually learned
anything useful yet. Skipping a baseline is how projects end up impressed
by a score that a coin flip would have matched.

**Iteration means trying real alternatives and comparing them honestly.**
Fit more than one model, cross-validate each of them the same way,
compare the numbers. A crucial, humbling thing genuinely happens in real
projects and happens later in this lesson too: the more complex model does
not always win. Knowing that in advance — and trusting the cross-validated
numbers over your intuition about which model *should* be smarter — is
part of doing this honestly.

**Evaluation means picking the metric the problem actually cares about.**
The model evaluation lesson's whole point resurfaces here for real:
accuracy alone can hide a model that's useless at the one thing you built
it for. Which metric matters — precision, recall, ROC-AUC — depends on
what mistake is more expensive in your specific problem, and that's a
business judgement call as much as a technical one.

**A write-up is part of the deliverable, not an afterthought.** A model
nobody can understand, reproduce, or trust is a model nobody will use. A
short written summary — what you predicted, what you tried, what worked,
what you'd do next — is what turns a script into a project someone else
(including future you) can pick back up.

## In code

**Problem framing.** Predict whether a subscription customer will churn
(cancel) next month, from their contract details and usage — so a
retention team can proactively reach out to the highest-risk customers.
The label is `churned` (0/1); each row is one customer.

**EDA — first look at the (synthetic, deliberately messy) dataset:**

```python
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
n = 1000
contract_type = rng.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20])
tech_support = rng.choice(["Yes", "No"], size=n, p=[0.45, 0.55])
tenure_months = np.clip(rng.exponential(24, n), 0, 72).round(0)
monthly_charges = np.clip(rng.normal(70, 25, n), 20, 150).round(2)
signup_date = pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 900, n), unit="D")
contract_risk = np.select(
    [contract_type == "Month-to-month", contract_type == "One year", contract_type == "Two year"],
    [0.9, 0.0, -0.5])
support_risk = np.where(tech_support == "No", 0.35, -0.15)
tenure_risk = -0.02 * tenure_months
charge_risk = 0.006 * (monthly_charges - 70)
churn_logit = -1.3 + contract_risk + support_risk + tenure_risk + charge_risk + rng.normal(0, 0.3, n)
churned = (rng.random(n) < 1 / (1 + np.exp(-churn_logit))).astype(int)
total_charges = (tenure_months * monthly_charges + rng.normal(0, 50, n)).round(2)
df = pd.DataFrame({
    "customer_id": np.arange(1, n + 1), "tenure_months": tenure_months,
    "contract_type": contract_type, "tech_support": tech_support,
    "monthly_charges": monthly_charges, "total_charges": total_charges,
    "signup_date": signup_date, "churned": churned,
})
# Realistic mess: brand-new customers have a blank total_charges (a real
# quirk of this exact shape in the well-known Telco Customer Churn
# dataset), some tech_support values were never recorded, and a handful of
# rows are exact duplicates or have a typo'd extra digit in monthly_charges
rng2 = np.random.default_rng(7)
df["total_charges"] = df["total_charges"].astype(object)
df.loc[df["tenure_months"] == 0, "total_charges"] = " "
missing_idx = rng2.choice(df.index, size=40, replace=False)
df.loc[missing_idx, "tech_support"] = np.nan
dupes = df.sample(5, random_state=7)
df = pd.concat([df, dupes], ignore_index=True)
typo_idx = rng2.choice(df.index, size=3, replace=False)
df.loc[typo_idx, "monthly_charges"] = df.loc[typo_idx, "monthly_charges"] * 10
df = df.sample(frac=1, random_state=7).reset_index(drop=True)

print(df.shape)
print(df.head())
print()
df.info()
```

```
(1005, 8)
   customer_id  tenure_months  ... signup_date churned
0          114            2.0  ...  2024-03-09       0
1          793           11.0  ...  2022-11-19       0
2          741           28.0  ...  2023-12-26       0
3          432           31.0  ...  2022-11-16       0
4          243            2.0  ...  2024-04-03       1

[5 rows x 8 columns]

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1005 entries, 0 to 1004
Data columns (total 8 columns):
 #   Column           Non-Null Count  Dtype         
---  ------           --------------  -----         
 0   customer_id      1005 non-null   int64         
 1   tenure_months    1005 non-null   float64       
 2   contract_type    1005 non-null   object        
 3   tech_support     965 non-null    object        
 4   monthly_charges  1005 non-null   float64       
 5   total_charges    1005 non-null   object        
 6   signup_date      1005 non-null   datetime64[ns]
 7   churned          1005 non-null   int64         
dtypes: datetime64[ns](1), float64(2), int64(2), object(3)
memory usage: 62.9+ KB
```

`total_charges` reading as `object` (text) instead of a number is already
a red flag `.info()` caught for us — worth remembering for the cleaning
step.

**EDA — the label's balance, and how it relates to two candidate
features** (continuing with the same `df`):

```python
print(df["churned"].value_counts())
print("churn rate:", round(df["churned"].mean(), 4))
print()
print("churn rate by contract type:")
print(df.groupby("contract_type")["churned"].mean().round(3).sort_values(ascending=False))
print()
print("churn rate by tech support:")
print(df.groupby("tech_support")["churned"].mean().round(3))

churn_by_contract = df.groupby("contract_type")["churned"].mean().reindex(
    ["Month-to-month", "One year", "Two year"])
fig, ax = plt.subplots()
churn_by_contract.plot(kind="bar", ax=ax, color=["#c0392b", "#e67e22", "#27ae60"])
ax.set_xlabel("Contract type")
ax.set_ylabel("Churn rate")
ax.set_title("Churn rate by contract type")
ax.set_ylim(0, 0.5)
fig.tight_layout()
fig.savefig("churn_by_contract.png")
print("saved churn_by_contract.png")
```

```
churned
0    737
1    268
Name: count, dtype: int64
churn rate: 0.2667

churn rate by contract type:
contract_type
Month-to-month    0.352
One year          0.189
Two year          0.131
Name: churned, dtype: float64

churn rate by tech support:
tech_support
No     0.311
Yes    0.214
Name: churned, dtype: float64
saved churn_by_contract.png
```

Two findings that will matter for modelling: the label is real but only
mildly imbalanced (26.7% churn — nowhere near the 3% fraud scenario from
the model evaluation lesson, but still worth watching), and both
`contract_type` and `tech_support` show a clear, monotonic relationship
with churn — exactly the kind of signal EDA is meant to surface before you
build anything.

**Cleaning — run the checklist from the real-datasets lesson first**
(continuing with the same `df`):

```python
print("duplicate rows:", df.duplicated().sum())
print("missing per column:\n", df.isna().sum())
print("dtypes:\n", df.dtypes)
```

```
duplicate rows: 5
missing per column:
 customer_id         0
tenure_months       0
contract_type       0
tech_support       40
monthly_charges     0
total_charges       0
signup_date         0
churned             0
dtype: int64
dtypes:
 customer_id                 int64
tenure_months             float64
contract_type              object
tech_support               object
monthly_charges           float64
total_charges              object
signup_date        datetime64[ns]
churned                     int64
dtype: object
```

Four distinct problems, four distinct fixes needed: a wrong dtype, missing
values, duplicate rows, and — still to be found — outliers.

**Cleaning — fix what the checklist found** (continuing with the same
`df`):

```python
# total_charges is 'object' because brand-new customers (tenure=0) have a
# blank " " instead of a number - convert to numeric, blanks become NaN,
# then it's fair to fill them with 0 (they've been charged nothing yet)
df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
print("total_charges NaN after conversion:", df["total_charges"].isna().sum())
print("...all from tenure_months == 0?", (df.loc[df["total_charges"].isna(), "tenure_months"] == 0).all())
df["total_charges"] = df["total_charges"].fillna(0)

# tech_support missing: not recorded at signup - keep it explicit as its
# own category instead of guessing, so the model can learn from "unknown"
df["tech_support"] = df["tech_support"].fillna("Unknown")

before = len(df)
df = df.drop_duplicates()
print("dropped", before - len(df), "duplicate rows")

# outliers: the IQR rule is a starting point, not a verdict - it flags
# candidates, a human still has to look at them
q1, q3 = df["monthly_charges"].quantile([0.25, 0.75])
iqr = q3 - q1
upper = q3 + 1.5 * iqr
flagged = df[df["monthly_charges"] > upper]
print(f"IQR flags {len(flagged)} rows above {upper:.2f}:")
print(sorted(flagged["monthly_charges"].tolist()))
# Sorted, the split is obvious: six values sit just above the whisker in
# the 138-147 range - unusually high, but plausible for a premium plan.
# Three values (689, 811, 978) are in a different world entirely, ten
# times too big - decimal-point-style data entry typos. We only touch the
# three that are actually impossible, using domain knowledge (this
# business's plans top out well under $200), not the IQR cutoff alone.
typo_mask = df["monthly_charges"] > 300
print("true typos fixed:", typo_mask.sum())
df.loc[typo_mask, "monthly_charges"] = df.loc[typo_mask, "monthly_charges"] / 10
print("max monthly_charges after fix:", df["monthly_charges"].max())

print()
print(df.shape)
print(df.isna().sum().sum(), "missing values remain")
```

```
total_charges NaN after conversion: 23
...all from tenure_months == 0? True
dropped 5 duplicate rows
IQR flags 9 rows above 136.96:
[138.49, 139.99, 140.36, 141.13, 145.79, 147.21, 689.3000000000001, 811.9, 978.4000000000001]
true typos fixed: 3
max monthly_charges after fix: 147.21

(1000, 8)
0 missing values remain
```

Blindly "fixing" all 9 IQR-flagged rows would have wrecked six legitimate
high-but-real charges by dividing them by 10 too — this is exactly the
"look before you fix" habit the real-datasets lesson warned about, paying
off on a case that actually mattered.

**Features — encode, split, scale** (continuing with the same `df`):

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

feature_df = pd.get_dummies(
    df.drop(columns=["customer_id", "signup_date", "churned"]),
    columns=["contract_type", "tech_support"], drop_first=True,
)
print("feature columns:", list(feature_df.columns))

X = feature_df.values
y = df["churned"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("train:", X_train.shape, " test:", X_test.shape)

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

```
feature columns: ['tenure_months', 'monthly_charges', 'total_charges', 'contract_type_One year', 'contract_type_Two year', 'tech_support_Unknown', 'tech_support_Yes']
train: (800, 7)  test: (200, 7)
```

**Baseline** (continuing with the same split):

```python
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

dummy = DummyClassifier(strategy="most_frequent").fit(X_train_scaled, y_train)
print("dummy accuracy:", round(accuracy_score(y_test, dummy.predict(X_test_scaled)), 4),
      "(always predicts 'stays' - our floor to beat)")

logreg = LogisticRegression(max_iter=2000)
logreg_acc = cross_val_score(logreg, X_train_scaled, y_train, cv=cv, scoring="accuracy").mean()
logreg_auc = cross_val_score(logreg, X_train_scaled, y_train, cv=cv, scoring="roc_auc").mean()
print(f"logistic regression  cv accuracy={logreg_acc:.4f}  cv ROC-AUC={logreg_auc:.4f}")
```

```
dummy accuracy: 0.735 (always predicts 'stays' - our floor to beat)
logistic regression  cv accuracy=0.7500  cv ROC-AUC=0.6749
```

The dummy floor is 73.5% accuracy — a reminder that "75% accurate" on its
own, without this floor, would have sounded far more impressive than it
actually is.

**Iterate — try a random forest and gradient boosting** (continuing with
the same split):

```python
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

candidates = {
    "Logistic Regression": (logreg, X_train_scaled),
    "Random Forest": (RandomForestClassifier(n_estimators=300, random_state=42), X_train),
    "Gradient Boosting": (HistGradientBoostingClassifier(random_state=42), X_train),
}
for name, (model, X_fit) in candidates.items():
    acc = cross_val_score(model, X_fit, y_train, cv=cv, scoring="accuracy").mean()
    auc = cross_val_score(model, X_fit, y_train, cv=cv, scoring="roc_auc").mean()
    print(f"{name:20s} cv accuracy={acc:.4f}  cv ROC-AUC={auc:.4f}")
```

```
Logistic Regression  cv accuracy=0.7500  cv ROC-AUC=0.6749
Random Forest        cv accuracy=0.7087  cv ROC-AUC=0.6353
Gradient Boosting    cv accuracy=0.7063  cv ROC-AUC=0.6072
```

Here's the honest, humbling finding this lesson promised: the fancier
models *lose* to plain logistic regression on this dataset. That's a
completely legitimate outcome, not a bug — this churn data was built from
mostly additive, close-to-linear relationships (each factor pushes risk up
or down somewhat independently), which is exactly the shape logistic
regression is built for. Trees and boosting earn their keep when
relationships are more tangled and nonlinear; here, the simpler model
matches the true structure of the problem better. **The model that wins
cross-validation is the one you ship** — not the one that sounds more
impressive.

**Evaluation — the chosen model (logistic regression) on the held-out
test set, touched for the first and only time:**

```python
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

final_model = LogisticRegression(max_iter=2000).fit(X_train_scaled, y_train)
test_predictions = final_model.predict(X_test_scaled)
test_probs = final_model.predict_proba(X_test_scaled)[:, 1]

print(confusion_matrix(y_test, test_predictions))
print()
print(classification_report(y_test, test_predictions, target_names=["stayed", "churned"]))
print("test ROC-AUC:", round(roc_auc_score(y_test, test_probs), 4))

print()
print("coefficients:")
for name, coef in sorted(zip(feature_df.columns, final_model.coef_[0]), key=lambda t: -abs(t[1])):
    direction = "higher churn risk" if coef > 0 else "lower churn risk"
    print(f"  {name:24s} {coef:7.3f}  ({direction})")
```

```
[[140   7]
 [ 48   5]]

              precision    recall  f1-score   support

      stayed       0.74      0.95      0.84       147
     churned       0.42      0.09      0.15        53

    accuracy                           0.72       200
   macro avg       0.58      0.52      0.49       200
weighted avg       0.66      0.72      0.66       200

test ROC-AUC: 0.6909

coefficients:
  total_charges             -0.662  (lower churn risk)
  contract_type_Two year    -0.479  (lower churn risk)
  contract_type_One year    -0.360  (lower churn risk)
  tech_support_Yes          -0.294  (lower churn risk)
  monthly_charges            0.230  (higher churn risk)
  tenure_months              0.099  (higher churn risk)
  tech_support_Unknown      -0.018  (lower churn risk)
```

Two things worth reading carefully in this output. First, `recall` for
`churned` is 0.09 — the model catches only 5 of 53 actual churners at the
default threshold, despite 72% overall accuracy. That's the imbalanced-data
lesson's whole point, showing up again in a real project: a retention team
relying on this model as shipped would miss 90% of the customers they most
need to reach. Second, `tenure_months` shows a *positive* coefficient
("higher churn risk" the longer someone stays), which looks backwards —
longer-tenured customers should be safer, not riskier. This is the exact
multicollinearity trap from the regression lesson: `total_charges` is
almost a direct function of `tenure_months × monthly_charges` (they
correlate at 0.87 in this data), so the two features compete for credit,
and `tenure_months`'s own coefficient sign becomes unreliable. The model's
*predictions* are still fine — multicollinearity mainly damages
*interpreting individual coefficients*, not overall accuracy.

**One more iteration, driven directly by that recall problem** — does
`class_weight="balanced"` fix it?

```python
balanced_model = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_train_scaled, y_train)
balanced_predictions = balanced_model.predict(X_test_scaled)
print(confusion_matrix(y_test, balanced_predictions))
print(classification_report(y_test, balanced_predictions, target_names=["stayed", "churned"]))
```

```
[[98 49]
 [16 37]]
              precision    recall  f1-score   support

      stayed       0.86      0.67      0.75       147
     churned       0.43      0.70      0.53        53

    accuracy                           0.68       200
   macro avg       0.64      0.68      0.64       200
weighted avg       0.75      0.68      0.69       200
```

Recall for `churned` jumps from 9% to 70% — the retention team would now
catch most at-risk customers — at a real, visible cost: overall accuracy
*drops* from 72% to 68%, because catching more real churners means
tolerating more false alarms (customers wrongly flagged as at-risk). Which
model actually ships depends on a business call, not a technical one: if a
false alarm costs a $20 discount offer and a missed churner costs a lost
customer worth hundreds of dollars, the balanced model is the obvious
choice, and the write-up below needs to say so explicitly.

**The write-up.** A results write-up doesn't need to be long, but it needs
to answer five questions so someone who never saw your screen can trust
and reuse your work:

```markdown
# Customer Churn Prediction

## Problem
Predict which subscription customers will churn next month, from contract
and usage data, so the retention team can reach out before they leave.

## Data
1,000 customers, 8 columns (synthetic, modelled on a real telecom churn
dataset). Churn rate 26.7%. Cleaned: fixed a text/number dtype mismatch,
filled 40 missing tech_support values as "Unknown", dropped 5 duplicate
rows, corrected 3 data-entry typos in monthly_charges using domain
knowledge rather than a blanket outlier rule.

## Method
Baseline: DummyClassifier, 73.5% accuracy. Compared logistic regression,
random forest, and gradient boosting with 5-fold cross-validation.
Logistic regression won (75.0% cv accuracy, 0.67 cv ROC-AUC) - the
relationships in this data are close to linear, so the simpler model
matched the problem better than the ensembles did.

## Result
Final model: logistic regression, test accuracy 72%, ROC-AUC 0.69. At the
default threshold, recall on churners was only 9% - unacceptable for the
business goal. Switching to class_weight="balanced" raised recall to 70%
at the cost of accuracy (68%) and precision (43%). Recommended for
deployment: the balanced model, since a missed churner is far more
expensive than a false alarm.

## What I'd try next
More recent usage features (support tickets, login frequency) would
likely help more than further model tuning - the current features cap
out around ROC-AUC 0.69 regardless of algorithm.
```

## Build this

Take this lesson's pipeline and run it on a dataset you find yourself —
[Kaggle Datasets](https://www.kaggle.com/datasets) or the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/) both work,
and a real churn, loan-default, or customer-conversion dataset makes the
closest match to this lesson. Work through every stage: frame the problem
in one sentence, run the five-step EDA checklist, clean what it finds,
engineer at least two features, establish a baseline, compare at least two
real models with cross-validation, evaluate with the metric that fits the
problem (not just accuracy), and write it up using the template above.
Push the project to a public GitHub repository with a `README.md` built
from that write-up, and include the code that produced your numbers.

**Stretch:** add one paragraph to your README specifically about a
decision where you had to trade one metric against another (like the
accuracy-vs-recall tradeoff in this lesson), and justify the choice you'd
actually recommend shipping, in business terms, not just numbers.

## Go deeper

- [Kaggle: Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — the real dataset this lesson's synthetic data was modelled on, if you want to run this exact pipeline on real numbers.
- [Google's Machine Learning Crash Course: Framing an ML problem](https://developers.google.com/machine-learning/problem-framing) — a deeper treatment of the problem-framing step this lesson opened with.
- [scikit-learn: Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html) — how to combine cleaning, encoding, and modelling into one reusable object, once you're doing this regularly.
- [Made With ML: Machine Learning Design](https://madewithml.com/) — a free, practical course on structuring ML projects end to end, covering everything from this lesson in more depth.
- [Kaggle Learn](https://www.kaggle.com/learn) — free micro-courses with graded exercises across every stage of this pipeline, if any one step needs more practice.

**Next:** [Neural Networks](18-neural-networks.md)


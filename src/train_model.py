
import json
import os, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "customer_churn.csv")
MODEL_DIR = os.path.join(ROOT, "models")
REPORT_DIR = os.path.join(ROOT, "reports")
os.makedirs(MODEL_DIR, exist_ok=True); os.makedirs(REPORT_DIR, exist_ok=True)

def make_dataset(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 76, n)
    tenure = rng.integers(1, 73, n)
    monthly = np.round(rng.uniform(20, 180, n), 2)
    support = rng.poisson(1.5, n)
    logins = np.maximum(1, rng.poisson(14, n))
    late_pay = rng.binomial(1, 0.18, n)
    contract = rng.choice(["Month-to-month","One year","Two year"], n, p=[.55,.27,.18])
    internet = rng.choice(["DSL","Fiber optic","None"], n, p=[.35,.48,.17])
    payment = rng.choice(["Electronic check","Credit card","Bank transfer","Mailed check"], n)
    paperless = rng.binomial(1, .62, n)
    dependents = rng.binomial(1, .25, n)
    satisfaction = rng.integers(1, 11, n)
    data_usage = np.round(rng.gamma(3, 20, n), 1)

    score = (
        -1.7
        + 0.035*(monthly-80)
        - 0.018*tenure
        + 0.38*support
        + 0.12*(12-logins)
        + 0.9*late_pay
        + 0.55*(contract=="Month-to-month")
        - 0.45*(contract=="Two year")
        + 0.38*(internet=="Fiber optic")
        - 0.35*(internet=="None")
        + 0.18*(payment=="Electronic check")
        - 0.25*paperless
        - 0.35*dependents
        - 0.08*(satisfaction-5)
        + 0.12*(data_usage<20)
        + rng.normal(0, .65, n)
    )
    prob = 1/(1+np.exp(-score))
    churn = rng.binomial(1, prob)
    df = pd.DataFrame({
        "age":age, "tenure_months":tenure, "monthly_charges":monthly,
        "support_tickets":support, "monthly_logins":logins,
        "late_payment":late_pay, "contract_type":contract,
        "internet_service":internet, "payment_method":payment,
        "paperless_billing":paperless, "has_dependents":dependents,
        "satisfaction_score":satisfaction, "data_usage_gb":data_usage,
        "churn":churn
    })
    return df

def main():
    if not os.path.exists(DATA):
        make_dataset().to_csv(DATA, index=False)
    df = pd.read_csv(DATA)
    X = df.drop(columns=["churn"])
    y = df["churn"]

    cat = X.select_dtypes(include=["object"]).columns.tolist()
    num = X.select_dtypes(exclude=["object"]).columns.tolist()

    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat)
    ])

    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300,max_depth=10,min_samples_leaf=3,
                                                 class_weight="balanced",random_state=42,n_jobs=-1)
    }
    results = []
    fitted = {}
    for name, clf in models.items():
        pipe = Pipeline([("preprocess", prep), ("select", SelectKBest(mutual_info_classif, k="all")), ("model", clf)])
        pipe.fit(Xtr,ytr)
        p=pipe.predict(Xte); proba=pipe.predict_proba(Xte)[:,1]
        results.append({
            "Model":name,
            "Accuracy":accuracy_score(yte,p),
            "Precision":precision_score(yte,p),
            "Recall":recall_score(yte,p),
            "F1":f1_score(yte,p),
            "ROC_AUC":roc_auc_score(yte,proba)
        })
        fitted[name]=pipe

    res=pd.DataFrame(results).sort_values("ROC_AUC",ascending=False)
    best_name=res.iloc[0]["Model"]; best=fitted[best_name]
    with open(os.path.join(MODEL_DIR,"churn_model.pkl"),"wb") as f: pickle.dump(best,f)
    with open(os.path.join(MODEL_DIR,"model_info.json"),"w") as f:
        json.dump({"model":best_name,"features":X.columns.tolist()},f,indent=2)

    pred=best.predict(Xte); proba=best.predict_proba(Xte)[:,1]
    cm=confusion_matrix(yte,pred).tolist()
    report=classification_report(yte,pred,target_names=["Retain","Churn"])
    lines=["# Customer Churn Prediction — Performance Report","",
           f"Best model: **{best_name}**","",
           "## Model comparison", "", res.to_markdown(index=False), "",
           "## Confusion matrix", "", str(cm), "",
           "## Classification report","", "```", report, "```", "",
           "## Dataset", f"- Rows: {len(df):,}", f"- Churn rate: {y.mean():.2%}",
           "- Train/test split: 80/20 stratified", "- Random seed: 42"]
    open(os.path.join(REPORT_DIR,"performance_report.md"),"w",encoding="utf-8").write("\n".join(lines))
    print(res.to_string(index=False))
    print("Best:", best_name)

if __name__=="__main__":
    main()

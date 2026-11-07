"""Compare two tabular classifiers on synthetic examples."""
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from .data import generate_flows

def run(seed=42, n=2000, save=True):
    data=generate_flows(n,seed)
    X=data.drop(columns='label'); y=data.label
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,stratify=y,random_state=seed)
    models={'logistic_regression':make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000,random_state=seed)),
            'random_forest':RandomForestClassifier(n_estimators=100,random_state=seed,n_jobs=1)}
    results={}
    for name,model in models.items():
        model.fit(X_train,y_train)
        pred=model.predict(X_test); scores=model.predict_proba(X_test)[:,1]
        results[name]={
            'accuracy':float(accuracy_score(y_test,pred)),
            'precision':float(precision_score(y_test,pred,zero_division=0)),
            'recall':float(recall_score(y_test,pred,zero_division=0)),
            'f1':float(f1_score(y_test,pred,zero_division=0)),
            'roc_auc':float(roc_auc_score(y_test,scores)),
        }
    if save:
        out=Path('outputs');out.mkdir(exist_ok=True)
        (out/'metrics.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))
    return results

if __name__=='__main__': run()

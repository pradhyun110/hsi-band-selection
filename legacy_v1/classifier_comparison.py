"""
Classifier comparison on the SAME selected-band sets (no change to band selection).
Primary protocol: the fixed split used by the band search (stratified 20/80, seed 42, z-score on train).
Stability: (a) across the 6 algorithm subsets, (b) across 5 repeated stratified splits (seeds 1..5)
using the same bands. NOTE: bands were chosen using the seed-42 test split, so seed-42 numbers are
slightly optimistic for every classifier equally; repeated-split numbers are the less biased check.
"""
import sys, json, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data_setup import load_fixed_setup, stratified_split, DATA_DIR
from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import cohen_kappa_score, confusion_matrix

OUT = __import__("pathlib").Path(__file__).resolve().parent.parent / "results" / "legacy"
subsets = json.load(open(OUT/"best_band_subsets.json"))
setup = load_fixed_setup()
cand = setup["candidate_bands"]

def make():
    return {
     "SVM-RBF": lambda: SVC(kernel="rbf", C=10, gamma="scale"),
     "SVM-Linear": lambda: LinearSVC(C=1.0, dual=False, max_iter=5000),
     "LDA": lambda: LDA(),
     "LogReg": lambda: LogisticRegression(C=1.0, max_iter=2000),
     "RandomForest": lambda: RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=1),
     "KNN(k=5)": lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=1),
     "GaussianNB": lambda: GaussianNB(),
    }

def metrics(y, p):
    cm = confusion_matrix(y, p)
    with np.errstate(invalid="ignore", divide="ignore"):
        aa = np.nanmean(np.diag(cm)/cm.sum(1))
    return (p==y).mean(), aa, cohen_kappa_score(y, p)

def model_cost(name, m, k, n_train):
    """returns (params stored, MACs/sample, memory bytes at float32, memory bytes at int16)"""
    C = 16
    if name == "SVM-RBF":
        nsv = int(m.n_support_.sum()); params = nsv*k + m.dual_coef_.size + m.intercept_.size
        macs = nsv*k + nsv*(C-1)         # kernel dot/dist + coefficient accumulation
    elif name in ("SVM-Linear","LogReg"):
        params = m.coef_.size + m.intercept_.size; macs = m.coef_.size
    elif name == "LDA":
        params = m.coef_.size + m.intercept_.size; macs = m.coef_.size
    elif name == "RandomForest":
        nodes = sum(e.tree_.node_count for e in m.estimators_)
        params = nodes*2          # feature id + threshold per node (leaf values separate, approx)
        macs = 0                  # comparisons only
        depth = np.mean([e.tree_.max_depth for e in m.estimators_])
        macs = int(len(m.estimators_)*depth)   # comparisons/sample (not MACs)
        params = nodes*3          # feat, thresh, child/leaf-class
    elif name == "KNN(k=5)":
        params = n_train*k; macs = n_train*k
    elif name == "GaussianNB":
        params = m.theta_.size + m.var_.size + m.class_prior_.size; macs = 2*m.theta_.size
    return int(params), int(macs), int(params*4), int(params*2)

def run_one(name, factory, Xtr, ytr, Xte, yte, k, ntrain, reps=3):
    tt=[]; 
    for _ in range(reps):
        m = factory(); t=time.perf_counter(); m.fit(Xtr,ytr); tt.append(time.perf_counter()-t)
    it=[]
    for _ in range(reps):
        t=time.perf_counter(); p=m.predict(Xte); it.append(time.perf_counter()-t)
    oa,aa,kap = metrics(yte,p)
    params,macs,m32,m16 = model_cost(name,m,k,ntrain)
    return dict(OA=oa,AA=aa,Kappa=kap,train_s=np.median(tt),infer_s=np.median(it),
                infer_us_per_px=np.median(it)/len(yte)*1e6,params=params,ops_per_px=macs,
                mem_KB_fp32=m32/1024,mem_KB_int16=m16/1024)

rows=[]
# ---- Part A: fixed split, 6 algorithm subsets
for algo, d in subsets.items():
    orig = np.array(d["band_indices_in_original_200"])
    cols = np.searchsorted(cand, orig)
    assert (cand[cols]==orig).all()
    Xtr, Xte = setup["X_train"][:,cols], setup["X_test"][:,cols]
    for name, f in make().items():
        r = run_one(name,f,Xtr,setup["y_train"],Xte,setup["y_test"],len(cols),len(Xtr))
        r.update(subset=algo,k=len(cols),protocol="fixed_split42"); r["classifier"]=name
        rows.append(r); print(algo,name,round(r["OA"],4),round(r["train_s"],3),round(r["infer_s"],3),flush=True)
dfA = pd.DataFrame(rows); dfA.to_csv(OUT/"classifier_comparison_raw.csv",index=False)

# ---- Part B: repeated splits, same bands (SA subset = rank-1 algorithm, and GWO as a second)
cube = np.load(DATA_DIR/"indianpinearray.npy").astype(np.float64); gt=np.load(DATA_DIR/"IPgt.npy")
X_all = cube.reshape(-1,cube.shape[2]); y_all=gt.reshape(-1); msk=y_all!=0
X_lab, y_lab = X_all[msk], y_all[msk]
rowsB=[]
for algo in ["SA","Jaya"]:
    orig = np.array(subsets[algo]["band_indices_in_original_200"])
    for s in [1,2,3,4,5]:
        tr,te = stratified_split(y_lab,0.80,s)
        Xtr0,Xte0 = X_lab[tr][:,orig],X_lab[te][:,orig]
        mu,sd = Xtr0.mean(0),Xtr0.std(0); sd[sd==0]=1
        Xtr,Xte=(Xtr0-mu)/sd,(Xte0-mu)/sd
        for name,f in make().items():
            r=run_one(name,f,Xtr,y_lab[tr],Xte,y_lab[te],len(orig),len(tr),reps=1)
            rowsB.append(dict(subset=algo,split_seed=s,classifier=name,OA=r["OA"],AA=r["AA"],Kappa=r["Kappa"]))
    print("done splits",algo,flush=True)
dfB=pd.DataFrame(rowsB); dfB.to_csv(OUT/"classifier_repeated_splits_raw.csv",index=False)

# ---- Summaries
order=list(make().keys())
A=dfA.groupby("classifier").agg(OA_mean=("OA","mean"),OA_std_across_subsets=("OA","std"),AA=("AA","mean"),Kappa=("Kappa","mean"),
   train_s=("train_s","mean"),infer_s=("infer_s","mean"),us_per_px=("infer_us_per_px","mean"),params=("params","mean"),
   ops_per_px=("ops_per_px","mean"),mem_KB_fp32=("mem_KB_fp32","mean"),mem_KB_int16=("mem_KB_int16","mean")).loc[order]
B=dfB.groupby(["classifier","subset"])["OA"].agg(["mean","std"]).reset_index().groupby("classifier").agg(
   OA_repeated_mean=("mean","mean"),OA_repeated_std=("std","mean")).loc[order]
S=A.join(B).round(4); S.to_csv(OUT/"classifier_comparison_summary.csv")
pd.set_option("display.width",250); print(S.to_string())

import sys; sys.argv=['x']
exec(open('fitness_comparison.py').read().split("# eval cost")[0])
rows=[]
for lam in [0.0,0.01,0.05,0.1,0.2]:
    def f(m, lam=lam):
        c=np.where(m)[0]; a=acc_val(c); return a-lam*len(c)/N, a
    for aname,alg in [("SA",run_sa),("Jaya",run_jaya)]:
        for seed in range(5):
            r=alg(f,N,pop_size=25,max_iter=50,seed=seed); c=np.where(r["best_mask"])[0]
            svm=SVC(kernel="rbf",C=10,gamma="scale").fit(Xtr[:,c],ytr).score(Xte[:,c],yte)
            lr=LogisticRegression(max_iter=2000).fit(Xtr[:,c],ytr).score(Xte[:,c],yte)
            rows.append(dict(lam=lam,algo=aname,seed=seed,k=len(c),svm=svm,logreg=lr))
    print("lam",lam,flush=True)
d=pd.DataFrame(rows); d.to_csv(OUT/"fitness_lambda_sweep_raw.csv",index=False)
print(d.groupby("lam").agg(k=("k","mean"),k_std=("k","std"),svm=("svm","mean"),svm_std=("svm","std"),logreg=("logreg","mean")).round(4).to_string())

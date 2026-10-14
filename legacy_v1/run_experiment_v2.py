"""Rerun of the 6-algorithm comparison with the train-only fitness F1 = A_val - 0.01*k/N.
Everything else identical to run_experiment.py (164 candidates, pop 25 x iter 50 = 1250 evals, seeds 0-4)."""
import sys, time, json
exec(open('fitness_comparison.py').read().split("# eval cost")[0])
from algorithms import ALGORITHMS
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows=[]; hist={}; best={}
for an, alg in ALGORITHMS.items():
    H=[]
    for seed in range(5):
        t=time.perf_counter(); r=alg(F1,N,pop_size=25,max_iter=50,seed=seed); wt=time.perf_counter()-t
        c=np.where(r["best_mask"])[0]; Xa,Xb=Xtr[:,c],Xte[:,c]
        knn=KNeighborsClassifier(5).fit(Xa,ytr).score(Xb,yte)
        svm=SVC(kernel="rbf",C=10,gamma="scale").fit(Xa,ytr).score(Xb,yte)
        lr=LogisticRegression(max_iter=2000).fit(Xa,ytr).score(Xb,yte)
        rows.append(dict(algorithm=an,seed=seed,n_evals=r["n_evals"],k=len(c),fitness=r["best_fitness"],acc_val=r["best_acc"],
            knn=knn,svm=svm,logreg=lr,redundancy=redund(c),time_s=wt))
        H.append(r["history_fitness"])
        if an not in best or knn>best[an][0]: best[an]=(knn,svm,[int(S["candidate_bands"][i]) for i in c])
    hist[an]=np.vstack(H); print(an,"done",flush=True)
df=pd.DataFrame(rows); df.to_csv(OUT/"v2_algo_comparison_raw.csv",index=False)
g=df.groupby("algorithm").agg(k=("k","mean"),k_std=("k","std"),fit=("fitness","mean"),fit_std=("fitness","std"),
  accval=("acc_val","mean"),knn=("knn","mean"),knn_std=("knn","std"),knn_best=("knn","max"),svm=("svm","mean"),svm_std=("svm","std"),
  svm_best=("svm","max"),lr=("logreg","mean"),red=("redundancy","mean"),time=("time_s","mean"),evals=("n_evals","mean")).round(4)
g=g.sort_values(["knn","knn_std","k"],ascending=[False,True,True]); g.insert(0,"rank",range(1,7)); g.to_csv(OUT/"v2_algo_ranking.csv")
pd.set_option("display.width",250); print(g.to_string())
np.savez(OUT/"v2_convergence.npz",**hist); json.dump({a:dict(knn=b[0],svm=b[1],bands=b[2]) for a,b in best.items()},open(OUT/"v2_best_subsets.json","w"),indent=1)
fig,ax=plt.subplots(1,3,figsize=(15,4)); COL={a:f"C{i}" for i,a in enumerate(ALGORITHMS)}
for a,h in hist.items():
    m=h.mean(0); ax[0].plot(m,label=a,color=COL[a])
ax[0].set_xlabel("fitness evaluations"); ax[0].set_ylabel("best-so-far F1 (mean of 5 seeds)"); ax[0].set_ylim(np.percentile([h.mean(0)[100] for h in hist.values()],0)-0.01,None); ax[0].legend(); ax[0].set_title("Convergence")
for a in g.index: ax[1].errorbar(g.loc[a,"k"],g.loc[a,"svm"],yerr=g.loc[a,"svm_std"],fmt="o",label=a,color=COL[a])
ax[1].set_xlabel("bands selected"); ax[1].set_ylabel("test SVM-RBF OA"); ax[1].set_title("Accuracy vs bands"); ax[1].legend()
for a in g.index: ax[2].scatter(g.loc[a,"time"],g.loc[a,"svm"],label=a,color=COL[a])
ax[2].set_xlabel("search time (s)"); ax[2].set_ylabel("test SVM-RBF OA"); ax[2].set_title("Accuracy vs time")
plt.tight_layout(); plt.savefig(OUT/"v2_algo_plots.png",dpi=130)

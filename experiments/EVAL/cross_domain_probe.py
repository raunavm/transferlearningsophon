"""Cross-domain flavour probe: is the QCD axis a second feature or the same one?

Fits the frozen-probe linear model on each flavour axis separately and on their
union, then applies each fitted direction to the other axis. Same cached
features_v2, same SPLIT_SEED, same C grid as experiments/EVAL/probe.py, so the
in-domain numbers here are a re-derivation of Table 1 from raw features and must
match paper/ml4ps/results.tex to 4-5 decimals -- if they stop matching, this
script is wrong, not the paper.

Input: <arm>.npz per arm with X (N,128) features and L (N,) label188, for the
four rows label 0/1/169/181, staged from the PVC under /data/results/eval.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
ARMS={"L162_s1b":"mtx-l162-s1b","R16_Q1_s2":"mtx-r16q1-s2","R16_Q1_s3":"mtx-r16q1-s3","R16_Q1_s4":"mtx-r16q1-s4"}
TASKS={"A":(0,1),"B":(169,181)}
def splits(n):
    p=np.random.default_rng(20260822).permutation(n); a,b=int(.6*n),int(.8*n); return p[:a],p[a:b],p[b:]
data={a:np.load(f"{d}.npz") for a,d in ARMS.items()}; L=data["L162_s1b"]["L"]
T={}
for t,(s,bk) in TASKS.items():
    idx=np.where((L==s)|(L==bk))[0]; y=(L[idx]==s).astype(int); tr,va,te=splits(idx.size); T[t]=(idx,y,tr,va,te)
lg=lambda a: np.log(1-a)
for arm in ARMS:
    X=data[arm]["X"].astype(float)
    XA,yA,trA,vaA,teA=[X[T["A"][0]]]+list(T["A"][1:]); XB,yB,trB,vaB,teB=[X[T["B"][0]]]+list(T["B"][1:])
    # common scaler on union of train rows, so w's live in one space
    sc=StandardScaler().fit(np.vstack([XA[trA],XB[trB]]))
    sA,sB=sc.transform(XA),sc.transform(XB)
    print(f"\n##### {arm}")
    print(" C     | A->A    A->B   | B->B    B->A   | cos(wA,wB) | pooled->A pooled->B")
    for C in [0.01,0.1,1.0,10.0,100.0]:
        cA=LogisticRegression(C=C,max_iter=3000).fit(sA[trA],yA[trA])
        cB=LogisticRegression(C=C,max_iter=3000).fit(sB[trB],yB[trB])
        # pooled: balance domains by sample weight so B is not swamped 3:1
        Xp=np.vstack([sA[trA],sB[trB]]); yp=np.concatenate([yA[trA],yB[trB]])
        wp=np.concatenate([np.full(trA.size,1.0/trA.size),np.full(trB.size,1.0/trB.size)])*(trA.size+trB.size)/2
        cP=LogisticRegression(C=C,max_iter=3000).fit(Xp,yp,sample_weight=wp)
        aAA=roc_auc_score(yA[teA],cA.decision_function(sA[teA])); aAB=roc_auc_score(yB[teB],cA.decision_function(sB[teB]))
        aBB=roc_auc_score(yB[teB],cB.decision_function(sB[teB])); aBA=roc_auc_score(yA[teA],cB.decision_function(sA[teA]))
        aPA=roc_auc_score(yA[teA],cP.decision_function(sA[teA])); aPB=roc_auc_score(yB[teB],cP.decision_function(sB[teB]))
        w1,w2=cA.coef_[0],cB.coef_[0]; cos=w1@w2/np.linalg.norm(w1)/np.linalg.norm(w2)
        print(f" {C:<5g} | {aAA:.5f} {aAB:.5f} | {aBB:.5f} {aBA:.5f} | {cos:+.3f}     | {aPA:.5f}   {aPB:.5f}")
    # best transferable A-learned direction (C picked on B val) -- upper bound on "A direction explains B"
    best=max(((roc_auc_score(yB[vaB],LogisticRegression(C=C,max_iter=3000).fit(sA[trA],yA[trA]).decision_function(sB[vaB])),C) for C in [0.01,0.1,1.0,10.0,100.0]))
    cA=LogisticRegression(C=best[1],max_iter=3000).fit(sA[trA],yA[trA])
    print(f" best-transfer A-dir (C={best[1]} chosen on B val): B test AUC {roc_auc_score(yB[teB],cA.decision_function(sB[teB])):.5f}  vs in-domain B best {max(roc_auc_score(yB[teB],LogisticRegression(C=C,max_iter=3000).fit(sB[trB],yB[trB]).decision_function(sB[teB])) for C in [0.01,0.1,1.0,10.0,100.0]):.5f}")

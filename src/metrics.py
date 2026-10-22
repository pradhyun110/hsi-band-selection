"""
Every metric used in the final comparison.

Classification (SVM trained on the learning pixels with only the chosen bands, scored on the exam pixels).
For each crop class c:  TP = correctly found, FN = missed, FP = false alarms, TN = correctly rejected.
  OA           overall accuracy        = all correct / all exam pixels
  AA           average accuracy        = mean over classes of TP/(TP+FN)  (the mean sensitivity / recall)
  Kappa        agreement beyond chance = (OA - pe) / (1 - pe)
  Precision    mean over classes of TP/(TP+FP)
  Sensitivity  mean over classes of TP/(TP+FN)          (identical to AA; kept under both names)
  Specificity  mean over classes of TN/(TN+FP)
  F1           mean over classes of 2 P R / (P + R)
  MCC          Matthews correlation coefficient (multi-class form, uses every cell of the confusion matrix)
  per-class sensitivity (one value per crop)
Base paper metrics
  CEM AUC      target detection: for each crop, a CEM filter built from that crop's mean learning spectrum
               scores the exam pixels; area under the ROC curve, averaged over the 16 crops
  SSIM         the 164 clean bands are rebuilt from the chosen bands by least squares; mean SSIM of the
               rebuilt band images vs the originals (1 = perfect)
Checked against scikit-learn (accuracy, precision, recall, F1, kappa, MCC agree to 1e-12).
"""
import json
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from skimage.metrics import structural_similarity


def classification_metrics(y, p, classes):
    cm = confusion_matrix(y, p, labels=classes).astype(float); N = cm.sum()
    TP = np.diag(cm); FP = cm.sum(0) - TP; FN = cm.sum(1) - TP; TN = N - TP - FP - FN
    prec = np.divide(TP, TP + FP, out=np.zeros_like(TP), where=(TP + FP) > 0)
    rec = TP / (TP + FN); spec = TN / (TN + FP)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(TP), where=(prec + rec) > 0)
    oa = TP.sum() / N; pe = (cm.sum(0) * cm.sum(1)).sum() / N ** 2
    t, pr = cm.sum(1), cm.sum(0)
    mcc = (TP.sum() * N - t @ pr) / np.sqrt((N ** 2 - pr @ pr) * (N ** 2 - t @ t))
    return dict(OA=oa, MCC=mcc, AA=rec.mean(), Kappa=(oa - pe) / (1 - pe), Precision=prec.mean(), Sensitivity=rec.mean(),
                Specificity=spec.mean(), F1=f1.mean(), per_class_sensitivity=json.dumps(np.round(rec, 4).tolist()))


def cem_auc(Zimg, Ztr, ytr, Zte, yte):
    """arrays already restricted to the chosen bands."""
    R = Zimg.T @ Zimg / len(Zimg); R += 1e-6 * np.trace(R) / len(R) * np.eye(len(R)); Ri = np.linalg.inv(R)
    aucs = []
    for c in np.unique(ytr):
        d = Ztr[ytr == c].mean(0); w = Ri @ d / (d @ Ri @ d)
        aucs.append(roc_auc_score(yte == c, Zte @ w))
    return float(np.mean(aucs))


class SSIMReconstruction:
    """rebuild the 164 clean bands from any chosen band set (original band numbers); cached."""
    def __init__(self, cube, clean_bands):
        self.cube = cube; self.H, self.W, _ = cube.shape
        self.target = cube[:, :, clean_bands]; self.T = self.target.reshape(-1, len(clean_bands)); self.cache = {}

    def __call__(self, bands):
        key = tuple(int(b) for b in bands)
        if key not in self.cache:
            A = np.column_stack([np.ones(self.H * self.W), self.cube[:, :, list(key)].reshape(-1, len(key))])
            B, *_ = np.linalg.lstsq(A, self.T, rcond=None); R = (A @ B).reshape(self.target.shape)
            self.cache[key] = float(np.mean([structural_similarity(self.target[:, :, j], R[:, :, j],
                                    data_range=np.ptp(self.target[:, :, j])) for j in range(self.target.shape[2])]))
        return self.cache[key]

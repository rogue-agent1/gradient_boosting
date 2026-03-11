#!/usr/bin/env python3
"""Gradient boosting regressor with decision stumps — zero deps."""
import math

class Stump:
    def __init__(self): self.feature=0; self.threshold=0; self.left_val=0; self.right_val=0
    def predict_one(self, x): return self.left_val if x[self.feature]<=self.threshold else self.right_val
    def predict(self, X): return [self.predict_one(x) for x in X]

class GradientBoosting:
    def __init__(self, n_estimators=50, lr=0.1):
        self.n_est=n_estimators; self.lr=lr; self.trees=[]; self.init_val=0
    def fit(self, X, y):
        self.init_val = sum(y)/len(y)
        preds = [self.init_val]*len(y)
        for _ in range(self.n_est):
            residuals = [y[i]-preds[i] for i in range(len(y))]
            stump = self._fit_stump(X, residuals)
            self.trees.append(stump)
            sp = stump.predict(X)
            preds = [preds[i]+self.lr*sp[i] for i in range(len(y))]
    def _fit_stump(self, X, y):
        best, best_mse = Stump(), float('inf')
        for f in range(len(X[0])):
            vals = sorted(set(x[f] for x in X))
            for t in vals:
                left = [y[i] for i in range(len(X)) if X[i][f]<=t]
                right = [y[i] for i in range(len(X)) if X[i][f]>t]
                if not left or not right: continue
                ml, mr = sum(left)/len(left), sum(right)/len(right)
                mse = sum((v-ml)**2 for v in left)+sum((v-mr)**2 for v in right)
                if mse < best_mse:
                    best_mse=mse; best.feature=f; best.threshold=t; best.left_val=ml; best.right_val=mr
        return best
    def predict(self, X):
        preds = [self.init_val]*len(X)
        for t in self.trees:
            sp = t.predict(X)
            preds = [preds[i]+self.lr*sp[i] for i in range(len(X))]
        return preds

def test():
    import random; random.seed(42)
    X = [[x/10] for x in range(100)]
    y = [math.sin(x[0]) + random.gauss(0, 0.1) for x in X]
    gb = GradientBoosting(n_estimators=30, lr=0.3); gb.fit(X, y)
    preds = gb.predict(X)
    mse = sum((p-t)**2 for p,t in zip(preds,y))/len(y)
    assert mse < 0.1, f"MSE too high: {mse}"
    print(f"Gradient boosting MSE: {mse:.4f}")
    print("All tests passed!")

if __name__ == "__main__": test()

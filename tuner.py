import numpy as np
from scipy.optimize import minimize
import pypandoc


def build_A_from_d(d):
    A = np.empty(12, dtype=float)
    A[0] = 0.0
    A[1:] = np.cumsum(d)
    return A

R = np.array([1,16/15,9/8,6/5,5/4,4/3,45/32,3/2,8/5,5/3,9/5,15/8,2])
target_logs = np.log2(R)
w = 0.0001
nD=[1,w,1,w,1,1,w,1,w,1,w,1,1]
def score_for_A(A, Bs):
    total = 0.0
    for Bi in Bs:
        b = A[Bi]
        e = []
        for j in range(13):
            p = Bi+j
            d = A[p%12] + p//12 - b - target_logs[j]
            e.append(d*d*nD[j])
        total += np.mean(e)
    return total / len(Bs)

def objective(d, Bs):
    A = build_A_from_d(d)
    return score_for_A(A, Bs)

def anneal_optimize(Bs, trials=1000, temp0=0.1, decay=0.9995):
    rng = np.random.default_rng(0)
    d = np.full(11, 1.0 / 12.0)
    best_d = d.copy()
    best_score = objective(d, Bs)
    temp = temp0
    for t in range(trials):
        cand = d + rng.normal(0, 0.002, size=11)
        if np.any(cand <= 0): continue
        if np.sum(cand) >= 1.0: continue
        sc = objective(cand, Bs)
        if sc < best_score or rng.random() < np.exp((best_score - sc)/temp):
            d, best_score = cand, sc
            best_d = d.copy()
        temp *= decay
    return best_d, best_score

def optimize_A(Bs, verbose=False):
    best_score = 1e9
    best_d = None
    for _ in range(10):
        d0, s0 = anneal_optimize(Bs, trials=1500)
        if s0 < best_score:
            best_score = s0
            best_d = d0.copy()
    bounds = [(1e-6, 1.0)] * 11
    cons = ({'type': 'ineq', 'fun': lambda d: 1.0 - 1e-6 - np.sum(d)},)
    res = minimize(lambda d: objective(d, Bs),
                   x0=best_d, bounds=bounds, constraints=cons,
                   method='SLSQP',
                   options={'ftol':1e-12, 'maxiter':1000, 'disp': verbose})
    d_opt = res.x
    A_opt = build_A_from_d(d_opt)
    score = score_for_A(A_opt, Bs)
    return A_opt, score, res

def save_scl(filename, A_opt, description="Custom Just Tuning"):
    # log2(比) → cent値に変換
    cents = (A_opt - A_opt[0]) * 1200.0
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"! {filename}\n!\n{description}\n")
        f.write("12\n")  # 音の数（12音）
        for c in cents[1:]:
            f.write(f"{c:.6f}\n")
        f.write("2/1\n")  # 最後だけ比率表記でOK
    print(f"SCLファイルを保存しました → {filename}")

if __name__ == "__main__":
    Bs = list(map(int, input("基音の半音階番号をスペース区切りで入力してください: ").split()))
    A_opt, score, res = optimize_A(Bs, verbose=False)
    print("最適 A:")
    for i, a in enumerate(A_opt):
        print(f"A{i+1}: {a:.9f}")
    print("Copy:")
    print(*A_opt)
    print("スコア =", score)

    # Vital用チューニングファイル出力
    save_scl("vital_tuning.scl", A_opt)

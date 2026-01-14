import numpy as np
import sounddevice as sd
import mido
import threading
import time
from scipy.optimize import minimize
print("\n"*50)
# ====================================================
# === 音律最適化部分 =================================
# ====================================================

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

# ====================================================
# === 音源シンセサイザー部分 ==========================
# ====================================================

BASE_FREQ = 261.63  # C4
SAMPLE_RATE = 44100

active_notes = {}
sustain_on = False
sustain_notes = set()
lock = threading.Lock()

ATTACK = 0.02
DECAY = 0.1
SUSTAIN = 0.7
RELEASE = 0.3

# ==== 音生成 ====
def audio_callback(outdata, frames, time_info, status):
    global active_notes
    t = np.arange(frames) / SAMPLE_RATE
    signal = np.zeros(frames, dtype=np.float32)

    with lock:
        to_delete = []
        for note, d in active_notes.items():
            dt = frames / SAMPLE_RATE
            d["time"] += dt
            f = d["freq"]

            # 倍音追加
            wave = (
                np.sin(2 * np.pi * f * (t + d["phase"])) * 0.6 +
                np.sin(2 * np.pi * f * 2 * (t + d["phase"])) * 0.3 +
                np.sin(2 * np.pi * f * 3 * (t + d["phase"])) * 0.1
            )

            # === エンベロープ ===
            if not d["release"]:
                if d["time"] < ATTACK:
                    env = d["time"] / ATTACK
                elif d["time"] < ATTACK + DECAY:
                    env = 1 - (1 - SUSTAIN) * ((d["time"] - ATTACK) / DECAY)
                else:
                    env = SUSTAIN
            else:
                d["release_time"] += dt
                env = d["release_start"] * (1 - d["release_time"] / RELEASE)
                if env <= 0:
                    to_delete.append(note)
                    continue

            d["env"] = env
            d["phase"] += dt
            signal += wave * env * d["amp"]

        for note in to_delete:
            active_notes.pop(note, None)

    outdata[:] = np.clip(signal, -1, 1).reshape(-1, 1)

# ==== MIDI受信 ====
def midi_listener(portname, tuning_log2):
    global sustain_on
    with mido.open_input(portname) as inport:
        print(f"MIDI入力ポート {portname} を監視中...")
        for msg in inport:
            if hasattr(msg, "channel") and msg.channel != 0:
                continue

            if msg.type == "note_on" and msg.velocity > 0:
                note = msg.note
                base = note % 12
                octave = note // 12 - 5
                freq = BASE_FREQ * (2 ** tuning_log2[base]) * (2 ** octave)
                amp = msg.velocity / 127.0 * 0.4

                with lock:
                    active_notes[note] = {
                        "phase": 0.0,
                        "freq": freq,
                        "amp": amp,
                        "time": 0.0,
                        "env": 0.0,
                        "release": False,
                        "release_time": 0.0,
                        "release_start": 0.0,
                    }
                print(f"note_on {note:3d} -> {freq:.2f} Hz")

            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                note = msg.note
                with lock:
                    if sustain_on:
                        sustain_notes.add(note)
                    elif note in active_notes:
                        d = active_notes[note]
                        d["release"] = True
                        d["release_time"] = 0.0
                        d["release_start"] = d["env"]
                print(f"note_off {note:3d}")

            elif msg.type == "control_change" and msg.control == 64:
                if hasattr(msg, "channel") and msg.channel != 0:
                    continue
                if msg.value >= 64:
                    sustain_on = True
                    print("Sustain pedal ON")
                else:
                    sustain_on = False
                    print("Sustain pedal OFF")
                    with lock:
                        for n in list(sustain_notes):
                            if n in active_notes:
                                d = active_notes[n]
                                d["release"] = True
                                d["release_time"] = 0.0
                                d["release_start"] = d["env"]
                        sustain_notes.clear()

# ====================================================
# === メイン ==========================================
# ====================================================

if __name__ == "__main__":
    print("基音の半音階番号をスペース区切りで入力してください（例: 0 7 9）:")
    Bs = list(map(int, input().split()))

    print("\n=== 音律最適化中... ===")
    A_opt, score, res = optimize_A(Bs, verbose=False)
    print("最適化完了！")
    print("tuning_log2 =", np.round(A_opt, 9))
    print("score =", score)

    tuning_log2 = A_opt

    ports = mido.get_input_names()
    if not ports:
        print("MIDI入力デバイスが見つかりません。")
        exit()

    print("\n使用可能なMIDI入力ポート:")
    for i, p in enumerate(ports):
        print(f"  {i}: {p}")
    idx = int(input("使用するポート番号を選んでください: "))

    stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=audio_callback,
        blocksize=256,
        latency="low",
    )
    stream.start()

    threading.Thread(target=midi_listener, args=(ports[idx], tuning_log2), daemon=True).start()

    print("\n演奏中です。Ctrl+Cで終了。")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        stream.stop()
        stream.close()
        sd.stop()
        print("\n終了しました。")

#!/usr/bin/env python3
"""
MIDI の Key Signature (調号) を最優先して区間を割り出し、
各区間で使用されている音を解析して音律を最適化するスクリプト。

ロジック:
1. MIDIメタデータの KeySignature を取得 (例: 0s: CMaj, 45s: GMaj...)
2. 調号が切り替わるタイミングごとに区間を分割
3. 各区間内にあるノートを収集し、「そのキーの時に使われた音」として記録
4. これを基に焼きなまし法で全曲に最適な12音階(A)を算出
"""

import argparse
import numpy as np
from scipy.optimize import minimize
import pretty_midi
import mido
import sounddevice as sd
import soundfile as sf
from music21 import converter

# ---------- 最適化ロジック (変更なし) ----------

R_ratios = np.array([1, 16/15, 9/8, 6/5, 5/4, 4/3, 45/32, 3/2, 8/5, 5/3, 9/5, 15/8, 2])
target_logs = np.log2(R_ratios)
w_default = 0.0001
nD_template = [1, w_default, 1, w_default, 1, 1, w_default, 1, w_default, 1, w_default, 1, 1]

def Afromd(d):
    A = np.empty(12)
    A[0] = 0.0
    A[1:] = np.cumsum(d)
    return A

def score(A, Bs, tonic_usage_map):
    total = 0.0
    valid_keys = 0
    
    for Bi in Bs:
        # このキーが使用された区間がない（マップにない）場合はスキップ
        if Bi not in tonic_usage_map:
            continue
            
        used_in_this_key = tonic_usage_map[Bi]
        valid_keys += 1
        b = A[Bi]
        
        e = []
        for j in range(13):
            target_pc = (Bi + j) % 12
            
            p = Bi + j
            d_val = A[p % 12] + p // 12 - b - target_logs[j]
            
            weight = nD_template[j]
            # 「このキーの区間」で「実際に鳴った音」なら重み1.0
            if (j < 12) and (target_pc in used_in_this_key):
                weight = 1.0
            elif j == 12:
                weight = 1.0
            
            e.append(d_val * d_val * weight)
        
        total += np.mean(e)
        
    if valid_keys == 0: return 100.0 # fallback bad score
    return total / valid_keys

def obj(d, Bs, tonic_usage_map):
    return score(Afromd(d), Bs, tonic_usage_map)

def anneal_once(Bs, tonic_usage_map, n_iter=2000, t0=0.2, rng_seed=0):
    r = np.random.default_rng(rng_seed)
    d = np.full(11, 1.0/12.0)
    best_d = d.copy()
    best_s = obj(d, Bs, tonic_usage_map)
    t = t0
    
    for i in range(n_iter):
        c = d + r.normal(0, 0.0025, size=11)
        if np.any(c <= 0) or np.sum(c) >= 1.0:
            t *= 0.9995
            continue
        sc = obj(c, Bs, tonic_usage_map)
        if sc < best_s or r.random() < np.exp((best_s - sc) / t):
            d = c
            best_s = sc
            best_d = d.copy()
        t *= 0.9995
    return best_d, best_s

def optimize_once(Bs, tonic_usage_map, anneal_iters=2000, verbose=False):
    bd, bs = anneal_once(Bs, tonic_usage_map, n_iter=anneal_iters)
    bounds = [(1e-6, 1.0)] * 11
    cons = ({'type': 'ineq', 'fun': lambda d: 1.0 - 1e-6 - np.sum(d)},)
    func = lambda d: obj(d, Bs, tonic_usage_map)
    res = minimize(func, bd, bounds=bounds, constraints=cons,
                   method='SLSQP', options={'ftol': 1e-12, 'maxiter': 1000, 'disp': False})
    A = Afromd(res.x)
    final_score = score(A, Bs, tonic_usage_map)
    if verbose:
        print(f"Anneal score: {bs:.6f} -> SLSQP success: {res.success}, fun: {res.fun:.6f}")
    return A, final_score

# ---------- 解析ロジック (Key Signature 優先) ----------

note_name_to_pc = {'C':0,'C#':1,'Db':1,'D':2,'D#':3,'Eb':3,'E':4,'F':5,'F#':6,'Gb':6,
                   'G':7,'G#':8,'Ab':8,'A':9,'A#':10,'Bb':10,'B':11}

def get_usage_by_key_signature(midi_path):
    """
    PrettyMIDIの key_signature_changes を使用して、
    調号ごとの区間を特定し、その区間内のノートを収集する。
    
    Returns:
      Bs: [Tonic1, Tonic2, ...]
      tonic_usage_map: { Tonic: set(UsedPCs) }
    """
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"MIDI読み込みエラー: {e}")
        return [], {}

    # キーシグネチャの取得
    # pretty_midi.KeySignature(key_number, time)
    # key_number: 0-11 Major, 12-23 Minor (ex. 0=C, 12=Cm)
    ks_changes = pm.key_signature_changes
    
    # キーシグネチャがない場合は空を返す -> フォールバックへ
    if not ks_changes:
        return [], {}

    print(f"MIDI Key Signature を検出しました: {len(ks_changes)} 箇所")

    # 時間順にソート（念のため）
    ks_changes.sort(key=lambda x: x.time)
    
    # 終了時間を取得
    end_time = pm.get_end_time()
    
    # 区間リスト作成: [(start, end, tonic_pc), ...]
    segments = []
    for i, ks in enumerate(ks_changes):
        start = ks.time
        # 次のキーチェンジがあればそこまで、なければ曲の終わりまで
        end = ks_changes[i+1].time if (i + 1 < len(ks_changes)) else end_time
        
        # Tonic PC 計算 (0-11: Maj, 12-23: Min -> どちらも % 12 で基音PCになる)
        tonic = ks.key_number % 12
        segments.append((start, end, tonic))

    # マップ初期化
    # 同じキーに転調して戻ってくる場合もあるので、setに追加していく
    tonic_usage_map = {}
    found_tonics = set()
    
    for _, _, t in segments:
        found_tonics.add(t)
        if t not in tonic_usage_map:
            tonic_usage_map[t] = set()

    # 全ノートを走査して、該当区間のキーに紐付ける
    # (楽器数 * ノート数 * 区間数 になるが、MIDIなら通常高速)
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            # ノートの開始位置がどの区間にあるか探す
            # 区間は時間順なので、線形探索で十分
            # (二分探索も可能だが、転調回数は高々知れているためこれでOK)
            for start, end, tonic in segments:
                # ノートの開始点が区間内なら採用
                if start <= note.start < end:
                    tonic_usage_map[tonic].add(note.pitch % 12)
                    break # 1つのノートは1つの区間に所属
    
    return sorted(list(found_tonics)), tonic_usage_map

def analyze_keys_fallback_music21(midi_path, measure_window=4):
    """
    調号情報がない場合のフォールバック（music21で小節解析）
    """
    print("調号情報が見つからないため、music21 による内容解析を行います...")
    s = converter.parse(midi_path)
    parts = s.parts if s.parts else [s]
    
    usage_map = {}
    found_tonics = set()

    for p in parts:
        measures = p.makeMeasures()
        total = len(measures.getElementsByClass('Measure'))
        i = 1
        while i <= total:
            j = min(i+measure_window-1, total)
            seg = measures.measures(i, j)
            
            # キー推定
            current_tonic = None
            try:
                k = seg.analyze('key')
                if hasattr(k, 'tonic'):
                    tname = k.tonic.name
                    if tname in note_name_to_pc:
                        current_tonic = note_name_to_pc[tname]
                    else:
                        current_tonic = note_name_to_pc.get(tname[0], 0)
            except:
                pass
            
            if current_tonic is not None:
                found_tonics.add(current_tonic)
                if current_tonic not in usage_map:
                    usage_map[current_tonic] = set()
                
                # ノート収集
                flat_seg = seg.flatten()
                for el in flat_seg.notes:
                    if el.isNote:
                        usage_map[current_tonic].add(el.pitch.pitchClass)
                    elif el.isChord:
                        for n in el.pitches:
                            usage_map[current_tonic].add(n.pitchClass)
            i += measure_window

    if not found_tonics:
        # 完全失敗時: C基準で全音登録
        t = 0
        used = set()
        for el in s.flatten().notes:
             if el.isNote: used.add(el.pitch.pitchClass)
             elif el.isChord: 
                 for n in el.pitches: used.add(n.pitchClass)
        return [0], {0: used}

    return sorted(list(found_tonics)), usage_map

# ---------- 合成 ----------

def midi_note_to_freq_with_A(note, tonic_midi, A):
    rel = note - tonic_midi
    octave = np.floor_divide(rel, 12)
    deg = int(rel % 12)
    tonic_freq = 440.0 * 2 ** ((tonic_midi - 69) / 12.0)
    freq = tonic_freq * (2.0 ** (A[deg] + octave))
    return float(freq)

def synthesize_midi_to_audio(midi_path, A, tonic_mode='global', out_wav=None, sr=44100):
    pm = pretty_midi.PrettyMIDI(midi_path)
    
    note_counts = np.zeros(12, dtype=int)
    for inst in pm.instruments:
        if inst.is_drum: continue
        for n in inst.notes:
            note_counts[n.pitch % 12] += 1
    most_freq_pc = int(np.argmax(note_counts))
    tonic_midi_center = 60 + most_freq_pc

    duration = pm.get_end_time()
    t = np.linspace(0, duration, int(sr*duration)+1)
    audio = np.zeros_like(t)

    def adsr_env(length, sr, attack=0.01, decay=0.02, sustain=0.8, release=0.03):
        N = int(length*sr)
        if N <= 0: return np.array([])
        aN = max(1, int(attack*sr))
        dN = max(1, int(decay*sr))
        rN = max(1, int(release*sr))
        sN = max(0, N - (aN+dN+rN))
        env = np.concatenate([
            np.linspace(0,1,aN,endpoint=False),
            np.linspace(1,sustain,dN,endpoint=False),
            np.ones(sN)*sustain,
            np.linspace(sustain,0,rN)
        ])
        if len(env) < N:
            env = np.pad(env, (0, N-len(env)))
        elif len(env) > N:
            env = env[:N]
        return env

    for inst in pm.instruments:
        if inst.is_drum: continue
        for n in inst.notes:
            start_i = int(n.start * sr)
            end_i = int(n.end * sr)
            length = n.end - n.start
            if end_i <= start_i: continue
            
            if tonic_mode == 'center':
                tonic_midi = tonic_midi_center
            else:
                tonic_midi = 60
                
            freq = midi_note_to_freq_with_A(n.pitch, tonic_midi, A)
            
            cur_len = end_i - start_i
            tt = np.linspace(0, length, cur_len, endpoint=False)
            wave = np.sin(2*np.pi*freq*tt) + 0.5*np.sin(2*np.pi*2*freq*tt)
            env = adsr_env(length, sr)
            
            minlen = min(env.size, wave.size)
            wave = wave[:minlen]
            env = env[:minlen]
            amp = (n.velocity / 127.0) * 0.2
            
            target_slice = slice(start_i, start_i+minlen)
            if audio[target_slice].shape[0] == wave.shape[0]:
                audio[target_slice] += amp * wave * env

    maxv = np.max(np.abs(audio)) + 1e-12
    audio = audio / maxv * 0.95

    if out_wav:
        sf.write(out_wav, audio, sr)
        print(f"WAV saved to {out_wav}")
    try:
        print("Playing audio...")
        sd.play(audio, sr)
        sd.wait()
    except Exception as e:
        print("再生エラー:", e)

# ---------- メイン ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('midi', help='input midi file path')
    parser.add_argument('--out', default='out_retuned.wav', help='output WAV path')
    parser.add_argument('--tonic-mode', choices=['global','center'], default='global')
    parser.add_argument('--anneal-iters', type=int, default=2000)
    args = parser.parse_args()

    midi_path = args.midi
    
    # 1) 調号に基づく区間解析
    Bs, tonic_usage_map = get_usage_by_key_signature(midi_path)
    
    # 調号が見つからない場合はフォールバック
    if not Bs:
        Bs, tonic_usage_map = analyze_keys_fallback_music21(midi_path)

    print("検出された基音(Bs):", Bs)
    print("各キー区間での使用音(PC):")
    for b in Bs:
        pcs = tonic_usage_map.get(b, [])
        print(f"  Key {b}: {sorted(list(pcs))}")

    # 2) 最適化
    print("焼きなまし -> 最適化実行...")
    A, s = optimize_once(Bs, tonic_usage_map, anneal_iters=args.anneal_iters, verbose=True)
    
    print("得られた A (log2 単位):")
    for i,a in enumerate(A):
        print(f"{i:2d}: {a:.9f}")
    print("スコア:", s)

    # 3) 合成
    synthesize_midi_to_audio(midi_path, A, tonic_mode=args.tonic_mode, out_wav=args.out)

if __name__ == "__main__":
    main()

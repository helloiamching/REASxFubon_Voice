import whisper
import torch
import os
import librosa
import numpy as np
from pathlib import Path
from collections import Counter

# ========== 設定路徑 ==========
device = torch.device("cpu")
input_folder = "voice_file"
text_output_folder = "voice_text"
emo_output_folder = "voice_emo"

os.makedirs(text_output_folder, exist_ok=True)
os.makedirs(emo_output_folder, exist_ok=True)

# ========== 載入 Whisper 模型 ==========
print("Loading Whisper model...")
model = whisper.load_model("medium", device=device)

# ========== 情緒關鍵字字典 ==========
emotion_keywords = {
    "生氣/不滿": ["生氣", "氣死", "不爽", "太誇張", "怎麼會", "什麼鬼", "受不了", "投訴", "抱怨"],
    "焦慮/擔心": ["擔心", "害怕", "緊張", "怎麼辦", "來不及", "急", "趕快", "快一點"],
    "滿意/開心": ["謝謝", "太好了", "不錯", "很棒", "滿意", "開心", "感謝"],
    "疑惑/困擾": ["不懂", "看不懂", "怎麼", "為何", "搞不清楚", "不知道", "不確定"],
}

def extract_audio_features(audio_path, start_time, end_time):
    """提取音頻片段的聲學特徵"""
    try:
        # 載入音頻
        y, sr = librosa.load(audio_path, sr=16000, offset=start_time, duration=end_time-start_time)
        
        if len(y) == 0:
            return None
        
        # 1. 音高（Pitch）分析
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        avg_pitch = np.mean(pitch_values) if pitch_values else 0
        pitch_variance = np.var(pitch_values) if pitch_values else 0
        
        # 2. 音量（Energy/Intensity）
        rms = librosa.feature.rms(y=y)[0]
        avg_energy = np.mean(rms)
        energy_variance = np.var(rms)
        
        # 3. 語速（Speech Rate）- 透過過零率估算
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        avg_zcr = np.mean(zcr)
        
        # 4. 音調變化率
        pitch_change_rate = pitch_variance / (avg_pitch + 1e-6)
        
        return {
            'avg_pitch': avg_pitch,
            'pitch_variance': pitch_variance,
            'pitch_change_rate': pitch_change_rate,
            'avg_energy': avg_energy,
            'energy_variance': energy_variance,
            'avg_zcr': avg_zcr,
            'duration': end_time - start_time
        }
    except Exception as e:
        print(f"  警告: 音頻特徵提取失敗 - {e}")
        return None

def analyze_emotion_multimodal(text, audio_features, speaker_baseline=None):
    """結合文本和音頻特徵的多模態情緒分析"""
    emotion_scores = {
        "生氣/不滿": 0,
        "焦慮/擔心": 0,
        "滿意/開心": 0,
        "疑惑/困擾": 0,
        "平靜/中性": 0
    }
    
    # ========== 1. 文本關鍵字分析 ==========
    for emotion, keywords in emotion_keywords.items():
        for keyword in keywords:
            if keyword in text:
                emotion_scores[emotion] += 2  # 文本權重較高
    
    # ========== 2. 音頻特徵分析 ==========
    if audio_features and speaker_baseline:
        # 與說話人基線比較
        pitch_diff = (audio_features['avg_pitch'] - speaker_baseline['avg_pitch']) / (speaker_baseline['avg_pitch'] + 1e-6)
        energy_diff = (audio_features['avg_energy'] - speaker_baseline['avg_energy']) / (speaker_baseline['avg_energy'] + 1e-6)
        
        # 音調升高 + 音量增大 → 生氣/焦慮
        if pitch_diff > 0.15 and energy_diff > 0.2:
            emotion_scores["生氣/不滿"] += 1.5
            emotion_scores["焦慮/擔心"] += 1.0
        
        # 音調波動大 → 情緒激動
        if audio_features['pitch_change_rate'] > speaker_baseline['pitch_change_rate'] * 1.5:
            emotion_scores["生氣/不滿"] += 1.0
            emotion_scores["焦慮/擔心"] += 0.8
        
        # 語速快（過零率高）→ 焦慮
        if audio_features['avg_zcr'] > speaker_baseline['avg_zcr'] * 1.3:
            emotion_scores["焦慮/擔心"] += 1.2
        
        # 音調降低 + 音量小 → 滿意/平靜
        if pitch_diff < -0.1 and energy_diff < -0.1:
            emotion_scores["滿意/開心"] += 0.5
            emotion_scores["平靜/中性"] += 1.0
        
        # 音量變化大 → 情緒起伏
        if audio_features['energy_variance'] > speaker_baseline['energy_variance'] * 1.5:
            emotion_scores["生氣/不滿"] += 0.8
    
    # ========== 3. 返回得分最高的情緒 ==========
    if max(emotion_scores.values()) > 0:
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = emotion_scores[dominant_emotion] / sum(emotion_scores.values())
        return dominant_emotion, confidence, emotion_scores
    else:
        return "平靜/中性", 0.5, emotion_scores

def calculate_speaker_baseline(audio_path, dialogue, speaker):
    """計算說話人的基線音頻特徵"""
    speaker_segments = [(start, end, text) for start, spk, end, text in dialogue if spk == speaker]
    
    if not speaker_segments:
        return None
    
    all_features = []
    for start, end, text in speaker_segments[:5]:  # 取前5段計算基線
        features = extract_audio_features(audio_path, start, end)
        if features:
            all_features.append(features)
    
    if not all_features:
        return None
    
    # 計算平均值作為基線
    baseline = {
        'avg_pitch': np.mean([f['avg_pitch'] for f in all_features]),
        'pitch_variance': np.mean([f['pitch_variance'] for f in all_features]),
        'avg_energy': np.mean([f['avg_energy'] for f in all_features]),
        'energy_variance': np.mean([f['energy_variance'] for f in all_features]),
        'avg_zcr': np.mean([f['avg_zcr'] for f in all_features]),
        'pitch_change_rate': np.mean([f['pitch_change_rate'] for f in all_features])
    }
    
    return baseline

# ========== 處理所有音檔 ==========
audio_files = list(Path(input_folder).glob("*.wav")) + list(Path(input_folder).glob("*.mp3"))

if not audio_files:
    print(f"在 {input_folder} 資料夾中找不到音檔")
else:
    print(f"找到 {len(audio_files)} 個音檔\n")

for audio_file in audio_files:
    print(f"正在處理: {audio_file.name}")
    
    # ========== Whisper 辨識 ==========
    print("  - 語音辨識中...")
    result = model.transcribe(
        str(audio_file),
        language="zh",
        initial_prompt="客服對話逐字稿，請正確辨識數字。",
    )
    
    segments = result["segments"]
    
    # ========== 說話人分離 ==========
    dialogue = []
    current_speaker = "客服"
    客服關鍵字 = ["您好", "請問", "幫您", "為您", "感謝", "歡迎", "服務", "這邊"]
    客戶關鍵字 = ["我想", "我要", "可以嗎", "謝謝", "好的", "我的"]
    
    for i, seg in enumerate(segments):
        text = seg["text"].strip()
        start = seg["start"]
        end = seg["end"]
        
        if any(kw in text for kw in 客服關鍵字):
            speaker = "客服"
        elif any(kw in text for kw in 客戶關鍵字):
            speaker = "客戶"
        else:
            if i > 0 and start - segments[i-1]["end"] > 2:
                speaker = "客戶" if current_speaker == "客服" else "客服"
            else:
                speaker = current_speaker
        
        dialogue.append((start, speaker, end, text))
        current_speaker = speaker
    
    # ========== 計算說話人基線 ==========
    print("  - 分析音頻特徵中...")
    customer_baseline = calculate_speaker_baseline(str(audio_file), dialogue, "客戶")
    agent_baseline = calculate_speaker_baseline(str(audio_file), dialogue, "客服")
    
    # ========== 儲存逐字稿 ==========
    base_name = audio_file.stem
    text_file = os.path.join(text_output_folder, f"{base_name}.txt")
    
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(f"音檔: {audio_file.name}\n")
        f.write("="*50 + "\n\n")
        for start, speaker, end, text in dialogue:
            m, s = divmod(int(start), 60)
            f.write(f"[{m:02d}:{s:02d}] {speaker}: {text}\n")
    
    print(f"  ✓ 逐字稿已存至: {text_file}")
    
    # ========== 多模態情緒分析 ==========
    print("  - 情緒分析中...")
    emo_file = os.path.join(emo_output_folder, f"{base_name}_emotion.txt")
    
    with open(emo_file, "w", encoding="utf-8") as f:
        f.write(f"音檔: {audio_file.name}\n")
        f.write("="*50 + "\n\n")
        
        customer_emotions = []
        agent_emotions = []
        
        f.write("【逐句情緒分析】（結合文本+音調+音量+語速）\n")
        f.write("-"*50 + "\n")
        
        for start, speaker, end, text in dialogue:
            # 提取音頻特徵
            audio_features = extract_audio_features(str(audio_file), start, end)
            
            # 選擇對應的基線
            baseline = customer_baseline if speaker == "客戶" else agent_baseline
            
            # 多模態情緒分析
            emotion, confidence, scores = analyze_emotion_multimodal(text, audio_features, baseline)
            
            m, s = divmod(int(start), 60)
            f.write(f"[{m:02d}:{s:02d}] {speaker}: {text}\n")
            f.write(f"  情緒: {emotion} (信心度: {confidence:.2%})\n")
            
            if audio_features:
                f.write(f"  音頻特徵: 音調={audio_features['avg_pitch']:.1f}Hz, ")
                f.write(f"音量={audio_features['avg_energy']:.3f}, ")
                f.write(f"波動={audio_features['pitch_variance']:.1f}\n")
            f.write("\n")
            
            if speaker == "客戶":
                customer_emotions.append(emotion)
            else:
                agent_emotions.append(emotion)
        
        # ========== 整體情緒統計 ==========
        f.write("\n" + "="*50 + "\n")
        f.write("【整體情緒統計】\n")
        f.write("-"*50 + "\n")
        
        if customer_emotions:
            customer_emotion_counter = Counter(customer_emotions)
            f.write(f"\n客戶情緒分佈:\n")
            for emo, count in customer_emotion_counter.most_common():
                percentage = (count / len(customer_emotions)) * 100
                f.write(f"  {emo}: {count}次 ({percentage:.1f}%)\n")
            f.write(f"\n客戶主要情緒: {customer_emotion_counter.most_common(1)[0][0]}\n")
        
        if agent_emotions:
            agent_emotion_counter = Counter(agent_emotions)
            f.write(f"\n客服情緒分佈:\n")
            for emo, count in agent_emotion_counter.most_common():
                percentage = (count / len(agent_emotions)) * 100
                f.write(f"  {emo}: {count}次 ({percentage:.1f}%)\n")
            f.write(f"\n客服主要情緒: {agent_emotion_counter.most_common(1)[0][0]}\n")
        
        # ========== 對話品質評估 ==========
        f.write(f"\n" + "="*50 + "\n")
        f.write("【對話品質評估】\n")
        f.write("-"*50 + "\n")
        
        if customer_emotions:
            negative_count = sum(1 for e in customer_emotions if e in ["生氣/不滿", "焦慮/擔心"])
            positive_count = sum(1 for e in customer_emotions if e == "滿意/開心")
            negative_ratio = negative_count / len(customer_emotions)
            positive_ratio = positive_count / len(customer_emotions)
            
            if negative_ratio > 0.5:
                f.write(f"⚠ 警示: 客戶負面情緒佔 {negative_ratio:.1%}，建議重點關注\n")
            elif positive_ratio > 0.3:
                f.write(f"✓ 良好: 客戶正向情緒佔 {positive_ratio:.1%}\n")
            else:
                f.write(f"→ 中性: 客戶情緒整體平穩\n")
            
            # 情緒變化趨勢
            if len(customer_emotions) >= 3:
                early_negative = sum(1 for e in customer_emotions[:len(customer_emotions)//2] if e in ["生氣/不滿", "焦慮/擔心"])
                late_negative = sum(1 for e in customer_emotions[len(customer_emotions)//2:] if e in ["生氣/不滿", "焦慮/擔心"])
                
                if late_negative < early_negative:
                    f.write("↗ 趨勢: 客戶情緒逐漸好轉\n")
                elif late_negative > early_negative:
                    f.write("↘ 趨勢: 客戶情緒逐漸惡化\n")
    
    print(f"  ✓ 情緒分析已存至: {emo_file}")
    print()

print("="*50)
print(f"所有處理完成！")
print(f"逐字稿儲存於: {text_output_folder}/")
print(f"情緒分析儲存於: {emo_output_folder}/")
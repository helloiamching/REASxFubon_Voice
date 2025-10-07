import os
import librosa
import numpy as np
from pathlib import Path
from collections import Counter
from openai import OpenAI
from time import time
from datetime import datetime
import json
from concurrent.futures import ThreadPoolExecutor

# ========== 設定路徑 ==========
input_folder = "voice_file"

CLASS_FOLDERS = {
    "疾病險": "class_disease",
    "旅行平安險": "class_travel", 
    "輔助器具險": "class_tool",
    "其他": "class_other"
}

# 創建所有必要的文件夾
for folder in CLASS_FOLDERS.values():
    os.makedirs(os.path.join(folder, "voice_text"), exist_ok=True)
    os.makedirs(os.path.join(folder, "voice_emo"), exist_ok=True)

def get_api_key():
    import os
    from dotenv import load_dotenv
    load_dotenv()
    return os.getenv('OPENAI_API_KEY')

# ========== 初始化 OpenAI 客戶端 ==========
print("初始化 OpenAI 客戶端...")
try:
    api_key = get_api_key()
    client = OpenAI(api_key=api_key)
    print("✓ OpenAI 客戶端初始化成功")
except Exception as e:
    print(f"✗ 錯誤: OpenAI 初始化失敗 - {e}")
    exit(1)

# ========== 情緒關鍵字字典 ==========
emotion_keywords = {
    "生氣/不滿": ["生氣", "氣死", "不爽", "太誇張", "怎麼會", "什麼鬼", "受不了", "投訴", "抱怨", "不滿"],
    "焦慮/擔心": ["擔心", "害怕", "緊張", "怎麼辦", "來不及", "急", "趕快", "快一點", "著急"],
    "滿意/開心": ["謝謝", "太好了", "不錯", "很棒", "滿意", "開心", "感謝", "好的"],
    "疑惑/困擾": ["不懂", "看不懂", "怎麼", "為何", "搞不清楚", "不知道", "不確定", "疑問"],
}

def transcribe_with_whisper_api(audio_path):
    """使用 Whisper API 進行語音辨識（快速版）"""
    try:
        print(f"  - 上傳音檔到 Whisper API...")
        
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="zh",
                response_format="verbose_json",  # 獲取時間戳
                prompt="這是客服與客戶的對話，請正確辨識數字和專業術語。"
            )
        
        # 轉換為與原本相同的格式
        segments = []
        if hasattr(transcript, 'segments') and transcript.segments:
            for seg in transcript.segments:
                segments.append({
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text
                })
        else:
            # 如果沒有分段，創建單一段落
            segments.append({
                "start": 0,
                "end": transcript.duration if hasattr(transcript, 'duration') else 0,
                "text": transcript.text
            })
        
        return segments
    
    except Exception as e:
        print(f"  ✗ Whisper API 錯誤: {e}")
        raise

def extract_audio_features_fast(audio_path, start_time, end_time):
    """快速音頻特徵提取（僅提取關鍵特徵）"""
    try:
        # 使用較低採樣率加快處理
        y, sr = librosa.load(audio_path, sr=8000, offset=start_time, duration=end_time-start_time)
        
        if len(y) == 0:
            return None
        
        # 只提取最關鍵的特徵
        rms = librosa.feature.rms(y=y)[0]
        avg_energy = float(np.mean(rms))
        
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        avg_zcr = float(np.mean(zcr))
        
        return {
            'avg_energy': avg_energy,
            'avg_zcr': avg_zcr,
            'duration': end_time - start_time
        }
    except Exception as e:
        return None

def analyze_emotion_fast(text, audio_features=None):
    """快速情緒分析（文本為主 + 簡單音頻特徵）"""
    emotion_scores = {
        "生氣/不滿": 0,
        "焦慮/擔心": 0,
        "滿意/開心": 0,
        "疑惑/困擾": 0,
        "平靜/中性": 0
    }
    
    # 文本關鍵字分析
    for emotion, keywords in emotion_keywords.items():
        for keyword in keywords:
            if keyword in text:
                emotion_scores[emotion] += 2
    
    # 簡化的音頻分析
    if audio_features:
        # 高能量 → 情緒激動
        if audio_features['avg_energy'] > 0.05:
            emotion_scores["生氣/不滿"] += 1
            emotion_scores["焦慮/擔心"] += 0.5
        
        # 高過零率 → 語速快 → 焦慮
        if audio_features['avg_zcr'] > 0.15:
            emotion_scores["焦慮/擔心"] += 1
    
    # 返回主要情緒
    if max(emotion_scores.values()) > 0:
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = emotion_scores[dominant_emotion] / sum(emotion_scores.values())
        return dominant_emotion, confidence
    else:
        return "平靜/中性", 0.5

def classify_and_summarize(dialogue_text):
    """使用 GPT 進行分類和摘要"""
    try:
        prompt = f"""請分析以下客服對話，完成兩項任務：
1. 分類：判斷屬於「疾病險」、「旅行平安險」、「輔助器具險」或「其他」
2. 摘要：用30-50字總結對話重點

對話內容：
{dialogue_text[:1500]}

請用JSON格式回覆：
{{"class": "分類結果", "abstract": "摘要內容"}}"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1
        )
        
        result = json.loads(response.choices[0].message.content.strip())
        return result.get("class", "其他"), result.get("abstract", "無摘要")
    
    except Exception as e:
        print(f"  ✗ GPT 分類錯誤: {e}")
        return "其他", "分類失敗"

def separate_speakers(segments):
    """說話人分離（基於關鍵字和時間間隔）"""
    dialogue = []
    current_speaker = "客服"
    
    客服關鍵字 = ["您好", "請問", "幫您", "為您", "感謝", "歡迎", "服務", "這邊", "我們", "可以為"]
    客戶關鍵字 = ["我想", "我要", "我的", "可以嗎", "我是", "我有", "幫我"]
    
    for i, seg in enumerate(segments):
        text = seg["text"].strip()
        start = seg["start"]
        end = seg["end"]
        
        # 優先根據關鍵字判斷
        if any(kw in text for kw in 客服關鍵字):
            speaker = "客服"
        elif any(kw in text for kw in 客戶關鍵字):
            speaker = "客戶"
        else:
            # 根據時間間隔判斷（超過2秒可能換人）
            if i > 0 and start - segments[i-1]["end"] > 2.0:
                speaker = "客戶" if current_speaker == "客服" else "客服"
            else:
                speaker = current_speaker
        
        dialogue.append((start, speaker, end, text))
        current_speaker = speaker
    
    return dialogue

def process_audio_file(audio_file):
    """處理單個音檔（使用 Whisper API）"""
    print(f"\n{'='*60}")
    print(f"處理音檔: {audio_file.name}")
    print(f"開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('-' * 60)
    
    total_start = time()
    base_name = audio_file.stem
    
    # ========== Step 1: Whisper API 辨識（最快！）==========
    step1_start = time()
    try:
        segments = transcribe_with_whisper_api(str(audio_file))
        step1_time = time() - step1_start
        print(f"  ✓ 語音辨識完成: {step1_time:.2f}秒")
    except Exception as e:
        print(f"  ✗ 語音辨識失敗: {e}")
        return None
    
    # ========== Step 2: 說話人分離 ==========
    step2_start = time()
    dialogue = separate_speakers(segments)
    step2_time = time() - step2_start
    print(f"  ✓ 說話人分離完成: {step2_time:.2f}秒")
    
    # ========== Step 3 & 4: 並行處理（分類 + 情緒分析）==========
    step3_start = time()
    
    # 準備對話文本
    dialogue_text = "\n".join([f"{speaker}: {text}" for _, speaker, _, text in dialogue])
    
    # 使用多線程並行處理
    with ThreadPoolExecutor(max_workers=2) as executor:
        # 線程1: GPT分類和摘要
        future_classify = executor.submit(classify_and_summarize, dialogue_text)
        
        # 線程2: 情緒分析（主線程也可以做）
        customer_emotions = []
        customer_dialogues = []
        
        for start, speaker, end, text in dialogue:
            if speaker == "客戶":
                # 提取音頻特徵（可選，如果想要更高準確度）
                audio_features = extract_audio_features_fast(str(audio_file), start, end)
                emotion, confidence = analyze_emotion_fast(text, audio_features)
                customer_emotions.append(emotion)
                customer_dialogues.append((start, end, text, emotion, confidence))
        
        # 等待分類完成
        problem_type, abstract = future_classify.result()
    
    step3_time = time() - step3_start
    print(f"  ✓ 分類與情緒分析完成: {step3_time:.2f}秒")
    print(f"  → 分類結果: {problem_type}")
    
    # ========== Step 5: 儲存結果 ==========
    step4_start = time()
    
    # 確定儲存路徑
    class_folder = CLASS_FOLDERS.get(problem_type, CLASS_FOLDERS["其他"])
    text_output_folder = os.path.join(class_folder, "voice_text")
    emo_output_folder = os.path.join(class_folder, "voice_emo")
    
    # 儲存逐字稿
    text_file = os.path.join(text_output_folder, f"{base_name}.txt")
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(f"音檔: {audio_file.name}\n")
        f.write(f"分類: {problem_type}\n")
        f.write(f"處理時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"摘要: {abstract}\n")
        f.write("="*60 + "\n\n")
        
        for start, speaker, end, text in dialogue:
            m, s = divmod(int(start), 60)
            f.write(f"[{m:02d}:{s:02d}] {speaker}: {text}\n")
    
    # 儲存情緒分析
    emo_file = os.path.join(emo_output_folder, f"{base_name}_emotion.txt")
    with open(emo_file, "w", encoding="utf-8") as f:
        f.write(f"音檔: {audio_file.name}\n")
        f.write(f"分類: {problem_type}\n")
        f.write(f"處理時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"摘要: {abstract}\n")
        f.write("="*60 + "\n\n")
        
        # 詳細情緒分析
        f.write("【客戶情緒分析】\n")
        f.write("-"*60 + "\n")
        
        for start, end, text, emotion, confidence in customer_dialogues:
            m, s = divmod(int(start), 60)
            f.write(f"[{m:02d}:{s:02d}] {text}\n")
            f.write(f"  → 情緒: {emotion} (信心度: {confidence:.1%})\n\n")
        
        # 統計摘要
        if customer_emotions:
            f.write("\n" + "="*60 + "\n")
            f.write("【情緒統計摘要】\n")
            f.write("-"*60 + "\n")
            
            emotion_counter = Counter(customer_emotions)
            total_count = len(customer_emotions)
            
            for emo, count in emotion_counter.most_common():
                percentage = (count / total_count) * 100
                f.write(f"{emo}: {count}次 ({percentage:.1f}%)\n")
            
            # 主要情緒
            dominant_emotion = emotion_counter.most_common(1)[0][0]
            f.write(f"\n主要情緒: {dominant_emotion}\n")
            
            # 品質評估
            negative_count = sum(1 for e in customer_emotions if e in ["生氣/不滿", "焦慮/擔心"])
            positive_count = sum(1 for e in customer_emotions if e == "滿意/開心")
            negative_ratio = negative_count / total_count
            positive_ratio = positive_count / total_count
            
            f.write("\n【對話品質評估】\n")
            f.write("-"*60 + "\n")
            
            if negative_ratio > 0.5:
                f.write(f"⚠ 警示: 客戶負面情緒佔 {negative_ratio:.1%}，建議重點關注\n")
            elif positive_ratio > 0.3:
                f.write(f"✓ 良好: 客戶正向情緒佔 {positive_ratio:.1%}\n")
            else:
                f.write(f"→ 中性: 客戶情緒整體平穩\n")
            
            # 情緒趨勢
            if total_count >= 4:
                mid_point = total_count // 2
                early_negative = sum(1 for e in customer_emotions[:mid_point] if e in ["生氣/不滿", "焦慮/擔心"])
                late_negative = sum(1 for e in customer_emotions[mid_point:] if e in ["生氣/不滿", "焦慮/擔心"])
                
                if late_negative < early_negative:
                    f.write("↗ 趨勢: 客戶情緒逐漸好轉\n")
                elif late_negative > early_negative:
                    f.write("↘ 趨勢: 客戶情緒逐漸惡化\n")
                else:
                    f.write("→ 趨勢: 情緒保持穩定\n")
    
    step4_time = time() - step4_start
    print(f"  ✓ 檔案儲存完成: {step4_time:.2f}秒")
    
    # 總結
    total_time = time() - total_start
    print('-' * 60)
    print(f"完成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"總處理時間: {total_time:.2f}秒")
    print(f"  - 語音辨識: {step1_time:.2f}秒 ({step1_time/total_time*100:.1f}%)")
    print(f"  - 說話人分離: {step2_time:.2f}秒 ({step2_time/total_time*100:.1f}%)")
    print(f"  - 分類+情緒: {step3_time:.2f}秒 ({step3_time/total_time*100:.1f}%)")
    print(f"  - 檔案儲存: {step4_time:.2f}秒 ({step4_time/total_time*100:.1f}%)")
    print(f"已儲存至: {class_folder}/")
    
    return {
        'file': audio_file.name,
        'time': total_time,
        'class': problem_type,
        'abstract': abstract,
        'emotions': dict(Counter(customer_emotions))
    }

# ========== 主程序 ==========
if __name__ == "__main__":
    print("="*60)
    print("語音分析系統 - Whisper API 版本")
    print("="*60)
    
    # 尋找音檔
    audio_files = list(Path(input_folder).glob("*.wav")) + \
                  list(Path(input_folder).glob("*.mp3")) + \
                  list(Path(input_folder).glob("*.m4a"))
    
    if not audio_files:
        print(f"\n✗ 在 {input_folder} 資料夾中找不到音檔")
        print("支援格式: .wav, .mp3, .m4a")
    else:
        print(f"\n找到 {len(audio_files)} 個音檔")
        print(f"使用 Whisper API + GPT-4o-mini")
        print("="*60)
        
        overall_start = time()
        results = []
        
        # 處理每個音檔
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}]")
            result = process_audio_file(audio_file)
            if result:
                results.append(result)
        
        # 總結報告
        overall_time = time() - overall_start
        
        print("\n" + "="*60)
        print("處理完成！")
        print("="*60)
        print(f"\n總處理時間: {overall_time:.2f}秒 ({overall_time/60:.2f}分鐘)")
        print(f"平均每檔: {overall_time/len(results):.2f}秒" if results else "")
        
        print(f"\n處理摘要:")
        print("-"*60)
        for r in results:
            print(f"✓ {r['file']}")
            print(f"  時間: {r['time']:.2f}秒 | 分類: {r['class']}")
            print(f"  摘要: {r['abstract'][:50]}...")
            if r['emotions']:
                top_emotion = max(r['emotions'].items(), key=lambda x: x[1])
                print(f"  主要情緒: {top_emotion[0]} ({top_emotion[1]}次)")
            print()
        
        print("="*60)
        print("檔案儲存結構:")
        for class_name, folder in CLASS_FOLDERS.items():
            count = sum(1 for r in results if r['class'] == class_name)
            if count > 0:
                print(f"  {class_name} ({count}個): {folder}/")
        print("="*60)

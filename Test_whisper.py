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
import subprocess
import wave
import tempfile
from pydub import AudioSegment

# ========== 設定路徑 ==========
# input_folder = "voice_file"
input_folder = "test_file"

# ========== 音檔切分設定 ==========
MAX_FILE_SIZE_MB = 24  # Whisper API 限制 25MB，我們設 24MB 安全一點
MAX_CHUNK_DURATION = 180  # 最長 3 分鐘 (180 秒)

CLASS_FOLDERS = {
    "傷害健康保險": "class_disease",
    "旅平險": "class_travel", 
    "車險": "class_car",
    "其他": "class_other"
}

# 創建所有必要的文件夾
for folder in CLASS_FOLDERS.values():
    os.makedirs(os.path.join(folder, "voice_text"), exist_ok=True)
    os.makedirs(os.path.join(folder, "voice_emo"), exist_ok=True)
load_dotenv()

def get_api_key():
    return os.getenv('OPENAI_API_KEY')

def check_audio_format(audio_path):
    """檢查音檔格式詳細信息"""
    print(f"\n{'='*60}")
    print(f"檢查音檔格式: {audio_path.name}")
    print('-'*60)
    
    # 方法1: 使用 wave 模組（Python內建，只支援WAV）
    if audio_path.suffix.lower() == '.wav':
        try:
            with wave.open(str(audio_path), 'rb') as wav_file:
                print(f"✓ WAV 格式資訊:")
                print(f"  聲道數: {wav_file.getnchannels()}")
                print(f"  採樣寬度: {wav_file.getsampwidth()} bytes")
                print(f"  採樣率: {wav_file.getframerate()} Hz")
                print(f"  幀數: {wav_file.getnframes()}")
                print(f"  音頻長度: {wav_file.getnframes() / wav_file.getframerate():.2f} 秒")
                
                # 檢查是否是標準格式
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frame_rate = wav_file.getframerate()
                
                issues = []
                if channels > 2:
                    issues.append(f"聲道數異常 ({channels})")
                if sample_width not in [1, 2, 3, 4]:
                    issues.append(f"採樣寬度異常 ({sample_width})")
                if frame_rate not in [8000, 16000, 22050, 44100, 48000]:
                    issues.append(f"採樣率非標準 ({frame_rate})")
                
                if issues:
                    print(f"⚠ 發現問題:")
                    for issue in issues:
                        print(f"  - {issue}")
                    return False
                else:
                    print(f"✓ WAV 格式正常")
                    return True
                    
        except Exception as e:
            print(f"✗ WAV 檢查失敗: {e}")
    
    # 方法2: 使用 ffprobe（如果有安裝 ffmpeg）
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_format', '-show_streams', 
             '-print_format', 'json', str(audio_path)],
            capture_output=True, 
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            info = json.loads(result.stdout)
            print(f"\n✓ FFprobe 詳細資訊:")
            
            if 'format' in info:
                fmt = info['format']
                print(f"  格式: {fmt.get('format_name', 'N/A')}")
                print(f"  長度: {float(fmt.get('duration', 0)):.2f} 秒")
                print(f"  位元率: {int(fmt.get('bit_rate', 0))/1000:.0f} kbps")
            
            if 'streams' in info and len(info['streams']) > 0:
                stream = info['streams'][0]
                print(f"  編碼器: {stream.get('codec_name', 'N/A')}")
                print(f"  採樣率: {stream.get('sample_rate', 'N/A')} Hz")
                print(f"  聲道數: {stream.get('channels', 'N/A')}")
            
            return True
    except FileNotFoundError:
        print(f"\n⚠ 未安裝 ffmpeg，無法使用 ffprobe")
        print(f"  安裝方式: brew install ffmpeg (macOS)")
    except Exception as e:
        print(f"\n⚠ FFprobe 檢查失敗: {e}")
    
    print('='*60)
    return None

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

def split_audio_file(audio_path, max_duration=MAX_CHUNK_DURATION):
    """切分大型音檔為多個小片段"""
    try:
        print(f"  - 音檔過大，正在切分...")
        
        # 讀取音檔
        audio = AudioSegment.from_file(str(audio_path))
        duration_seconds = len(audio) / 1000  # pydub 使用毫秒
        
        print(f"    總長度: {duration_seconds:.1f} 秒")
        print(f"    切分為每段 {max_duration} 秒")
        
        # 計算需要切成幾段
        num_chunks = int(np.ceil(duration_seconds / max_duration))
        print(f"    將切分為 {num_chunks} 段")
        
        # 切分音檔
        chunks = []
        for i in range(num_chunks):
            start_ms = i * max_duration * 1000
            end_ms = min((i + 1) * max_duration * 1000, len(audio))
            
            chunk = audio[start_ms:end_ms]
            
            # 儲存到臨時檔案
            temp_file = tempfile.NamedTemporaryFile(
                suffix='.wav', 
                delete=False,
                dir=tempfile.gettempdir()
            )
            chunk.export(temp_file.name, format='wav')
            
            chunks.append({
                'file': temp_file.name,
                'start_time': i * max_duration,
                'end_time': min((i + 1) * max_duration, duration_seconds)
            })
            
            print(f"      段 {i+1}/{num_chunks}: {chunks[-1]['start_time']:.1f}s - {chunks[-1]['end_time']:.1f}s")
        
        return chunks
    
    except Exception as e:
        print(f"  ✗ 音檔切分失敗: {e}")
        # 嘗試使用 ffmpeg 切分
        try:
            return split_audio_with_ffmpeg(audio_path, max_duration)
        except:
            raise

def split_audio_with_ffmpeg(audio_path, max_duration=MAX_CHUNK_DURATION):
    """使用 ffmpeg 切分音檔（備用方案）"""
    print(f"  - 使用 ffmpeg 切分音檔...")
    
    # 獲取音檔長度
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', str(audio_path)],
        capture_output=True,
        text=True
    )
    
    duration_seconds = float(result.stdout.strip())
    num_chunks = int(np.ceil(duration_seconds / max_duration))
    
    print(f"    總長度: {duration_seconds:.1f} 秒")
    print(f"    將切分為 {num_chunks} 段")
    
    chunks = []
    for i in range(num_chunks):
        start_time = i * max_duration
        
        temp_file = tempfile.NamedTemporaryFile(
            suffix='.wav',
            delete=False,
            dir=tempfile.gettempdir()
        )
        
        # 使用 ffmpeg 切分
        subprocess.run(
            ['ffmpeg', '-i', str(audio_path), '-ss', str(start_time),
             '-t', str(max_duration), '-acodec', 'copy', temp_file.name],
            capture_output=True
        )
        
        chunks.append({
            'file': temp_file.name,
            'start_time': start_time,
            'end_time': min(start_time + max_duration, duration_seconds)
        })
        
        print(f"      段 {i+1}/{num_chunks}: {start_time:.1f}s - {chunks[-1]['end_time']:.1f}s")
    
    return chunks

def remove_repetitive_segments(segments, similarity_threshold=0.85, consecutive_limit=2):
    """移除重複的語音片段（處理 Whisper 幻覺問題）- 連續2次即去重"""
    if not segments or len(segments) < 2:
        return segments
    
    cleaned_segments = []
    skip_until_index = -1
    
    for i, seg in enumerate(segments):
        # 如果這個片段已被標記為要跳過
        if i <= skip_until_index:
            continue
        
        current_text = seg['text'].strip()
        
        # 檢查後續是否有連續重複
        consecutive_count = 1
        similar_texts = [current_text]  # 記錄所有相似的文本
        
        for j in range(i + 1, min(i + 15, len(segments))):  # 往後看15個片段
            next_text = segments[j]['text'].strip()
            
            # 計算相似度
            similarity = calculate_similarity(current_text, next_text, similar_texts)
            
            if similarity >= similarity_threshold:
                consecutive_count += 1
                similar_texts.append(next_text)
            else:
                break
        
        # 如果發現連續重複超過限制（2次），只保留第一個
        if consecutive_count >= consecutive_limit:
            print(f"  [去重複] 發現 {consecutive_count} 個重複片段: '{current_text[:30]}...'")
            cleaned_segments.append(seg)
            skip_until_index = i + consecutive_count - 1
        else:
            cleaned_segments.append(seg)
        pattern_length, pattern_count = detect_alternating_pattern(segments, i, similarity_threshold)
        
        if pattern_length > 0 and pattern_count >= 2:  # 至少重複2次完整模式
            total_segments = pattern_length * pattern_count
            print(f"  [去重複-交替] 發現 {pattern_count} 組重複模式 (每組{pattern_length}句): '{current_text[:30]}...'")
            # 只保留第一組模式
            for k in range(pattern_length):
                if i + k < len(segments):
                    cleaned_segments.append(segments[i + k])
            skip_until_index = i + total_segments - 1
            continue

    removed_count = len(segments) - len(cleaned_segments)
    if removed_count > 0:
        print(f"  ✓ 移除了 {removed_count} 個重複片段")
    
    return cleaned_segments

def detect_alternating_pattern(segments, start_idx, similarity_threshold=0.85):
    """
    檢測交替重複模式
    返回: (pattern_length, pattern_count)
    例如: A-B-A-B-A-B 返回 (2, 3) 表示2句一組，重複3次
    """
    max_check = min(20, len(segments) - start_idx)  # 最多往後看20個片段
    
    # 嘗試不同的模式長度 (2句一組、3句一組等)
    for pattern_len in range(2, min(6, max_check // 2 + 1)):
        if start_idx + pattern_len * 2 > len(segments):
            continue
        
        # 提取第一組模式
        first_pattern = [segments[start_idx + k]['text'].strip() for k in range(pattern_len)]
        
        # 檢查後續是否有相同模式重複
        repeat_count = 1
        current_pos = start_idx + pattern_len
        
        while current_pos + pattern_len <= len(segments):
            # 提取當前組
            current_pattern = [segments[current_pos + k]['text'].strip() for k in range(pattern_len)]
            
            # 比對是否與第一組模式相似
            is_match = True
            for j in range(pattern_len):
                similarity = calculate_similarity(first_pattern[j], current_pattern[j])
                if similarity < similarity_threshold:
                    is_match = False
                    break
            
            if is_match:
                repeat_count += 1
                current_pos += pattern_len
            else:
                break
        
        # 如果找到至少重複2次的模式，返回結果
        if repeat_count >= 2:
            return pattern_len, repeat_count
    
    return 0, 0



def calculate_similarity(text1, text2, context_texts=None):
    """
    計算兩段文本的相似度
    text1: 當前文本
    text2: 比對文本
    context_texts: 已發現的相似文本列表（用於檢測變體）
    """
    # 完全相同
    if text1 == text2:
        return 1.0
    
    if len(text1) == 0 or len(text2) == 0:
        return 0.0
    
    # 檢查是否為子字串關係
    if text1 in text2 or text2 in text1:
        shorter = min(len(text1), len(text2))
        longer = max(len(text1), len(text2))
        return shorter / longer
    
    # 如果有上下文，檢查 text2 是否與任何已知的重複文本相似
    if context_texts and len(context_texts) > 1:
        for ctx_text in context_texts:
            if text2 == ctx_text or text2 in ctx_text or ctx_text in text2:
                return 0.95  # 高相似度
    
    # 計算字符級別的相似度（使用滑動窗口找最長公共子串）
    shorter = min(len(text1), len(text2))
    longer = max(len(text1), len(text2))
    
    max_match = 0
    for offset in range(-shorter, shorter):
        matches = 0
        for k in range(shorter):
            idx1 = k if offset >= 0 else k - offset
            idx2 = k + offset if offset >= 0 else k
            
            if 0 <= idx1 < len(text1) and 0 <= idx2 < len(text2):
                if text1[idx1] == text2[idx2]:
                    matches += 1
        max_match = max(max_match, matches)
    
    return max_match / longer

def transcribe_with_whisper_api(audio_path):
    """使用 Whisper API 進行語音辨識（支援大檔案自動切分）"""
    try:
        # 確保 audio_path 是 Path 對象
        if isinstance(audio_path, str):
            audio_path = Path(audio_path)
        
        # 檢查檔案大小
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        
        if file_size_mb > MAX_FILE_SIZE_MB:
            print(f"  - 檔案大小 {file_size_mb:.1f}MB 超過限制 {MAX_FILE_SIZE_MB}MB")
            
            # 切分音檔
            chunks = split_audio_file(audio_path, MAX_CHUNK_DURATION)
            
            # 處理每個片段
            all_segments = []
            
            for i, chunk_info in enumerate(chunks, 1):
                print(f"  - 處理片段 {i}/{len(chunks)} ({chunk_info['start_time']:.1f}s - {chunk_info['end_time']:.1f}s)...")
                
                try:
                    with open(chunk_info['file'], "rb") as audio_file:
                        transcript = client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file,
                            language="zh",
                            response_format="verbose_json",
                            temperature=0.0  # 降低溫度以減少幻覺
                        )
                    
                    # 處理這個片段的結果
                    if hasattr(transcript, 'segments') and transcript.segments:
                        for seg in transcript.segments:
                            try:
                                if hasattr(seg, 'start'):
                                    start = seg.start
                                    end = seg.end
                                    text = seg.text
                                elif isinstance(seg, dict):
                                    start = seg['start']
                                    end = seg['end']
                                    text = seg['text']
                                else:
                                    continue
                                
                                # 調整時間戳記（加上這個片段的起始時間）
                                all_segments.append({
                                    "start": float(start) + chunk_info['start_time'],
                                    "end": float(end) + chunk_info['start_time'],
                                    "text": text.strip()
                                })
                            except Exception as e:
                                print(f"    [警告] 跳過無法解析的片段: {e}")
                                continue
                    else:
                        # 如果沒有分段，使用完整文本
                        full_text = transcript.text if hasattr(transcript, 'text') else ""
                        if full_text:
                            all_segments.append({
                                "start": chunk_info['start_time'],
                                "end": chunk_info['end_time'],
                                "text": full_text.strip()
                            })
                    
                    print(f"    ✓ 片段 {i} 完成，獲得 {len(all_segments)} 個累計片段")
                
                finally:
                    # 清理臨時檔案
                    try:
                        os.unlink(chunk_info['file'])
                    except:
                        pass
            
            segments = all_segments
            print(f"  ✓ 所有片段處理完成，共 {len(segments)} 個語音片段")
        
        else:
            # 檔案大小正常，直接處理
            print(f"  - 上傳音檔到 Whisper API (大小: {file_size_mb:.1f}MB)...")
            
            with open(audio_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="zh",
                    response_format="verbose_json",
                    temperature=0.0  # 降低溫度以減少幻覺
                )
            
            print(f"  - API 回傳成功，解析結果中...")
            
            segments = []
            
            # 檢查是否有 segments 屬性
            if hasattr(transcript, 'segments') and transcript.segments:
                print(f"  - 找到 {len(transcript.segments)} 個片段")
                for seg in transcript.segments:
                    try:
                        if hasattr(seg, 'start'):
                            start = seg.start
                            end = seg.end
                            text = seg.text
                        elif isinstance(seg, dict):
                            start = seg['start']
                            end = seg['end']
                            text = seg['text']
                        else:
                            continue
                        
                        segments.append({
                            "start": float(start),
                            "end": float(end),
                            "text": text.strip()
                        })
                    except Exception as e:
                        print(f"  [警告] 跳過無法解析的片段: {e}")
                        continue
            
            # 如果沒有 segments，使用完整文本並嘗試自己分段
            if not segments:
                print(f"  - 沒有片段資訊，使用完整文本")
                full_text = transcript.text if hasattr(transcript, 'text') else ""
                
                if not full_text:
                    raise ValueError("API 沒有返回任何文本內容")
                
                # 簡單按句號、問號、逗號分段
                sentences = []
                current = ""
                for char in full_text:
                    current += char
                    if char in ['。', '？', '！', '，', '\n']:
                        if current.strip():
                            sentences.append(current.strip())
                        current = ""
                if current.strip():
                    sentences.append(current.strip())
                
                # 為每個句子估算時間
                avg_chars_per_second = 5
                current_time = 0
                
                for sentence in sentences:
                    duration = len(sentence) / avg_chars_per_second
                    segments.append({
                        "start": current_time,
                        "end": current_time + duration,
                        "text": sentence
                    })
                    current_time += duration + 0.5
        
        if not segments:
            raise ValueError("無法從 API 回應中提取任何內容")
        
        # ========== 移除重複片段 ==========
        print(f"  - 檢查重複片段...")
        segments = remove_repetitive_segments(segments, similarity_threshold=0.85, consecutive_limit=2)
        
        print(f"  ✓ 成功獲得 {len(segments)} 個語音片段")
        
        return segments
    
    except Exception as e:
        print(f"  ✗ Whisper API 錯誤: {e}")
        import traceback
        traceback.print_exc()
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
1. 摘要：用30-50字總結對話重點，需要點出用的險種如「強制險」、「醫師責任險」等
2. 分類：利用摘要判斷屬於「傷害健康保險」、「旅平險」、「車險」或「其他」
   - 若有提到「車險」、「強制險」就歸類為車險
   - 若有提到「不便險」、「旅平險」、「旅行平安險」就優先歸類為旅平險
   - 若有提到「傷害健康保險」、「傷害險」就優先歸類為傷害健康保險
   - 其他情況只要不是車險或旅平險或傷害健康保險，就歸類為「其他」，

對話內容：
{dialogue_text[:1500]}

請只回傳JSON格式，不要有任何其他文字：
{{"abstract": "摘要內容", "class": "分類結果"}}"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1
        )
        
        content = response.choices[0].message.content.strip()
        
        # 處理可能的 markdown 代碼塊
        if content.startswith("```"):
            # 移除 ```json 和 ```
            content = content.replace("```json", "").replace("```", "").strip()
        
        # 嘗試找到 JSON 部分
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group()
        
        # 解析 JSON
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            # 如果還是失敗，打印出來看看
            print(f"  [DEBUG] GPT 回傳內容: {content[:200]}")
            raise
        
        # 驗證必要欄位
        class_result = result.get("class", "其他")
        abstract_result = result.get("abstract", "無摘要")
        
        # 確保分類在允許的範圍內
        valid_classes = ["傷害健康保險", "旅平險", "車險", "其他"]
        if class_result not in valid_classes:
            print(f"  [警告] 分類結果 '{class_result}' 不在預期範圍，改為'其他'")
            class_result = "其他"
        
        return class_result, abstract_result
    
    except Exception as e:
        print(f"  ✗ GPT 分類錯誤: {e}")
        # 打印更多調試信息
        try:
            print(f"  [DEBUG] Response content: {response.choices[0].message.content[:200]}")
        except:
            pass
        return "其他", "分類失敗"

def separate_speakers(segments):
    """說話人分離（基於關鍵字、語氣和時間間隔）"""
    dialogue = []
    current_speaker = "客服"
    
    # 擴充關鍵字庫
    客服關鍵字 = [
        "您好", "請問", "幫您", "為您", "感謝", "歡迎", "服務", "這邊", 
        "我們", "可以為", "麻煩", "謝謝您", "貴姓", "查詢", "幫你",
        "訊息服務", "小姐", "先生"
    ]
    客戶關鍵字 = [
        "我想", "我要", "我的", "可以嗎", "我是", "我有", "幫我",
        "我姓", "想要", "需要"
    ]
    
    for i, seg in enumerate(segments):
        text = seg["text"].strip()
        start = seg["start"]
        end = seg["end"]
        
        # 計算關鍵字匹配分數
        客服分數 = sum(1 for kw in 客服關鍵字 if kw in text)
        客戶分數 = sum(1 for kw in 客戶關鍵字 if kw in text)
        
        # 優先根據關鍵字判斷
        if 客服分數 > 客戶分數:
            speaker = "客服"
        elif 客戶分數 > 客服分數:
            speaker = "客戶"
        else:
            # 沒有明確關鍵字時的判斷邏輯
            
            # 規則1: 第一句通常是客服
            if i == 0:
                speaker = "客服"
            
            # 規則2: 超過1.5秒靜默 = 換人說話
            elif i > 0 and start - segments[i-1]["end"] > 1.5:
                speaker = "客戶" if current_speaker == "客服" else "客服"
            
            # 規則3: 非常短的回應（如"是"、"好"）通常是回應前一個說話人
            elif len(text) <= 3 and i > 0:
                # 短回應保持前一個說話人
                speaker = current_speaker
            
            # 規則4: 句子長度判斷（客服通常說較長的話）
            elif len(text) > 20:
                speaker = "客服"
            
            else:
                # 預設延續前一個說話人
                speaker = current_speaker
        
        dialogue.append((start, speaker, end, text))
        current_speaker = speaker
    
    # 後處理：修正明顯錯誤的標記
    # 如果連續多個短句都是同一人，可能需要調整
    for i in range(1, len(dialogue) - 1):
        prev_speaker = dialogue[i-1][1]
        curr_speaker = dialogue[i][1]
        next_speaker = dialogue[i+1][1] if i+1 < len(dialogue) else curr_speaker
        curr_text = dialogue[i][3]
        
        # 如果前後都是同一個人，當前是另一個人，且當前文本很短，可能標記錯誤
        if prev_speaker == next_speaker and curr_speaker != prev_speaker and len(curr_text) <= 5:
            # 修正為與前後一致
            dialogue[i] = (dialogue[i][0], prev_speaker, dialogue[i][2], dialogue[i][3])
    
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
                # 快速模式：只用文本分析，跳過音頻特徵提取
                emotion, confidence = analyze_emotion_fast(text, None)
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
        
        # 選項：是否檢查所有音檔格式
        check_format = input("\n是否檢查所有音檔格式？(y/n，預設n): ").lower().strip() == 'y'
        
        if check_format:
            print("\n" + "="*60)
            print("開始檢查音檔格式...")
            print("="*60)
            problem_files = []
            
            for audio_file in audio_files:
                status = check_audio_format(audio_file)
                if status == False:
                    problem_files.append(audio_file.name)
            
            if problem_files:
                print(f"\n⚠ 發現 {len(problem_files)} 個可能有問題的音檔:")
                for fname in problem_files:
                    print(f"  - {fname}")
                print("\n建議使用 ffmpeg 轉換為標準格式:")
                print("  ffmpeg -i input.wav -ar 16000 -ac 1 -sample_fmt s16 output.wav")
                
                proceed = input("\n是否繼續處理？(y/n): ").lower().strip()
                if proceed != 'y':
                    print("已取消處理")
                    exit(0)
        
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

# 主要改進：
# 1. 簡化音頻處理，避免引入 artifacts
# 2. 優化 Whisper prompt 策略（極簡化）
# 3. 強化幻覺檢測與過濾
# 4. 添加智慧重試機制
# 5. 移除過度的音量處理
# 6. 全部使用繁體中文

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
from pydub.silence import detect_silence
from dotenv import load_dotenv
import noisereduce as nr
import soundfile as sf

load_dotenv()

# ========== 設定路徑 ==========
input_folder = "recordings"
processed_folder = "recordings_processed"
os.makedirs(processed_folder, exist_ok=True)

MAX_FILE_SIZE_MB = 24
MAX_CHUNK_DURATION = 180

CLASS_FOLDERS = {
    "傷害健康保險": "class_disease",
    "旅平險": "class_travel",
    "車險": "class_car",
    "其他": "class_other"
}

for folder in CLASS_FOLDERS.values():
    os.makedirs(os.path.join(folder, "voice_text"), exist_ok=True)
    os.makedirs(os.path.join(folder, "voice_emo"), exist_ok=True)

api_key = os.getenv('OPENAI_API_KEY')

# ========== 核心詞彙庫（精簡版，避免重疊） ==========
CORE_VOCABULARY = {
    # Whisper 階段：極簡關鍵詞（5-10個最容易錯的）
    "whisper_keywords": [
        "富邦產險", "敝姓溫", "敝姓廖",
        "旅平險", "不便險", "強制險", "車險",
        "投保", "續保", "末四碼"
    ],
    
    # GPT 修正階段：完整的錯字對照表
    "correction_mapping": {
        # 最高優先級
        "富邦產險": ["禁邦產險", "富邦產線", "富邦產先", "溫哺巴數", "客邦產險"],
        "末四碼": ["默斯碼", "莫斯碼", "莫四碼", "末斯碼", "默四碼"],
        "不便險": ["不便雞", "不便鴨", "不便線"],
        "旅平險": ["旅平線", "旅評險", "旅萍險"],
        "身分證": ["身份證", "身分証"],
        "信用卡": ["信用卡", "新用卡"],
        
        # 高優先級
        "投保": ["頭包", "投報", "投堡"],
        "保費": ["報廢", "保肥", "報費"],
        "理賠": ["裡酬", "理配", "理陪"],
        "強制險": ["強迫險", "強制線", "強制先"],
        "續保": ["續報", "續堡", "續包"],
        "保單": ["報單", "保丹", "報丹"],
        
        # 中優先級
        "產險": ["產雞", "產線", "產先"],
        "車險": ["車線", "車先", "計車惜", "汽車惜"],
        "旅行險": ["旅行卷", "旅行線", "旅行先"],
        "要保人": ["要報人", "要包人"],
        "被保險人": ["被報險人", "被包險人"],
        
        # 低優先級（禮貌用語）
        "您好": ["您豪", "您號", "您毫"],
        "敝姓": ["敝性", "幣姓", "弊姓"],
        "麻煩": ["麻凡", "麻煩"],
        "謝謝": ["謝謝", "些些"],
        "稍等": ["稍鄧", "少等"],
        "抱歉": ["抱歉", "保歉"],
        "電子郵件": ["電子郵件", "電資郵件"],
        "通訊地址": ["通信地址", "通訊地址"],
        "Gmail信箱": ["居民姓鄉", "G妹兒信箱", "雞妹信箱"],
        "為您服務": ["為您服務", "位您服務", "威寧服務"]
    },
    
    # 情緒關鍵詞
    "emotion_keywords": {
        "生氣/不滿": ["生氣", "氣死", "不爽", "太誇張", "怎麼會", "什麼鬼", "受不了", "投訴", "抱怨", "不滿"],
        "焦慮/擔心": ["擔心", "害怕", "緊張", "怎麼辦", "來不及", "急", "趕快", "快一點", "著急"],
        "滿意/開心": ["謝謝", "太好了", "不錯", "很棒", "滿意", "開心", "感謝", "好的"],
        "疑惑/困惑": ["不懂", "看不懂", "怎麼", "為何", "搞不清楚", "不知道", "不確定", "疑問"]
    }
}


def build_whisper_prompt():
    """
    構建極簡的 Whisper prompt
    策略：只包含最關鍵、最容易被誤識別的詞彙
    限制：約40個中文字（60 tokens以內）
    """
    # 極簡 prompt - 只用最核心的
    prompt = "富邦產險您好，敝姓溫。旅平險、不便險、強制險。請提供身分證和信用卡末四碼。"
    
    return prompt


def build_correction_prompt():
    """
    構建 GPT 錯字修正的專用提示詞
    與 Whisper 階段完全分離
    """
    mapping = CORE_VOCABULARY["correction_mapping"]
    
    prompt_parts = ["**常見錯字修正對照表**（按優先級排序）：\n\n"]
    
    # 選擇前 30 個錯誤
    common_errors = list(mapping.items())[:30]
    
    for correct, wrongs in common_errors:
        wrong_list = " / ".join(wrongs[:4])
        prompt_parts.append(f"• {wrong_list} → {correct}\n")
    
    # 特別強調最常見的錯誤
    prompt_parts.append("\n**特別注意**：\n")
    prompt_parts.append("• 「默斯碼」、「莫斯碼」→ 末四碼\n")
    prompt_parts.append("• 「客邦產險」→ 富邦產險\n")
    
    return "".join(prompt_parts)


# ========== 初始化 ==========
print("初始化系統...")

try:
    client = OpenAI(api_key=api_key)
    print("✓ OpenAI 客戶端初始化成功")
except Exception as e:
    print(f"✗ 錯誤: OpenAI 初始化失敗 - {e}")
    exit(1)

print("✓ 使用優化的提示詞策略（極簡化 Whisper prompt）")


# ========== 音訊預處理簡化版 ==========
def preprocess_audio_minimal(audio_path, denoise_level='light'):
    """
    最小化音訊預處理 - 避免引入 artifacts
    
    參數:
        denoise_level: 'none', 'light', 'medium'
    """
    print(f"\n【音訊預處理 - {denoise_level}模式】")
    
    original_stem = audio_path.stem
    
    suffix_map = {
        'none': '_converted',
        'light': '_light_clean',
        'medium': '_medium_clean'
    }
    suffix = suffix_map.get(denoise_level, '_converted')
    
    output_path = Path(processed_folder) / f"{original_stem}{suffix}.wav"
    
    if output_path.exists():
        print(f"  → 使用已處理的音訊: {output_path.name}")
        return output_path, original_stem
    
    try:
        if denoise_level == 'none':
            # 只做基本轉換
            print(f"  [1/1] 基本格式轉換...")
            subprocess.run(
                [
                    'ffmpeg', '-i', str(audio_path),
                    '-ar', '16000',  # Whisper 最佳採樣率
                    '-ac', '1',      # 單聲道
                    '-acodec', 'pcm_s16le',
                    '-y', str(output_path)
                ],
                capture_output=True,
                check=True
            )
            print(f"    ✓ 格式轉換完成")
            
        elif denoise_level == 'light':
            # 輕度處理（推薦）
            print(f"  [1/2] 輕度降噪和標準化...")
            subprocess.run(
                [
                    'ffmpeg', '-i', str(audio_path),
                    '-af',
                    # 只用最基本的濾波
                    'highpass=f=80,'            # 移除極低頻（保留更多語音）
                    'lowpass=f=8000,'           # 保留更多高頻（保留清晰度）
                    'afftdn=nf=-25,'            # 輕度FFT降噪
                    'loudnorm=I=-16:TP=-1.5',   # 標準化響度
                    '-ar', '16000',
                    '-ac', '1',
                    '-acodec', 'pcm_s16le',
                    '-y', str(output_path)
                ],
                capture_output=True,
                check=True
            )
            print(f"    ✓ FFmpeg處理完成（輕度模式）")
            
        else:  # medium
            # 中度處理（適用於很吵的環境）
            print(f"  [1/3] 中度降噪...")
            temp_path = Path(processed_folder) / f"{original_stem}_temp.wav"
            
            # Step 1: FFmpeg基礎處理
            subprocess.run(
                [
                    'ffmpeg', '-i', str(audio_path),
                    '-af',
                    'highpass=f=100,'
                    'lowpass=f=7000,'
                    'afftdn=nf=-20',
                    '-ar', '16000', '-ac', '1',
                    '-acodec', 'pcm_s16le',
                    '-y', str(temp_path)
                ],
                capture_output=True,
                check=True
            )
            
            # Step 2: Noisereduce（保守參數）
            print(f"  [2/3] 智慧降噪...")
            data, rate = sf.read(str(temp_path))
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            
            reduced = nr.reduce_noise(
                y=data, sr=rate,
                stationary=True,
                prop_decrease=0.75,  # 降低到75%（更保守）
                freq_mask_smooth_hz=250,
                time_mask_smooth_ms=25
            )
            
            # Step 3: 輕微標準化
            print(f"  [3/3] 最終標準化...")
            sf.write(str(temp_path), reduced, rate)
            subprocess.run(
                [
                    'ffmpeg', '-i', str(temp_path),
                    '-af', 'loudnorm=I=-16:TP=-1.5',
                    '-ar', '16000', '-ac', '1',
                    '-y', str(output_path)
                ],
                capture_output=True,
                check=True
            )
            
            temp_path.unlink()
            print(f"    ✓ 中度降噪完成")
        
        print(f"  ✓ 預處理完成: {output_path.name}")
        return output_path, original_stem
    
    except Exception as e:
        print(f"  ✗ 預處理失敗: {e}")
        return audio_path, audio_path.stem


def split_audio_file(audio_path, max_duration=MAX_CHUNK_DURATION):
    """切分大型音訊為多個小片段"""
    try:
        print(f"  - 音訊過大，正在切分...")
        
        audio = AudioSegment.from_file(str(audio_path))
        duration_seconds = len(audio) / 1000
        
        print(f"    總長度: {duration_seconds:.1f} 秒")
        print(f"    切分為每段 {max_duration} 秒")
        
        num_chunks = int(np.ceil(duration_seconds / max_duration))
        print(f"    將切分為 {num_chunks} 段")
        
        chunks = []
        for i in range(num_chunks):
            start_ms = i * max_duration * 1000
            end_ms = min((i + 1) * max_duration * 1000, len(audio))
            
            chunk = audio[start_ms:end_ms]
            
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
            
            print(f"      段 {i + 1}/{num_chunks}: {chunks[-1]['start_time']:.1f}s - {chunks[-1]['end_time']:.1f}s")
        
        return chunks
    
    except Exception as e:
        print(f"  ✗ 音訊切分失敗: {e}")
        raise


def detect_hallucination(segment):
    """
    檢測 Whisper 幻覺的多種模式（強化版）
    返回 (is_hallucination, reason)
    """
    text = segment['text'].strip()
    
    # 模式 1: 短詞大量重複
    words = text.replace('，', ' ').replace('。', ' ').split()
    if len(words) > 10:
        word_counts = Counter(words)
        most_common = word_counts.most_common(1)
        if most_common:
            most_common_word, count = most_common[0]
            if count / len(words) > 0.7 and len(most_common_word) <= 3:
                return True, f"單詞重複: '{most_common_word}' 重複 {count} 次"
    
    # 模式 2: 字元級重複
    if len(text) > 5:
        for i in range(len(text) - 4):
            if text[i] == text[i+1] == text[i+2] == text[i+3] == text[i+4]:
                return True, f"字元重複: '{text[i]}' 連續出現"
    
    # 模式 3: 極長的片段（可能是幻覺）
    if len(text) > 300:
        unique_chars = len(set(text))
        if unique_chars < len(text) * 0.1:
            return True, f"內容單調: 只有 {unique_chars} 種字元"
    
    # 模式 4: 常見幻覺短語
    hallucination_phrases = [
        "謝謝收看", "謝謝觀看", "字幕製作", "翻譯",
        "subscribe", "like", "訂閱", "按讚",
        "请订阅", "请按赞"
    ]
    lower_text = text.lower()
    for phrase in hallucination_phrases:
        if phrase in lower_text:
            return True, f"疑似幻覺短語: '{phrase}'"
    
    # 模式 5: 標點符號異常
    punctuation_count = sum(1 for c in text if c in '，。！？、')
    if len(text) > 20 and punctuation_count / len(text) > 0.3:
        return True, f"標點符號過多: {punctuation_count}/{len(text)}"
    
    return False, ""


def remove_repetitive_segments(segments, similarity_threshold=0.90, consecutive_limit=3, enable_strict_filter=False):
    """
    移除重複的語音片段（處理 Whisper 幻覺問題）
    
    參數:
        enable_strict_filter: False = 保守模式（保留更多內容，包括音樂）
                             True = 嚴格模式（激進過濾）
    """
    if not segments or len(segments) < 2:
        return segments
    
    cleaned_segments = []
    skip_until_index = -1
    hallucination_count = 0
    
    for i, seg in enumerate(segments):
        if i <= skip_until_index:
            continue
        
        current_text = seg['text'].strip()
        
        # 只檢測明顯的幻覺（放寬標準）
        is_hallucination, reason = detect_hallucination(seg)
        if is_hallucination:
            print(f"  [檢測幻覺] 時間 {seg['start']:.1f}s: {reason}")
            print(f"             內容: '{current_text[:60]}...'")
            hallucination_count += 1
            continue
        
        # 關閉"完全相同文本去重"邏輯（允許歌詞重複）
        # 只檢查連續的極端重複
        
        consecutive_count = 1
        similar_texts = [current_text]
        
        for j in range(i + 1, min(i + 10, len(segments))):
            next_text = segments[j]['text'].strip()
            
            similarity = calculate_similarity(current_text, next_text)
            
            if similarity >= similarity_threshold:
                consecutive_count += 1
                similar_texts.append(next_text)
            else:
                break
        
        # 提高重複閾值：至少3-4次才算重複（而不是2次）
        if consecutive_count >= consecutive_limit:
            print(f"  [去重複] 發現 {consecutive_count} 個重複片段: '{current_text[:30]}...'")
            cleaned_segments.append(seg)
            skip_until_index = i + consecutive_count - 1
        else:
            # 保留此片段
            cleaned_segments.append(seg)
    
    removed_count = len(segments) - len(cleaned_segments)
    if removed_count > 0:
        print(f"  ✓ 移除了 {removed_count} 個問題片段（幻覺: {hallucination_count}）")
    else:
        print(f"  ✓ 未檢測到需要移除的片段")
    
    return cleaned_segments


def calculate_similarity(text1, text2):
    """計算兩段文本的相似度"""
    if text1 == text2:
        return 1.0
    
    if len(text1) == 0 or len(text2) == 0:
        return 0.0
    
    if text1 in text2 or text2 in text1:
        shorter = min(len(text1), len(text2))
        longer = max(len(text1), len(text2))
        return shorter / longer
    
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


def transcribe_with_whisper_api(audio_path, use_context=False):
    """
    使用 Whisper API 進行語音辨識
    增強版：極簡prompt + 激進幻覺過濾
    
    參數:
        use_context: 是否使用上下文（預設False，避免誤導）
    """
    try:
        if isinstance(audio_path, str):
            audio_path = Path(audio_path)
        
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        
        # 構建極簡的 Whisper prompt
        whisper_prompt = build_whisper_prompt()
        
        print(f"  - 使用極簡保險詞彙提示（{len(whisper_prompt)}字）")
        
        if file_size_mb > MAX_FILE_SIZE_MB:
            print(f"  - 檔案大小 {file_size_mb:.1f}MB 超過限制 {MAX_FILE_SIZE_MB}MB")
            
            chunks = split_audio_file(audio_path, MAX_CHUNK_DURATION)
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
                            temperature=0.0,  # 最低溫度，減少幻覺
                            prompt=whisper_prompt
                        )
                    
                    segments_added = 0
                    
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
                                
                                adjusted_start = float(start) + chunk_info['start_time']
                                adjusted_end = float(end) + chunk_info['start_time']
                                
                                all_segments.append({
                                    "start": adjusted_start,
                                    "end": adjusted_end,
                                    "text": text.strip()
                                })
                                segments_added += 1
                            except Exception as e:
                                print(f"    [警告] 跳過無法解析的片段: {e}")
                                continue
                    
                    print(f"    ✓ 片段 {i} 完成，本批次獲得 {segments_added} 個片段，累計 {len(all_segments)} 個")
                
                finally:
                    try:
                        os.unlink(chunk_info['file'])
                    except:
                        pass
            
            segments = all_segments
            print(f"  ✓ 所有片段處理完成，共 {len(segments)} 個語音片段")
        
        else:
            print(f"  - 上傳音訊到 Whisper API (大小: {file_size_mb:.1f}MB)...")
            
            with open(audio_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="zh",
                    response_format="verbose_json",
                    temperature=0.0,
                    prompt=whisper_prompt
                )
            
            print(f"  - API 回傳成功，解析結果中...")
            
            segments = []
            
            if hasattr(transcript, 'segments') and transcript.segments:
                print(f"  - 找到 {len(transcript.segments)} 個片段")
                for seg in transcript.segments:
                    try:
                        if hasattr(seg, 'start'):
                            start = seg.start
                            end = seg.end
                            text = seg.text
                        elif isinstance(seg, 'dict'):
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
        
        if not segments:
            raise ValueError("無法從 API 回應中提取任何內容")
        
        # 品質檢查
        print(f"  - 檢查轉錄品質...")
        quality_score = check_transcription_quality(segments)
        print(f"  → 轉錄品質評分: {quality_score:.1%}")
        
        hallucination_ratio = sum(1 for seg in segments if detect_hallucination(seg)[0]) / len(segments) if segments else 0
        print(f"  → 幻覺比率: {hallucination_ratio:.1%}")
        
        if quality_score < 0.5:
            print(f"  ⚠ 警告: 轉錄品質較低，可能需要重新處理或調整參數")
        
        print(f"  - 檢查重複片段和幻覺...")
        segments = remove_repetitive_segments(segments, similarity_threshold=0.90, consecutive_limit=2)
        
        print(f"  ✓ 成功獲得 {len(segments)} 個語音片段")
        
        return segments
    
    except Exception as e:
        print(f"  ✗ Whisper API 錯誤: {e}")
        import traceback
        traceback.print_exc()
        raise


def check_transcription_quality(segments):
    """
    檢查轉錄品質
    返回 0-1 的品質分數
    """
    if not segments:
        return 0.0
    
    quality_indicators = {
        'has_content': 0,
        'no_repetition': 0,
        'reasonable_length': 0,
        'has_punctuation': 0
    }
    
    # 檢查是否有實際內容
    total_text = "".join([seg['text'] for seg in segments])
    if len(total_text) > 10:
        quality_indicators['has_content'] = 1
    
    # 檢查重複率
    texts = [seg['text'] for seg in segments]
    unique_ratio = len(set(texts)) / len(texts) if texts else 0
    if unique_ratio > 0.7:
        quality_indicators['no_repetition'] = 1
    
    # 檢查片段長度是否合理
    avg_length = np.mean([len(seg['text']) for seg in segments])
    if 5 < avg_length < 100:
        quality_indicators['reasonable_length'] = 1
    
    # 檢查是否有標點符號（表示自然語音）
    punctuation_count = sum(1 for seg in segments if any(p in seg['text'] for p in '，。？！、'))
    if punctuation_count > len(segments) * 0.3:
        quality_indicators['has_punctuation'] = 1
    
    return sum(quality_indicators.values()) / len(quality_indicators)


def transcribe_with_retry(audio_path, max_retries=2):
    """
    帶智慧重試的轉錄（逐步降低處理強度）
    """
    denoise_levels = ['light', 'none']  # 從輕度開始，失敗就不降噪
    
    for attempt, level in enumerate(denoise_levels[:max_retries], 1):
        print(f"\n{'='*50}")
        print(f"嘗試 {attempt}/{max_retries} (降噪模式: {level})")
        print(f"{'='*50}")
        
        # 預處理
        processed_audio, stem = preprocess_audio_minimal(audio_path, denoise_level=level)
        
        # 轉錄
        try:
            segments = transcribe_with_whisper_api(processed_audio)
        except Exception as e:
            print(f"  ✗ 轉錄失敗: {e}")
            if attempt < max_retries:
                print(f"  → 將嘗試更輕的處理模式...")
                continue
            else:
                raise
        
        # 質量檢查
        if not segments:
            print(f"  ✗ 沒有獲得任何內容")
            if attempt < max_retries:
                print(f"  → 將嘗試更輕的處理模式...")
                continue
            else:
                return []
        
        # 計算重複率
        texts = [s['text'] for s in segments]
        unique_ratio = len(set(texts)) / len(texts)
        
        print(f"\n  【質量評估】")
        print(f"  → 獲得 {len(segments)} 個片段")
        print(f"  → 唯一性: {unique_ratio:.1%}")
        
        # 如果唯一性 > 70%，認為成功
        if unique_ratio > 0.7:
            print(f"  ✓ 質量合格，使用此結果")
            return segments
        else:
            print(f"  ⚠ 質量不佳（重複率高）")
            if attempt < max_retries:
                print(f"  → 將嘗試更輕的處理模式...")
                continue
            else:
                print(f"  → 已達最大重試次數，返回當前結果")
                return segments
    
    return segments


def speaker_separation_and_correction_with_gpt(segments):
    """
    說話人分離和錯字修正（優化版）
    明確分離兩個任務
    """
    if len(segments) > 150:
        print(f"  - 對話較長({len(segments)}個片段)，將分批處理")
        return process_segments_in_batches(segments)
    
    texts = []
    for i, seg in enumerate(segments):
        start_time = seg['start']
        m, s = divmod(int(start_time), 60)
        texts.append(f"{i + 1}. [{m:02d}:{s:02d}] {seg['text']}")
    
    full_text = "\n".join(texts)
    
    # 獲取錯字修正提示
    correction_prompt = build_correction_prompt()
    
    # 優化的提示詞：任務明確分離
    prompt = f"""你是專業的客服對話分析師。請分析以下保險客服對話逐字稿，完成兩項任務：

**任務1：說話人分離**

判斷每句話的說話人（客服 or 客戶）。

客服的明確特徵：
- 開場白：「XX保險為您服務」、「您好，我是XX」、「很高興為您服務」、「敝姓XX」
- 敬語：「請問」、「麻煩您」、「幫您」、「為您」、「感謝您」
- 詢問資訊：「請問貴姓」、「請提供身分證」、「您的保單號碼是」、「信用卡末四碼」
- 確認回應：「好的，我幫您查詢」、「收到」、「了解」、「沒問題」
- 解釋說明：「這個部分是...」、「根據您的保單...」

判斷規則：
- 第一句通常是客服開場
- 誰說「您」通常是客服
- 如果不是客服，另一說話者必定為客戶
- 看前後文和說話風格

**任務2：錯字修正**

根據以下對照表修正明顯的錯字和同音字錯誤，但不要改變語意。

{correction_prompt}

修正原則：
- **特別注意**：「默斯碼」、「莫斯碼」、「莫四碼」必須修正為「末四碼」
- **特別注意**：「客邦產險」必須修正為「富邦產險」
- 優先使用上述對照表中的正確用語
- 注意繁體中文的使用
- 保持原句語意不變
- 只修正明顯的錯字和同音字錯誤

逐字稿：
{full_text}

請只回傳 JSON 格式：
{{
  "dialogue": [
    {{"role": "客服", "text": "修正後的文字"}},
    {{"role": "客戶", "text": "修正後的文字"}},
    ...
  ]
}}"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=6000,
            temperature=0.1
        )
        
        content = response.choices[0].message.content.strip()
        
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()
        
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group()
        
        result = json.loads(content)
        dialogue_list = result.get("dialogue", [])
        
        if len(dialogue_list) != len(segments):
            print(f"  ⚠ 警告：GPT 返回 {len(dialogue_list)} 句，原始有 {len(segments)} 句")
            while len(dialogue_list) < len(segments):
                dialogue_list.append({"role": "未知", "text": segments[len(dialogue_list)]["text"]})
            dialogue_list = dialogue_list[:len(segments)]
        
        dialogue = []
        for i, seg in enumerate(segments):
            role = dialogue_list[i].get("role", "未知")
            corrected_text = dialogue_list[i].get("text", seg["text"])
            
            if role not in ["客服", "客戶"]:
                role = "未知"
            
            dialogue.append((seg["start"], role, seg["end"], corrected_text))
        
        print(f"  ✓ GPT 說話人分離與錯字修正完成")
        
        # 統計角色分布
        role_counts = {}
        for _, role, _, _ in dialogue:
            role_counts[role] = role_counts.get(role, 0) + 1
        print(f"  → 角色分布: {role_counts}")
        
        return dialogue
    
    except Exception as e:
        print(f"  ✗ GPT 處理失敗: {e}")
        import traceback
        traceback.print_exc()
        return [(seg["start"], "未知", seg["end"], seg["text"]) for seg in segments]


def process_segments_in_batches(segments):
    """分批處理長對話的說話人分離和錯字修正"""
    batch_size = 100
    all_dialogue = []
    
    num_batches = (len(segments) + batch_size - 1) // batch_size
    print(f"  → 將分為 {num_batches} 批次處理")
    
    correction_short = build_correction_prompt()
    
    for batch_idx in range(0, len(segments), batch_size):
        batch_num = batch_idx // batch_size + 1
        batch_segments = segments[batch_idx:batch_idx+batch_size]
        
        print(f"  → 處理第 {batch_num}/{num_batches} 批次 ({len(batch_segments)} 個片段)...")
        
        batch_texts = []
        for j, seg in enumerate(batch_segments):
            start_time = seg['start']
            m, s = divmod(int(start_time), 60)
            batch_texts.append(f"{batch_idx+j+1}. [{m:02d}:{s:02d}] {seg['text']}")
        
        batch_text = "\n".join(batch_texts)
        
        batch_prompt = f"""請標記說話人（客服/客戶）並修正錯字：

客服特徵：開場白、敝姓、敬語、詢問資訊（身分證、末四碼等）
{correction_short[:400]}

**特別注意**：
• 默斯碼/莫斯碼 → 末四碼
• 客邦產險 → 富邦產險

{batch_text}

返回JSON：
{{"dialogue": [{{"role": "客服", "text": "修正後文字"}}, ...]}}"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": batch_prompt}],
                max_tokens=5000,
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.replace("```json", "").replace("```", "").strip()
            
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group()
            
            batch_result = json.loads(content)
            batch_dialogue = batch_result.get("dialogue", [])
            
            for i, seg in enumerate(batch_segments):
                if i < len(batch_dialogue):
                    role = batch_dialogue[i].get("role", "未知")
                    text = batch_dialogue[i].get("text", seg["text"])
                else:
                    role = "未知"
                    text = seg["text"]
                
                if role not in ["客服", "客戶"]:
                    role = "未知"
                
                all_dialogue.append((seg["start"], role, seg["end"], text))
            
            print(f"    ✓ 第 {batch_num} 批次完成")
        
        except Exception as e:
            print(f"    ✗ 第 {batch_num} 批次失敗: {e}")
            for seg in batch_segments:
                all_dialogue.append((seg["start"], "未知", seg["end"], seg["text"]))
    
    return all_dialogue


def analyze_emotion_fast(text, audio_features=None):
    """快速情緒分析"""
    emotion_scores = {
        "生氣/不滿": 0,
        "焦慮/擔心": 0,
        "滿意/開心": 0,
        "疑惑/困惑": 0,
        "平靜/中性": 0
    }
    
    for emotion, keywords in CORE_VOCABULARY["emotion_keywords"].items():
        for keyword in keywords:
            if keyword in text:
                emotion_scores[emotion] += 2
    
    if audio_features:
        if audio_features['avg_energy'] > 0.05:
            emotion_scores["生氣/不滿"] += 1
            emotion_scores["焦慮/擔心"] += 0.5
        
        if audio_features['avg_zcr'] > 0.15:
            emotion_scores["焦慮/擔心"] += 1
    
    if max(emotion_scores.values()) > 0:
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = emotion_scores[dominant_emotion] / sum(emotion_scores.values())
        return dominant_emotion, confidence
    else:
        return "平靜/中性", 0.5


def classify_and_summarize_with_intent(dialogue_text):
    """
    使用 GPT 進行分類、摘要和意圖分類（三合一）
    與前面的錯字修正階段完全分離
    """
    try:
        prompt = f"""請分析以下客服對話，完成三項任務：

1. 摘要：用30-50字總結對話重點，需要點出用的險種如「強制險」、「醫師責任險」等
2. 分類：判斷屬於「傷害健康保險」、「旅平險」、「車險」或「其他」
   - 若有提到「車險」、「強制險」就歸類為車險
   - 若有提到「不便險」、「旅平險」、「旅行平安險」就優先歸類為旅平險
   - 若有提到「傷害健康保險」、「傷害險」就優先歸類為傷害健康保險
   - 其他情況只要不是車險或旅平險或傷害健康保險，就歸類為「其他」
3. 意圖分類：判斷客戶的主要意圖（可多選2-3個）
   - 選項：保單查詢、理賠服務、投保續保、申訴、諮詢問題、變更資料

對話內容：
{dialogue_text[:2000]}

請只回傳JSON格式，不要有任何其他文字：
{{"abstract": "摘要內容", "class": "分類結果", "intents": ["意圖1", "意圖2"]}}"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.1
        )
        
        content = response.choices[0].message.content.strip()
        
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()
        
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group()
        
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            print(f"  [DEBUG] GPT 回傳內容: {content[:200]}")
            raise
        
        class_result = result.get("class", "其他")
        abstract_result = result.get("abstract", "無摘要")
        intents_result = result.get("intents", ["其他"])
        
        valid_classes = ["傷害健康保險", "旅平險", "車險", "其他"]
        if class_result not in valid_classes:
            print(f"  [警告] 分類結果 '{class_result}' 不在預期範圍，改為'其他'")
            class_result = "其他"
        
        return class_result, abstract_result, intents_result
    
    except Exception as e:
        print(f"  ✗ GPT 分類錯誤: {e}")
        return "其他", "分類失敗", ["其他"]


def process_audio_file(audio_file, denoise_level='light', original_name=None):
    """
    處理單個音訊（優化版）
    
    參數:
        denoise_level: 'none', 'light', 'medium'
    """
    if original_name:
        display_name = original_name
        base_name = Path(original_name).stem
    else:
        display_name = audio_file.name
        base_name = audio_file.stem
    
    print(f"\n{'=' * 60}")
    print(f"處理音訊: {display_name}")
    print(f"開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('-' * 60)
    
    total_start = time()
    
    # Step 1: 帶重試的 Whisper 轉錄
    step1_start = time()
    try:
        whisper_segments = transcribe_with_retry(audio_file, max_retries=2)
        step1_time = time() - step1_start
        print(f"\n  ✓ 語音辨識完成: {step1_time:.2f}秒")
    except Exception as e:
        print(f"  ✗ 語音辨識失敗: {e}")
        return None
    
    # Step 2: GPT 說話人分離 + 錯字修正
    step2_start = time()
    dialogue = speaker_separation_and_correction_with_gpt(whisper_segments)
    step2_time = time() - step2_start
    print(f"  ✓ 說話人分離與錯字修正完成: {step2_time:.2f}秒")
    
    # Step 3: 並行處理（分類+摘要+意圖 和 情緒分析）
    step3_start = time()
    
    dialogue_text = "\n".join([f"{speaker}: {text}" for _, speaker, _, text in dialogue])
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_classify = executor.submit(classify_and_summarize_with_intent, dialogue_text)
        
        customer_emotions = []
        customer_dialogues = []
        
        for start, speaker, end, text in dialogue:
            if speaker == "客戶":
                emotion, confidence = analyze_emotion_fast(text, None)
                customer_emotions.append(emotion)
                customer_dialogues.append((start, end, text, emotion, confidence))
        
        problem_type, abstract, intents = future_classify.result()
    
    step3_time = time() - step3_start
    print(f"  ✓ 分類、意圖與情緒分析完成: {step3_time:.2f}秒")
    print(f"  → 分類結果: {problem_type}")
    print(f"  → 檢測到的意圖: {', '.join(intents)}")
    
    # Step 4: 儲存結果
    step4_start = time()
    
    class_folder = CLASS_FOLDERS.get(problem_type, CLASS_FOLDERS["其他"])
    text_output_folder = os.path.join(class_folder, "voice_text")
    emo_output_folder = os.path.join(class_folder, "voice_emo")
    
    os.makedirs(text_output_folder, exist_ok=True)
    os.makedirs(emo_output_folder, exist_ok=True)
    
    # 儲存逐字稿
    text_file = os.path.join(text_output_folder, f"{base_name}.txt")
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(f"音訊: {display_name}\n")
        f.write(f"分類: {problem_type}\n")
        f.write(f"意圖: {', '.join(intents)}\n")
        f.write(f"處理時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"摘要: {abstract}\n")
        f.write("=" * 60 + "\n\n")
        
        for start, speaker, end, text in dialogue:
            m, s = divmod(int(start), 60)
            f.write(f"[{m:02d}:{s:02d}] {speaker}: {text}\n")
    
    # 儲存情緒分析
    emo_file = os.path.join(emo_output_folder, f"{base_name}_emotion.txt")
    with open(emo_file, "w", encoding="utf-8") as f:
        f.write(f"音訊: {display_name}\n")
        f.write(f"分類: {problem_type}\n")
        f.write(f"意圖: {', '.join(intents)}\n")
        f.write(f"處理時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"摘要: {abstract}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("【客戶情緒分析】\n")
        f.write("-" * 60 + "\n")
        
        for start, end, text, emotion, confidence in customer_dialogues:
            m, s = divmod(int(start), 60)
            f.write(f"[{m:02d}:{s:02d}] {text}\n")
            f.write(f"  → 情緒: {emotion} (信心度: {confidence:.1%})\n\n")
        
        if customer_emotions:
            f.write("\n" + "=" * 60 + "\n")
            f.write("【情緒統計摘要】\n")
            f.write("-" * 60 + "\n")
            
            emotion_counter = Counter(customer_emotions)
            total_count = len(customer_emotions)
            
            for emo, count in emotion_counter.most_common():
                percentage = (count / total_count) * 100
                f.write(f"{emo}: {count}次 ({percentage:.1f}%)\n")
            
            dominant_emotion = emotion_counter.most_common(1)[0][0]
            f.write(f"\n主要情緒: {dominant_emotion}\n")
            
            negative_count = sum(1 for e in customer_emotions if e in ["生氣/不滿", "焦慮/擔心"])
            positive_count = sum(1 for e in customer_emotions if e == "滿意/開心")
            negative_ratio = negative_count / total_count
            positive_ratio = positive_count / total_count
            
            f.write("\n【對話品質評估】\n")
            f.write("-" * 60 + "\n")
            
            if negative_ratio > 0.5:
                f.write(f"⚠ 警示: 客戶負面情緒占 {negative_ratio:.1%}，建議重點關注\n")
            elif positive_ratio > 0.3:
                f.write(f"✓ 良好: 客戶正向情緒占 {positive_ratio:.1%}\n")
            else:
                f.write(f"→ 中性: 客戶情緒整體平穩\n")
    
    step4_time = time() - step4_start
    print(f"  ✓ 檔案儲存完成: {step4_time:.2f}秒")
    print(f"  → 儲存路徑: {class_folder}/voice_text/{base_name}.txt")
    
    # 總結
    total_time = time() - total_start
    print('-' * 60)
    print(f"完成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"總處理時間: {total_time:.2f}秒")
    print(f"  - 語音辨識: {step1_time:.2f}秒 ({step1_time / total_time * 100:.1f}%)")
    print(f"  - 說話人分離: {step2_time:.2f}秒 ({step2_time / total_time * 100:.1f}%)")
    print(f"  - 分類+意圖+情緒: {step3_time:.2f}秒 ({step3_time / total_time * 100:.1f}%)")
    print(f"  - 檔案儲存: {step4_time:.2f}秒 ({step4_time / total_time * 100:.1f}%)")
    print(f"已儲存至: {class_folder}/")
    
    return {
        'file': display_name,
        'time': total_time,
        'class': problem_type,
        'abstract': abstract,
        'intents': intents,
        'emotions': dict(Counter(customer_emotions))
    }


# ========== 主程式 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("語音分析系統 - 防幻覺優化版（Whisper + GPT）")
    print("核心改進：極簡音訊處理 + 激進幻覺過濾 + 智慧重試")
    print("=" * 60)
    
    audio_files = list(Path(input_folder).glob("*.wav")) + \
                  list(Path(input_folder).glob("*.mp3")) + \
                  list(Path(input_folder).glob("*.m4a"))
    
    if not audio_files:
        print(f"\n✗ 在 {input_folder} 資料夾中找不到音訊")
        print("支持格式: .wav, .mp3, .m4a")
    else:
        print(f"\n找到 {len(audio_files)} 個音訊")
        print(f"使用 Whisper API + GPT-4o-mini（防幻覺模式）")
        print(f"處理後音訊將儲存至: {processed_folder}/")
        
        # 詢問降噪模式
        print("\n選擇降噪模式:")
        print("1. 不降噪（最快，適合清晰錄音）")
        print("2. 輕度降噪（推薦，平衡速度與品質）")
        print("3. 中度降噪（適用於嘈雜環境，較慢）")
        denoise_option = input("請選擇 (1/2/3，預設 2): ").strip()
        
        denoise_map = {
            '1': 'none',
            '2': 'light',
            '3': 'medium'
        }
        denoise_level = denoise_map.get(denoise_option, 'light')
        
        print(f"\n✓ 已選擇: {denoise_level} 模式")
        print("\n【自動功能說明】")
        print("  ✓ 智慧重試：品質不佳時自動降低處理強度重試")
        print("  ✓ 幻覺過濾：激進檢測並移除重複/異常片段")
        print("  ✓ 極簡prompt：避免誤導Whisper")
        
        print("\n" + "=" * 60)
        
        overall_start = time()
        results = []
        
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}]")
            result = process_audio_file(
                audio_file,
                denoise_level=denoise_level
            )
            if result:
                results.append(result)
        
        overall_time = time() - overall_start
        
        print("\n" + "=" * 60)
        print("處理完成")
        print("=" * 60)
        print(f"\n總處理時間: {overall_time:.2f}秒 ({overall_time / 60:.2f}分鐘)")
        print(f"平均每檔: {overall_time / len(results):.2f}秒" if results else "")
        
        print(f"\n處理摘要:")
        print("-" * 60)
        for r in results:
            print(f"✓ {r['file']}")
            print(f"  時間: {r['time']:.2f}秒 | 分類: {r['class']}")
            print(f"  意圖: {', '.join(r['intents'])}")
            print(f"  摘要: {r['abstract'][:50]}...")
            if r['emotions']:
                top_emotion = max(r['emotions'].items(), key=lambda x: x[1])
                print(f"  主要情緒: {top_emotion[0]} ({top_emotion[1]}次)")
            print()
        
        print("=" * 60)
        print("檔案儲存結構:")
        for class_name, folder in CLASS_FOLDERS.items():
            count = sum(1 for r in results if r['class'] == class_name)
            if count > 0:
                print(f"  {class_name} ({count}個): {folder}/")
        print("=" * 60)
        print(f"\n處理後音訊: {processed_folder}/")
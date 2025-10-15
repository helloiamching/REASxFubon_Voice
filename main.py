# 1. 簡化流程：音檔 -> 280秒拆分 -> Whisper轉錄（無預處理，直接使用原始音訊）
# 2. 移除所有預處理：避免FFmpeg降噪和音量調整導致音訊變成無聲
# 3. 極輕度預處理：只去除連續3次以上完全相同的重複
# 4. 規則檢測幻覺：使用4種規則快速準確識別
# 5. **裁剪前後靜音**：重轉錄前裁剪開頭和結尾的靜音
#    - 例如：[1:00-1:30] 需要重錄，開頭 [1:00-1:05] 是靜音
#    - 結果：裁剪後轉錄 [1:05-1:30]
#    - 中間的靜音保留（可能只是說話停頓）
#    - 有效音訊 < 3秒 → 直接跳過重轉錄，移除該幻覺片段
#    - 避免對靜音段落浪費 API 調用
# 6. 多次重試機制：重轉錄時最多嘗試3次
#    - 策略1: temperature=0.5, 標準prompt
#    - 策略2: temperature=0.6, 高溫度（更隨機）
#    - 策略3: temperature=0.4, 低溫度（更保守）
#    - 自動評估品質，選擇最好的結果
#    - 品質>0.85自動停止重試
# 7. 品質評估系統：連續重複、短詞重複、內部重複三項檢測
# 8. 僅在重錄時增強：只在重新轉錄幻覺段落時進行溫和的音訊增強
# 9. 60秒拆分：重錄長片段時使用60秒
# 10. 最終去重：說話人分離後再檢查一次
# 11. OpenAI 1: 說話人分離和錯字修正
# 12. OpenAI 2: 摘要和分類

import os
import librosa
import numpy as np
from pathlib import Path
from openai import OpenAI
from time import time
from datetime import datetime
import json
import subprocess
import tempfile
from pydub import AudioSegment
from dotenv import load_dotenv

load_dotenv()

# ========== 設定路徑 ==========
input_folder = "recordings"
processed_folder = "recordings_processed"
os.makedirs(processed_folder, exist_ok=True)

MAX_FILE_SIZE_MB = 24
MAX_CHUNK_DURATION = 280  # 280秒 (約4.67分鐘)

CLASS_FOLDERS = {
    "傷害健康保險": "class_disease",
    "旅平險": "class_travel",
    "車險": "class_car",
    "其他": "class_other"
}

for folder in CLASS_FOLDERS.values():
    os.makedirs(os.path.join(folder, "voice_text"), exist_ok=True)

api_key = os.getenv('OPENAI_API_KEY')

# ========== 核心詞彙庫 ==========
CORE_VOCABULARY = {
    "whisper_keywords": [
        "富邦產險", "醫責險",
        "旅平險", "不便險", "強制險", "車險",
        "投保", "續保", "末四碼"
    ],
    
    "correction_mapping": {
        "富邦產險": ["福邦產險", "富邦產線", "富邦產先", "溫哺巴數", "客邦產險"],
        "末四碼": ["默斯碼", "莫斯碼", "莫四碼", "末斯碼", "默四碼"],
        "不便險": ["不便雞", "不便鴨", "不便線"],
        "旅平險": ["旅平線", "旅評險", "旅蘋險"],
        "身分證": ["身份證", "身分証"],
        "信用卡": ["信用卡", "新用卡"],
        "投保": ["頭包", "投報", "投堡"],
        "保費": ["報廢", "保肥", "報費"],
        "理賠": ["裡酬", "理配", "理陪","李培"],
        "醫責險": ["醫雞險", "醫鴨險", "醫責線"],
        "核保": ["何保"],
        "強制險": ["強迫險", "強制線", "強制先","長治險"],
        "續保": ["續報", "續堡", "續包"],
        "保單": ["報單", "保丹", "報丹"],
        "產險": ["產雞", "產線", "產先","傳血"],
        "車險": ["車線", "車先", "計車惜", "汽車惜"],
        "您好": ["您豪", "您號", "您毫"],
        "敝姓": ["敝性", "幣姓", "弊姓"],
        "麻煩": ["麻凡", "麻煩"],
        "謝謝": ["謝謝", "些些"],
        "死亡及失能": ["身部及心能"],
        "警示鍵": ["警示鍵"],
        "家中出發":["加州出發"],
        "權限的": ["全校的"],
        '加值型': ['加值行',"家支型"],
        '調解委員會': ['拆解委員會'],
        '忙線': ['盲線'],
        '臨櫃': ['靈貴']
    }
}


def build_whisper_prompt():
    """構建極簡的 Whisper prompt"""
    prompt = "富邦產險您好,敝姓溫。旅平險、不便險、強制險。請提供身分證和信用卡末四碼。"
    return prompt


def build_correction_prompt():
    """構建 GPT 錯字修正的專用提示詞"""
    mapping = CORE_VOCABULARY["correction_mapping"]
    
    prompt_parts = ["**常見錯字修正對照表**:\n\n"]
    
    common_errors = list(mapping.items())[:30]
    
    for correct, wrongs in common_errors:
        wrong_list = " / ".join(wrongs[:4])
        prompt_parts.append(f"• {wrong_list} → {correct}\n")
    
    prompt_parts.append("\n**特別注意**:\n")
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


# ========== 拆分邏輯 ==========
def split_audio_file(audio_path, num_chunks=2):
    """拆分大型音訊為多個片段"""
    try:
        print(f"  - 音訊過大，正在拆分為 {num_chunks} 個片段...")
        
        audio = AudioSegment.from_file(str(audio_path))
        duration_seconds = len(audio) / 1000
        
        print(f"    總長度: {duration_seconds:.1f} 秒")
        
        chunk_duration = duration_seconds / num_chunks
        print(f"    每段約 {chunk_duration:.1f} 秒")
        
        chunks = []
        for i in range(num_chunks):
            start_ms = int(i * chunk_duration * 1000)
            end_ms = int(min((i + 1) * chunk_duration * 1000, len(audio)))
            
            chunk = audio[start_ms:end_ms]
            
            temp_file = tempfile.NamedTemporaryFile(
                suffix='.wav',
                delete=False,
                dir=tempfile.gettempdir()
            )
            chunk.export(temp_file.name, format='wav')
            
            chunks.append({
                'file': temp_file.name,
                'start_time': i * chunk_duration,
                'end_time': min((i + 1) * chunk_duration, duration_seconds)
            })
            
            print(f"      段 {i + 1}/{num_chunks}: {chunks[-1]['start_time']:.1f}s - {chunks[-1]['end_time']:.1f}s")
        
        return chunks
    
    except Exception as e:
        print(f"  ✗ 音訊拆分失敗: {e}")
        raise


# ========== 極輕度預處理：只去除明顯連續重複 ==========
def light_deduplication(segments):
    """
    極輕度去重：只去除連續3次以上完全相同的片段
    
    保留：
    - 2次重複（可能是正常強調）
    - 高度相似但不完全相同的片段（可能是正常對話）
    
    去除：
    - 連續3次以上完全相同的片段（明顯的幻覺）
    
    參數:
        segments: Whisper 轉錄的片段列表
    
    返回:
        cleaned_segments: 清理後的片段列表
        removed_count: 去除的片段數
    """
    if not segments or len(segments) < 3:
        return segments, 0
    
    print(f"\n  【極輕度預處理】")
    print(f"  → 原始片段數: {len(segments)}")
    
    cleaned_segments = []
    removed_count = 0
    skip_until = -1
    
    i = 0
    while i < len(segments):
        if i <= skip_until:
            i += 1
            continue
        
        current_text = segments[i]['text'].strip()
        
        # 檢查連續相同的片段數量
        consecutive_same = 1
        for j in range(i + 1, min(i + 10, len(segments))):
            if segments[j]['text'].strip() == current_text:
                consecutive_same += 1
            else:
                break
        
        # 如果連續3次以上完全相同，只保留第一次
        if consecutive_same >= 3:
            print(f"  → 發現連續{consecutive_same}次重複: '{current_text[:30]}...'")
            cleaned_segments.append(segments[i])
            removed_count += consecutive_same - 1
            skip_until = i + consecutive_same - 1
        else:
            cleaned_segments.append(segments[i])
        
        i += 1
    
    print(f"  → 清理後片段數: {len(cleaned_segments)}")
    if removed_count > 0:
        print(f"  → 去除明顯重複: {removed_count} 個片段")
    else:
        print(f"  ✓ 未發現明顯連續重複")
    
    return cleaned_segments, removed_count


# ========== Whisper 轉錄 ==========
def transcribe_with_whisper_api(audio_path):
    """使用 Whisper API 進行語音辨識"""
    try:
        if isinstance(audio_path, str):
            audio_path = Path(audio_path)
        
        whisper_prompt = build_whisper_prompt()
        print(f"  - 使用極簡保險詞彙提示({len(whisper_prompt)}字)")
        
        # 檢查音訊長度
        audio = AudioSegment.from_file(str(audio_path))
        duration_seconds = len(audio) / 1000
        
        print(f"  - 音訊長度: {duration_seconds:.1f} 秒")
        
        # 如果超過280秒，拆分處理
        if duration_seconds > MAX_CHUNK_DURATION:
            print(f"  - 檔案大於280秒，拆分為 2 個片段")
            
            chunks = split_audio_file(audio_path, num_chunks=2)
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
                            temperature=0.0,
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
                                continue
                    
                    print(f"    ✓ 片段 {i} 完成，本批次獲得 {segments_added} 個片段")
                
                finally:
                    try:
                        os.unlink(chunk_info['file'])
                    except:
                        pass
            
            segments = all_segments
            print(f"  ✓ 所有片段處理完成，共 {len(segments)} 個語音片段")
        
        else:
            print(f"  - 上傳音訊到 Whisper API...")
            
            with open(audio_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="zh",
                    response_format="verbose_json",
                    temperature=0.0,
                    prompt=whisper_prompt
                )
            
            segments = []
            
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
                        
                        segments.append({
                            "start": float(start),
                            "end": float(end),
                            "text": text.strip()
                        })
                    except Exception as e:
                        continue
        
        if not segments:
            raise ValueError("無法從 API 回應中提取任何內容")
        
        print(f"  ✓ 成功獲得 {len(segments)} 個語音片段")
        
        return segments
    
    except Exception as e:
        print(f"  ✗ Whisper API 錯誤: {e}")
        import traceback
        traceback.print_exc()
        raise


# ========== 規則檢測幻覺（取代OpenAI）==========
def detect_hallucination_with_rules(segments):
    """
    使用規則檢測幻覺（比OpenAI更可靠）
    
    檢測規律：
    1. 連續3次以上完全相同的句子
    2. 短詞（<5字）重複超過5次
    3. 單句內部重複（如"A和B和A和B"）
    4. 時間跨度長但內容相同（如從02:50到05:42都是同一句）
    
    參數:
        segments: Whisper 轉錄的片段列表
    
    返回:
        hallucination_ranges: 幻覺時間段列表
    """
    print(f"\n【規則檢測幻覺時間段】")
    
    if not segments or len(segments) < 3:
        print(f"  ✓ 片段數太少 ({len(segments)}個)，無需檢測")
        return []
    
    print(f"  → 分析 {len(segments)} 個片段...")
    
    hallucination_ranges = []
    
    # 規則1: 檢測連續相同句子（≥3次）
    i = 0
    while i < len(segments) - 2:
        current_text = segments[i]['text'].strip()
        
        # 跳過空白或太短的片段
        if len(current_text) < 3:
            i += 1
            continue
        
        # 計算連續相同的次數
        consecutive_same = 1
        for j in range(i + 1, len(segments)):
            if segments[j]['text'].strip() == current_text:
                consecutive_same += 1
            else:
                break
        
        # 如果連續3次以上
        if consecutive_same >= 3:
            hallucination_ranges.append({
                'start': segments[i]['start'],
                'end': segments[i + consecutive_same - 1]['end'],
                'reason': f'連續重複{consecutive_same}次: "{current_text[:30]}..."',
                'type': 'consecutive_repeat'
            })
            print(f"  ⚠ 發現連續重複 {consecutive_same} 次 ({segments[i]['start']:.1f}s-{segments[i + consecutive_same - 1]['end']:.1f}s)")
            i += consecutive_same
            continue
        
        i += 1
    
    # 規則2: 檢測短詞狂重複（如"您好"重複20次）
    for i, seg in enumerate(segments):
        text = seg['text'].strip()
        
        # 跳過正常長度的文本
        if len(text) > 50:
            continue
        
        # 分析單句內的重複模式
        words = text.replace('，', ',').replace('。', '.').replace(' ', '').split(',')
        
        # 統計最常見的短詞
        word_counts = {}
        for word in words:
            word = word.strip()
            if 1 <= len(word) <= 4:  # 只看1-4字的短詞
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # 如果某個短詞重複超過5次
        for word, count in word_counts.items():
            if count >= 5:
                hallucination_ranges.append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'reason': f'短詞"{word}"重複{count}次',
                    'type': 'short_word_repeat'
                })
                print(f"  ⚠ 發現短詞狂重複: '{word}' 重複 {count} 次 ({seg['start']:.1f}s)")
                break
    
    # 規則3: 檢測單句內部重複（如"A和B和A和B"）
    for i, seg in enumerate(segments):
        text = seg['text'].strip()
        
        # 跳過太短的文本
        if len(text) < 10:
            continue
        
        # 檢測內部重複模式
        has_internal_repeat = False
        
        # 檢測完全重複的短語（如"身分證和信用卡末四碼"出現2次）
        for phrase_len in range(5, min(len(text) // 2, 30)):
            phrase = text[:phrase_len]
            if text.count(phrase) >= 2:
                has_internal_repeat = True
                hallucination_ranges.append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'reason': f'單句內部重複: "{phrase[:20]}..." 出現{text.count(phrase)}次',
                    'type': 'internal_repeat'
                })
                print(f"  ⚠ 發現單句內部重複 ({seg['start']:.1f}s): '{phrase[:20]}...'")
                break
        
        if has_internal_repeat:
            continue
    
    # 規則4: 檢測時間跨度長的相同內容（如02:50-05:42都是同一句）
    seen_texts = {}
    for i, seg in enumerate(segments):
        text = seg['text'].strip()
        
        # 跳過太短的文本
        if len(text) < 5:
            continue
        
        if text in seen_texts:
            # 計算時間跨度
            first_occurrence = seen_texts[text]
            time_span = seg['start'] - first_occurrence['start']
            
            # 如果同一句話跨越超過30秒
            if time_span > 30:
                # 找出所有相同的片段
                same_segments = [first_occurrence]
                for j, s in enumerate(segments):
                    if s['text'].strip() == text and s['start'] > first_occurrence['start']:
                        same_segments.append(s)
                
                if len(same_segments) >= 3:  # 至少3次
                    hallucination_ranges.append({
                        'start': same_segments[0]['start'],
                        'end': same_segments[-1]['end'],
                        'reason': f'長時間重複: "{text[:30]}..." (時間跨度{time_span:.0f}秒)',
                        'type': 'long_span_repeat'
                    })
                    print(f"  ⚠ 發現長時間重複 ({same_segments[0]['start']:.1f}s-{same_segments[-1]['end']:.1f}s): '{text[:30]}...'")
        else:
            seen_texts[text] = seg
    
    # 合併重疊的時間段
    if hallucination_ranges:
        hallucination_ranges = merge_hallucination_ranges(hallucination_ranges)
        print(f"\n  ✓ 合併後發現 {len(hallucination_ranges)} 個幻覺時間段")
    else:
        print(f"  ✓ 未發現明顯幻覺")
    
    return hallucination_ranges

from pydub import AudioSegment, silence

def detect_and_trim_silence(audio_segment, silence_thresh=-50, min_silence_len=500):
    """
    偵測音訊開頭與結尾的靜音，並回傳裁剪後的音訊。
    會同時回傳：
    - processed_audio: 去除開頭結尾靜音的音訊（若無靜音則原樣）
    - valid_duration: 有效音訊長度（秒）
    - has_silence: 是否有靜音被裁剪
    - sound_start, sound_end: 有效音訊的起訖時間（秒）
    """
    from pydub.silence import detect_nonsilent

    duration_ms = len(audio_segment)
    non_silence = detect_nonsilent(
        audio_segment, min_silence_len=min_silence_len, silence_thresh=silence_thresh
    )

    if not non_silence:
        # 全靜音
        return None, 0.0, False, 0, 0

    # 取最前與最後的非靜音區
    sound_start = non_silence[0][0] / 1000.0
    sound_end = non_silence[-1][1] / 1000.0
    valid_duration = sound_end - sound_start

    if valid_duration <= 0:
        return None, 0.0, False, sound_start, sound_end

    # 裁剪開頭與結尾的靜音
    processed_audio = audio_segment[int(sound_start * 1000):int(sound_end * 1000)]
    has_silence = (sound_start > 0.2 or (duration_ms / 1000.0 - sound_end) > 0.2)

    return processed_audio, valid_duration, has_silence, sound_start, sound_end




def merge_hallucination_ranges(ranges):
    """合併重疊或接近的幻覺時間段"""
    if not ranges:
        return []
    
    # 按開始時間排序
    sorted_ranges = sorted(ranges, key=lambda x: x['start'])
    
    merged = [sorted_ranges[0]]
    
    for current in sorted_ranges[1:]:
        last = merged[-1]
        
        # 如果時間重疊或接近（10秒內）
        if current['start'] <= last['end'] + 10:
            # 合併
            last['end'] = max(last['end'], current['end'])
            last['reason'] = f"{last['reason']} & {current['reason']}"
        else:
            merged.append(current)
    
    return merged


# ========== 音訊增強處理（溫和版）==========
def enhance_audio_for_retranscription(audio_segment):
    """
    對音訊進行增強處理，專門用於重新轉錄（溫和版）
    
    改進：降低激進程度，添加錯誤處理和降級方案
    
    處理包括：
    - 溫和的降噪
    - 語音增強
    - 正規化音量
    
    參數:
        audio_segment: pydub AudioSegment
    
    返回:
        enhanced_path: 增強後的音訊臨時文件路徑
    """
    try:
        # 保存為臨時文件
        temp_input = tempfile.NamedTemporaryFile(
            suffix='.wav',
            delete=False,
            dir=tempfile.gettempdir()
        )
        audio_segment.export(temp_input.name, format='wav')
        
        # 使用FFmpeg進行增強處理
        temp_output = tempfile.NamedTemporaryFile(
            suffix='.wav',
            delete=False,
            dir=tempfile.gettempdir()
        )
        
        subprocess.run(
            [
                'ffmpeg', '-i', temp_input.name,
                '-af',
                # 溫和的增強參數
                'highpass=f=100,'           # 高通濾波
                'lowpass=f=7000,'           # 低通濾波
                'afftdn=nf=-25,'            # 降噪（溫和）
                'speechnorm=e=12.5:r=0.0001:l=1,'  # 語音正規化
                'loudnorm=I=-16:TP=-1.5',   # 音量正規化
                '-ar', '16000',
                '-ac', '1',
                '-acodec', 'pcm_s16le',
                '-y', temp_output.name
            ],
            capture_output=True,
            check=True,
            timeout=30  # 添加超時限制
        )
        
        # 清理輸入文件
        os.unlink(temp_input.name)
        
        # 檢查輸出是否正常
        try:
            check_audio = AudioSegment.from_file(temp_output.name)
            if check_audio.dBFS < -60:  # 音量太小
                print(f"      ⚠ 增強後音量過小，使用原始音訊")
                os.unlink(temp_output.name)
                # 返回原始音訊
                temp_fallback = tempfile.NamedTemporaryFile(
                    suffix='.wav',
                    delete=False,
                    dir=tempfile.gettempdir()
                )
                audio_segment.export(temp_fallback.name, format='wav')
                return temp_fallback.name
        except:
            pass
        
        return temp_output.name
    
    except Exception as e:
        print(f"      ⚠ 音訊增強失敗，使用原始音訊: {e}")
        # 如果增強失敗，返回原始音訊
        temp_fallback = tempfile.NamedTemporaryFile(
            suffix='.wav',
            delete=False,
            dir=tempfile.gettempdir()
        )
        audio_segment.export(temp_fallback.name, format='wav')
        
        # 清理可能殘留的文件
        try:
            os.unlink(temp_input.name)
        except:
            pass
        try:
            os.unlink(temp_output.name)
        except:
            pass
        
        return temp_fallback.name


# ========== 檢測並跳過靜默段落 ==========
def detect_and_skip_silence(audio_segment, min_silence_len=1000, silence_thresh=-40):
    """
    檢測音訊開頭和結尾的靜默段落，返回有效音訊的起始和結束位置
    
    注意：只修剪開頭和結尾的靜音，中間的靜音保留（可能是說話停頓）
    
    參數:
        audio_segment: pydub AudioSegment
        min_silence_len: 靜默的最小長度（毫秒）
        silence_thresh: 靜默閾值（dBFS）
    
    返回:
        start_sec: 有效音訊的起始位置（秒）
        end_sec: 有效音訊的結束位置（秒）
        has_silence: 是否檢測到前後靜音
    """
    try:
        from pydub.silence import detect_nonsilent
        
        # 檢測非靜默段落
        nonsilent_ranges = detect_nonsilent(
            audio_segment,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            seek_step=100  # 檢測步長（毫秒）
        )
        
        if not nonsilent_ranges:
            # 整段都是靜默
            return 0, len(audio_segment) / 1000, False
        
        # 獲取第一個有聲音的位置（開頭）
        first_sound_ms = nonsilent_ranges[0][0]
        # 獲取最後一個有聲音的位置（結尾）
        last_sound_ms = nonsilent_ranges[-1][1]
        
        # 轉換為秒
        start_sec = first_sound_ms / 1000
        end_sec = last_sound_ms / 1000
        
        # 判斷是否有明顯的前後靜音（超過0.5秒）
        has_silence = (first_sound_ms > 500) or (len(audio_segment) - last_sound_ms > 500)
        
        return start_sec, end_sec, has_silence
    
    except Exception as e:
        # 如果檢測失敗，返回全段
        print(f"      ⚠ 靜默檢測失敗: {e}，使用全段音訊")
        total_sec = len(audio_segment) / 1000
        return 0, total_sec, False

import numpy as np
import librosa

def remove_customer_music(segment_audio, sample_rate=16000):
    """
    嘗試檢測並移除客服音樂（例如有旋律或歌詞的背景音）
    原理：
      - 將音訊轉為 Mel 頻譜
      - 偵測是否出現穩定、寬頻能量的「旋律樣式」
      - 若該區段 RMS 高且過於穩定，則視為音樂並裁剪
    """
    try:
        # 轉為 numpy array
        samples = np.array(segment_audio.get_array_of_samples()).astype(np.float32)
        if segment_audio.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)  # 轉單聲道
        
        # 重採樣（保證一致）
        y = librosa.resample(samples, orig_sr=segment_audio.frame_rate, target_sr=sample_rate)
        S = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_mels=64)
        rms = librosa.feature.rms(S=S)[0]
        
        # 若平均 RMS 太低（整段太安靜），不處理
        if np.mean(rms) < 0.005:
            print(f"    ✓ 平均能量過低，判定無明顯音樂")
            return segment_audio
        
        # 找出音樂段落：RMS 穩定且能量高
        diff = np.abs(np.diff(rms))
        music_mask = (rms > np.mean(rms) * 1.2) & (diff < np.mean(diff) * 0.3)
        music_ratio = np.mean(music_mask)

        # --- 情況 1：整段高比例音樂 ---
        if music_ratio > 0.4:
            print(f"    ⚠ 偵測到高比例客服音樂（{music_ratio*100:.1f}%），嘗試移除前半部分...")
            cut_ms = len(segment_audio) * 0.4
            segment_audio = segment_audio[int(cut_ms):]

        # --- 情況 2：短段音樂片段 ---
        elif 0.05 < music_ratio <= 0.4:
            music_indices = np.where(music_mask)[0]
            start_idx = librosa.frames_to_samples(music_indices[0])
            end_idx = librosa.frames_to_samples(music_indices[-1])
            start_ms = start_idx / sample_rate * 1000
            end_ms = end_idx / sample_rate * 1000
            print(f"    ⚠ 偵測到客服音樂片段 {start_ms/1000:.1f}s–{end_ms/1000:.1f}s，已裁剪")
            segment_audio = segment_audio[:int(start_ms)] + segment_audio[int(end_ms):]
        
        # --- 情況 3：未發現音樂 ---
        else:
            print(f"    ✓ 前後無明顯音樂，使用完整片段")
    
    except Exception as e:
        print(f"    ⚠ 音樂檢測失敗: {e}")
    
    return segment_audio


# ========== 評估轉錄品質 ==========
def evaluate_transcription_quality(segments):
    """
    評估轉錄品質
    
    檢查項目：
    1. 連續重複片段數
    2. 短詞重複嚴重程度
    3. 單句內部重複
    
    返回:
        quality_score: 品質分數 (0-1)，越高越好
        issues: 發現的問題列表
    """
    if not segments:
        return 0.0, ["無片段"]
    
    issues = []
    penalty = 0.0
    
    # 檢查1: 連續重複
    consecutive_repeat_count = 0
    i = 0
    while i < len(segments) - 1:
        current_text = segments[i]['text'].strip()
        consecutive_same = 1
        
        for j in range(i + 1, min(i + 6, len(segments))):
            if segments[j]['text'].strip() == current_text:
                consecutive_same += 1
            else:
                break
        
        if consecutive_same >= 3:
            consecutive_repeat_count += 1
            i += consecutive_same
        else:
            i += 1
    
    if consecutive_repeat_count > 0:
        issues.append(f"連續重複{consecutive_repeat_count}組")
        penalty += consecutive_repeat_count * 0.2
    
    # 檢查2: 短詞狂重複
    short_word_issues = 0
    for seg in segments:
        text = seg['text'].strip()
        if len(text) > 50:
            continue
        
        words = text.replace('，', ',').replace('。', '.').split(',')
        for word in words:
            word = word.strip()
            if 1 <= len(word) <= 4:
                count = text.count(word)
                if count >= 5:
                    short_word_issues += 1
                    break
    
    if short_word_issues > 0:
        issues.append(f"{short_word_issues}個短詞狂重複")
        penalty += short_word_issues * 0.15
    
    # 檢查3: 單句內部重複
    internal_repeat_count = 0
    for seg in segments:
        text = seg['text'].strip()
        if len(text) < 10:
            continue
        
        for phrase_len in range(5, min(len(text) // 2, 30)):
            phrase = text[:phrase_len]
            if text.count(phrase) >= 2:
                internal_repeat_count += 1
                break
    
    if internal_repeat_count > 0:
        issues.append(f"{internal_repeat_count}個內部重複")
        penalty += internal_repeat_count * 0.1
    
    # 計算品質分數
    quality_score = max(0.0, 1.0 - penalty)
    
    return quality_score, issues


# ========== 對幻覺段落重新轉錄（多次重試版）==========
def retranscribe_hallucination_segments(audio_path, hallucination_ranges, original_segments):
    """
    對檢測到的幻覺時間段重新用 Whisper 轉錄（裁剪前後靜音版）
    
    改進：
    1. **裁剪前後靜音**：只去除開頭和結尾的靜音，中間的靜音保留
       - 例如：[1:00-1:30] 需要重錄，開頭 [1:00-1:05] 是靜音
       - 結果：裁剪後轉錄 [1:05-1:30]
       - 中間的靜音不處理，因為可能只是說話停頓
    2. 最多重試3次
    3. 每次改變temperature和prompt
    4. 評估每次結果的品質
    5. 選擇最好的結果
    6. 如果片段>90秒，拆分成60秒片段
    
    靜音處理邏輯：
    - 有效音訊 < 3秒 → 判定為整段靜音，直接移除，不重轉錄
    - 有效音訊 ≥ 3秒 → 裁剪前後靜音後進行重轉錄
    - 無前後靜音 → 對完整片段進行重轉錄
    
    參數:
        audio_path: 音訊路徑
        hallucination_ranges: 幻覺時間段列表
        original_segments: 原始片段列表
    
    返回:
        updated_segments: 更新後的片段列表
    """
    if not hallucination_ranges:
        return original_segments
    
    print(f"\n【重新轉錄幻覺段落（多次重試版）】")
    print(f"  → 需要重新轉錄 {len(hallucination_ranges)} 個時間段")
    
    # 將原始片段轉換為字典，方便查找和替換
    segments_dict = {i: seg for i, seg in enumerate(original_segments)}
    
    # 不同的重試策略
    retry_strategies = [
        {
            'temperature': 0,
            'prompt': "富邦產險客服。",
            'name': '策略0：無溫度'
        },
        {
            'temperature': 0.5,
            'prompt': "富邦產險客服。",
            'name': '策略1：標準'
        },
        {
            'temperature': 0.6,
            'prompt': "富邦產險客服。",
            'name': '策略2：高溫度'
        },
        {
            'temperature': 0.4,
            'prompt': "富邦產險客服。",
            'name': '策略3：低溫度'
        }
    ]
    
    for idx, hr in enumerate(hallucination_ranges, 1):
        start_time = hr['start']
        end_time = hr['end']
        
        print(f"\n  [{idx}/{len(hallucination_ranges)}] 重新轉錄 {start_time:.1f}s - {end_time:.1f}s")
        print(f"    原因: {hr['reason']}")
        
        try:
            # 提取該時間段的音訊
            audio = AudioSegment.from_file(str(audio_path))
            
            # 擴展時間範圍（前後各加5秒緩衝）
            buffer = 5
            extract_start = max(0, start_time - buffer)
            extract_end = min(len(audio) / 1000, end_time + buffer)
            
            print(f"    → 實際重錄範圍: {extract_start:.1f}s - {extract_end:.1f}s")
            
            segment_audio = audio[int(extract_start * 1000):int(extract_end * 1000)]
            segment_duration = (extract_end - extract_start)
            
            print(f"    → 片段長度: {segment_duration:.1f}秒")
            
            # ========== 檢測並裁剪開頭和結尾的靜音 ==========
            print(f"    → 檢測靜音並裁剪...")
            processed_audio, valid_duration, has_silence, sound_start, sound_end = detect_and_trim_silence(segment_audio)

            if processed_audio is None or valid_duration < 3:
                print(f"    ⚠ 有效語音太短（{valid_duration:.1f} 秒），略過重錄")
                continue

            print(f"    → 裁剪後長度：{valid_duration:.1f} 秒")
            segment_audio = processed_audio
            segment_duration = valid_duration
            actual_extract_start = extract_start  # 不需更新時間偏移
            actual_extract_end = extract_start + valid_duration

            _, valid_duration = detect_and_trim_silence(segment_audio)
            
            # 如果有效音訊太短（<3秒），可能整段都是靜音
            if valid_duration < 3:
                print(f"    ⚠ 有效音訊太短（{valid_duration:.1f}秒 < 3秒），判定為靜音")
                print(f"       → 直接移除該幻覺段落，不進行重轉錄")
                
                # 移除這些幻覺片段
                indices_to_remove = []
                for i, seg in enumerate(original_segments):
                    if start_time <= seg['start'] <= end_time:
                        indices_to_remove.append(i)
                
                if indices_to_remove:
                    print(f"    → 移除原始片段 {indices_to_remove[0]+1}-{indices_to_remove[-1]+1}")
                    for idx_remove in indices_to_remove:
                        if idx_remove in segments_dict:
                            del segments_dict[idx_remove]
                
                continue
            
            # 裁剪掉前後的靜音（中間的靜音保留）
            if has_silence:
                print(f"    → 裁剪靜音：開頭 {sound_start:.1f}秒 + 結尾 {segment_duration - sound_end:.1f}秒")
                
                # 裁剪掉靜音部分
                segment_audio = segment_audio[int(sound_start * 1000):int(sound_end * 1000)]
                
                # 更新實際的起始和結束時間
                actual_extract_start = extract_start + sound_start
                actual_extract_end = extract_start + sound_end
                segment_duration = sound_end - sound_start
                
                print(f"    → 裁剪後音訊長度: {segment_duration:.1f}秒")
            else:
                print(f"    ✓ 前後無明顯靜音，使用完整片段")
                actual_extract_start = extract_start
                actual_extract_end = extract_end

                        # ========== 檢測並去除客服音樂 ==========
            print(f"    → 嘗試去除客服音樂...")
            segment_audio = remove_customer_music(segment_audio)
            
            # 多次重試
            best_segments = None
            best_quality = 0.0
            best_strategy = None
            
            for attempt, strategy in enumerate(retry_strategies, 1):
                print(f"\n    【嘗試 {attempt}/3】{strategy['name']}")
                
                try:
                    new_segments = []
                    
                    # 如果片段過長（超過90秒），拆分處理
                    if segment_duration > 90:
                        print(f"      → 拆分成60秒片段處理...")
                        
                        chunk_duration = 60
                        num_sub_chunks = int(np.ceil(segment_duration / chunk_duration))
                        
                        for sub_i in range(num_sub_chunks):
                            sub_start_ms = int(sub_i * chunk_duration * 1000)
                            sub_end_ms = int(min((sub_i + 1) * chunk_duration * 1000, len(segment_audio)))
                            sub_audio = segment_audio[sub_start_ms:sub_end_ms]
                            
                            # 音訊增強（溫和版）
                            enhanced_path = enhance_audio_for_retranscription(sub_audio)
                            
                            try:
                                with open(enhanced_path, "rb") as audio_file:
                                    transcript = client.audio.transcriptions.create(
                                        model="whisper-1",
                                        file=audio_file,
                                        language="zh",
                                        response_format="verbose_json",
                                        temperature=strategy['temperature'],
                                        prompt=strategy['prompt']
                                    )
                                
                                if hasattr(transcript, 'segments') and transcript.segments:
                                    for seg in transcript.segments:
                                        try:
                                            if hasattr(seg, 'start'):
                                                seg_start = seg.start
                                                seg_end = seg.end
                                                seg_text = seg.text
                                            elif isinstance(seg, dict):
                                                seg_start = seg['start']
                                                seg_end = seg['end']
                                                seg_text = seg['text']
                                            else:
                                                continue
                                            
                                            # 計算在原始音訊中的時間
                                            actual_start = actual_extract_start + (sub_i * chunk_duration) + float(seg_start)
                                            actual_end = actual_extract_start + (sub_i * chunk_duration) + float(seg_end)
                                            
                                            new_segments.append({
                                                "start": actual_start,
                                                "end": actual_end,
                                                "text": seg_text.strip()
                                            })
                                        except Exception as e:
                                            continue
                            finally:
                                try:
                                    os.unlink(enhanced_path)
                                except:
                                    pass
                    
                    else:
                        # 片段不長（≤90秒），直接處理
                        enhanced_path = enhance_audio_for_retranscription(segment_audio)
                        
                        try:
                            with open(enhanced_path, "rb") as audio_file:
                                transcript = client.audio.transcriptions.create(
                                    model="whisper-1",
                                    file=audio_file,
                                    language="zh",
                                    response_format="verbose_json",
                                    temperature=strategy['temperature'],
                                    prompt=strategy['prompt']
                                )
                            
                            if hasattr(transcript, 'segments') and transcript.segments:
                                for seg in transcript.segments:
                                    try:
                                        if hasattr(seg, 'start'):
                                            seg_start = seg.start
                                            seg_end = seg.end
                                            seg_text = seg.text
                                        elif isinstance(seg, dict):
                                            seg_start = seg['start']
                                            seg_end = seg['end']
                                            seg_text = seg['text']
                                        else:
                                            continue
                                        
                                        # 計算在原始音訊中的時間
                                        actual_start = actual_extract_start + float(seg_start)
                                        actual_end = actual_extract_start + float(seg_end)
                                        
                                        new_segments.append({
                                            "start": actual_start,
                                            "end": actual_end,
                                            "text": seg_text.strip()
                                        })
                                    except Exception as e:
                                        continue
                        finally:
                            try:
                                os.unlink(enhanced_path)
                            except:
                                pass
                    
                    # 評估品質
                    if new_segments:
                        quality_score, issues = evaluate_transcription_quality(new_segments)
                        print(f"      → 獲得 {len(new_segments)} 個片段，品質分數: {quality_score:.2f}")
                        
                        if issues:
                            print(f"      → 問題: {', '.join(issues)}")
                        
                        # 如果品質完美或比之前的好，保存
                        if quality_score > best_quality:
                            best_segments = new_segments
                            best_quality = quality_score
                            best_strategy = strategy['name']
                            
                            # 如果品質已經很好（>0.85），不需要繼續嘗試
                            if quality_score > 0.85:
                                print(f"      ✓ 品質良好，停止重試")
                                break
                    else:
                        print(f"      ✗ 未獲得任何片段")
                
                except Exception as e:
                    print(f"      ✗ 嘗試 {attempt} 失敗: {e}")
                    continue
            
            # 使用最好的結果
            if best_segments:
                print(f"\n    ✓ 最佳結果：{best_strategy}（品質: {best_quality:.2f}）")
                
                # 如果品質還是不夠好，使用去重
                if best_quality < 0.7:
                    print(f"    → 品質仍不理想，進行去重...")
                    best_segments, removed = light_deduplication(best_segments)
                    print(f"    → 去重後保留 {len(best_segments)} 個片段")
                
                # 找出需要替換的原始片段索引
                indices_to_remove = []
                for i, seg in enumerate(original_segments):
                    if start_time <= seg['start'] <= end_time:
                        indices_to_remove.append(i)
                
                # 替換片段
                if indices_to_remove:
                    print(f"    → 替換原始片段 {indices_to_remove[0]+1}-{indices_to_remove[-1]+1}")
                    for idx_remove in indices_to_remove:
                        if idx_remove in segments_dict:
                            del segments_dict[idx_remove]
                    
                    insert_pos = indices_to_remove[0]
                    for new_seg in best_segments:
                        segments_dict[insert_pos] = new_seg
                        insert_pos += 0.1
            else:
                print(f"    ✗ 所有嘗試都失敗，保留原始片段")
        
        except Exception as e:
            print(f"    ✗ 重新轉錄失敗: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 重新組織片段列表並按時間排序
    updated_segments = sorted(segments_dict.values(), key=lambda x: x['start'])
    
    print(f"\n  ✓ 更新完成，最終片段數: {len(updated_segments)}")
    
    return updated_segments


# ========== 最終去重（說話人分離後）==========
def final_deduplication_after_speaker(dialogue):
    """
    在說話人分離後進行最終去重
    
    為什麼需要：
    - 重新轉錄後可能產生新的重複
    - 說話人標記過程不會移除重複
    - 確保最終輸出沒有明顯的連續重複
    
    參數:
        dialogue: [(start, speaker, end, text), ...]
    
    返回:
        cleaned_dialogue: 清理後的對話
        removed_count: 去除的片段數
    """
    if not dialogue or len(dialogue) < 3:
        return dialogue, 0
    
    print(f"\n  【最終去重】")
    print(f"  → 原始片段數: {len(dialogue)}")
    
    cleaned_dialogue = []
    removed_count = 0
    skip_until = -1
    
    i = 0
    while i < len(dialogue):
        if i <= skip_until:
            i += 1
            continue
        
        current_text = dialogue[i][3].strip()  # text
        
        # 檢查連續相同的片段數量
        consecutive_same = 1
        for j in range(i + 1, min(i + 10, len(dialogue))):
            if dialogue[j][3].strip() == current_text:
                consecutive_same += 1
            else:
                break
        
        # 如果連續3次以上完全相同，只保留第一次
        if consecutive_same >= 3:
            print(f"  → 去除連續{consecutive_same}次重複: '{current_text[:30]}...'")
            cleaned_dialogue.append(dialogue[i])
            removed_count += consecutive_same - 1
            skip_until = i + consecutive_same - 1
        else:
            cleaned_dialogue.append(dialogue[i])
        
        i += 1
    
    print(f"  → 清理後片段數: {len(cleaned_dialogue)}")
    if removed_count > 0:
        print(f"  → 去除重複片段: {removed_count} 個")
    else:
        print(f"  ✓ 無需去重")
    
    return cleaned_dialogue, removed_count


def speaker_separation_and_correction_with_gpt(segments):
    """說話人分離和錯字修正"""
    if len(segments) > 150:
        print(f"  - 對話較長({len(segments)}個片段)，將分批處理")
        return process_segments_in_batches(segments)
    
    texts = []
    for i, seg in enumerate(segments):
        start_time = seg['start']
        m, s = divmod(int(start_time), 60)
        texts.append(f"{i + 1}. [{m:02d}:{s:02d}] {seg['text']}")
    
    full_text = "\n".join(texts)
    
    correction_prompt = build_correction_prompt()
    prompt = f"""
        你是專業的客服對話分析師。請分析以下保險客服對話逐字稿，完成兩項任務：

        ---

        ### 任務1：說話人分離
        判斷每句話的說話人（客服 or 客戶）。

        客服的明確特徵：
        - 開場白：「XX保險為您服務」、「您好，我是XX」、「很高興為您服務」、「敝姓XX」
        - 敬語：「請問」、「麻煩您」、「幫您」、「為您」、「感謝您」
        - 詢問資訊：「請問貴姓」、「請提供身分證」、「您的保單號碼是」、「信用卡末四碼」
        - 確認回應：「好的，我幫您查詢」、「收到」、「了解」、「沒問題」
        - 解釋說明：「這個部分是...」、「根據您的保單...」

        判斷要則：
        - 第一句通常是客服開場
        - 誰說「您」通常是客服
        - 如果不是客服，另一說話者必定為客戶
        - 看前後文和說話風格

        ---

        ### 任務2：錯字修正
        根據以下對照表修正明顯的錯字和同音字錯誤，但不要改變語意。

        {correction_prompt}

        修正原則：
        - 「默斯碼」「莫斯碼」「莫四碼」 → 修正為「末四碼」
        - 「客邦產險」 → 修正為「富邦產險」
        - 優先使用對照表中的正確用語
        - 使用繁體中文
        - 保持語意不變

        ---

        ### 對話逐字稿：
        {full_text}

        ---

        ### 輸出格式要求（務必嚴格遵守）：
        只回傳**合法 JSON**，不要包含任何說明或額外文字。  
        JSON 必須符合以下格式：

        ```json
        {{
        "dialogue": [
            {{"role": "客服", "text": "修正後的文字"}},
            {{"role": "客戶", "text": "修正後的文字"}}
        ]
        }}"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
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

**特別注意**:
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


# ========== OpenAI: 分類和摘要（單選意圖）==========
def classify_and_summarize_single_intent(dialogue_text):
    """使用 GPT 進行分類、摘要和意圖分類（單選）"""
    try:
        prompt = f"""請分析以下客服對話，完成三項任務：

1. 摘要：用30-50字總結對話重點，需要點出用的險種如「強制險」、「醫師責任險」等
2. 分類：判斷屬於「傷害健康保險」、「旅平險」、「車險」或「其他」
   - 若有提到「車險」、「強制險」就歸類為車險
   - 若有提到「不便險」、「旅平險」、「旅行平安險」就優先歸類為旅平險
   - 若有提到「傷害健康保險」、「傷害險」就優先歸類為傷害健康保險
   - 其他情況只要不是車險或旅平險或傷害健康保險，就歸類為「其他」
3. 意圖分類：判斷客戶的主要意圖，**只能選1個**
   - 選項：保單查詢、理賠服務、投保續保、申訴、諮詢問題、變更資料

對話內容：
{dialogue_text[:2000]}

請只回傳JSON格式，不要有任何其他文字：
{{"abstract": "摘要內容", "class": "分類結果", "intent": "意圖"}}"""
        
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
        intent_result = result.get("intent", "其他")
        
        valid_classes = ["傷害健康保險", "旅平險", "車險", "其他"]
        if class_result not in valid_classes:
            print(f"  [警告] 分類結果 '{class_result}' 不在預期範圍，改為'其他'")
            class_result = "其他"
        
        valid_intents = ["保單查詢", "理賠服務", "投保續保", "申訴", "諮詢問題", "變更資料"]
        if intent_result not in valid_intents:
            print(f"  [警告] 意圖 '{intent_result}' 不在預期範圍，改為'諮詢問題'")
            intent_result = "諮詢問題"
        
        return class_result, abstract_result, intent_result
    
    except Exception as e:
        print(f"  ✗ GPT 分類錯誤: {e}")
        return "其他", "分類失敗", "諮詢問題"


# ========== 主處理流程 ==========
def process_audio_file(audio_file, original_name=None):
    """
    處理單個音訊（裁剪前後靜音版 - 避免音訊變成無聲）
    
    流程:
    1. 音檔傳入（直接使用原始音訊，無預處理）
    2. 若檔案大於280秒，拆成2個片段
    3. Whisper 轉錄（提供關鍵詞）
    4. 極輕度預處理（只去除連續3次以上完全相同的片段）
    5. 規則檢測幻覺（4種規則：連續重複/短詞狂重複/內部重複/長時間重複）
    6. **裁剪前後靜音**：對幻覺段落裁剪開頭和結尾的靜音
       - 例如：[1:00-1:30] 需要重錄，開頭 [1:00-1:05] 是靜音
       - 結果：裁剪後轉錄 [1:05-1:30]
       - 中間的靜音保留（可能只是說話停頓）
    7. 對裁剪後的片段重新 Whisper 轉錄（僅在重錄時進行音訊增強）
    8. OpenAI API 1: 說話人分離和錯字修正
    9. OpenAI API 2: 分類、摘要、意圖分類（單選）
    10. 輸出
    
    特色：
    - 無預處理：直接使用原始音訊，避免過度處理導致無聲
    - 280秒自動拆分：超過280秒自動拆成2段
    - 輕度預處理：只去除連續3次以上完全相同的重複
    - 規則幻覺檢測：4種規則快速準確
    - **裁剪前後靜音**：去除開頭和結尾的靜音，中間保留
    - 多次智能重試：重轉錄時最多嘗試3次，自動選擇最佳結果
    - 僅在重錄時增強：只在重新轉錄幻覺段落時才進行音訊增強
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
    
    # Step 1: 使用原始音訊（無預處理）
    print(f"\n【跳過音訊預處理】")
    print(f"  ✓ 直接使用原始音訊，避免過度處理")
    
    original_audio = audio_file
    
    # Step 2: Whisper 轉錄（含280秒拆分邏輯）
    print(f"\n{'='*50}")
    print(f"語音轉錄（280秒自動拆分）")
    print(f"{'='*50}")
    
    step2_start = time()
    try:
        whisper_segments = transcribe_with_whisper_api(original_audio)
        step2_time = time() - step2_start
        print(f"\n  ✓ Whisper 轉錄完成: {step2_time:.2f}秒")
        print(f"  → 獲得 {len(whisper_segments)} 個語音片段")
    except Exception as e:
        print(f"  ✗ 語音辨識失敗: {e}")
        return None
    
    # Step 3: 極輕度預處理（去除明顯連續重複）
    step3_start = time()
    preprocessed_segments, removed_count = light_deduplication(whisper_segments)
    step3_time = time() - step3_start
    if removed_count > 0:
        print(f"  ✓ 預處理完成: {step3_time:.2f}秒（去除{removed_count}個明顯重複）")
    else:
        print(f"  ✓ 預處理完成: {step3_time:.2f}秒（無需清理）")
    
    # Step 4: 規則檢測幻覺（取代OpenAI）
    step4_start = time()
    hallucination_ranges = detect_hallucination_with_rules(preprocessed_segments)
    step4_time = time() - step4_start
    print(f"  ✓ 幻覺檢測完成: {step4_time:.2f}秒")
    
    # Step 5: 重新轉錄幻覺段落（僅在此步驟進行音訊增強）
    step5_start = time()
    if hallucination_ranges:
        final_segments = retranscribe_hallucination_segments(
            original_audio,  # 使用原始音訊
            hallucination_ranges, 
            preprocessed_segments
        )
        step5_time = time() - step5_start
        print(f"  ✓ 幻覺段落重新轉錄完成: {step5_time:.2f}秒")

        # Step 5.1: 第二輪幻覺檢測
        print(f"\n【第二輪幻覺檢測】")
        hallucination_ranges_2 = detect_hallucination_with_rules(final_segments)
        if hallucination_ranges_2:
            print(f"  ⚠ 偵測到 {len(hallucination_ranges_2)} 個殘留幻覺片段 → 進行第2次重轉錄")
            final_segments = retranscribe_hallucination_segments(
                original_audio,
                hallucination_ranges_2,
                final_segments
            )

            # Step 5.2: 第三輪檢測（最終確認）
            print(f"\n【最終幻覺檢測】")
            hallucination_ranges_final = detect_hallucination_with_rules(final_segments)
            if hallucination_ranges_final:
                print(f"  ⚠ 仍有 {len(hallucination_ranges_final)} 殘留問題段落，可能為音檔品質問題")
                for hr in hallucination_ranges_final:
                    print(f"    → 問題段 {hr['start']:.1f}s - {hr['end']:.1f}s，原因: {hr['reason']}")
            else:
                print(f"  ✓ 第二次重錄後無幻覺問題")
        else:
            print(f"  ✓ 重錄後已無幻覺問題")

    else:
        final_segments = preprocessed_segments
        step5_time = 0
        print(f"  ✓ 無需重新轉錄")

    
    # Step 6: OpenAI API 1 說話人分離 + 錯字修正
    print(f"\n【OpenAI API 1: 說話人分離與錯字修正】")
    step6_start = time()
    dialogue = speaker_separation_and_correction_with_gpt(final_segments)
    step6_time = time() - step6_start
    print(f"  ✓ 說話人分離與錯字修正完成: {step6_time:.2f}秒")
    
    # Step 7: 最終去重（處理重新轉錄後可能產生的重複）
    step7_start = time()
    dialogue, final_removed = final_deduplication_after_speaker(dialogue)
    step7_time = time() - step7_start
    if final_removed > 0:
        print(f"  ✓ 最終去重完成: {step7_time:.2f}秒（去除{final_removed}個重複）")
    
    # Step 8: OpenAI API 2 分類 + 摘要 + 意圖（單選）
    print(f"\n【OpenAI API 2: 分類與意圖分析】")
    step8_start = time()
    dialogue_text = "\n".join([f"{speaker}: {text}" for _, speaker, _, text in dialogue])
    problem_type, abstract, intent = classify_and_summarize_single_intent(dialogue_text)
    step8_time = time() - step8_start
    print(f"  ✓ 分類與意圖分析完成: {step8_time:.2f}秒")
    print(f"  → 分類結果: {problem_type}")
    print(f"  → 客戶意圖: {intent}")
    

    # Step 9: 儲存問題原因
    if 'hallucination_ranges_final' in locals() and hallucination_ranges_final:
        print(f"【疑似音檔問題段落】")
        for hr in hallucination_ranges_final:
            f.write(f"  - {hr['start']:.1f}s ~ {hr['end']:.1f}s: {hr['reason']}\n")
        print(f"\n⚠ 以上段落可能因雜訊或打碼導致語音辨識不穩。\n")


     # Step 10: 儲存結果
    step9_start = time()
    
    class_folder = CLASS_FOLDERS.get(problem_type, CLASS_FOLDERS["其他"])
    text_output_folder = os.path.join(class_folder, "voice_text")
    
    os.makedirs(text_output_folder, exist_ok=True)
    
    # 儲存逐字稿
    text_file = os.path.join(text_output_folder, f"{base_name}.txt")
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(f"音訊: {display_name}\n")
        f.write(f"分類: {problem_type}\n")
        f.write(f"意圖: {intent}\n")
        f.write(f"處理時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"摘要: {abstract}\n")
        f.write("=" * 60 + "\n\n")
        
        for start, speaker, end, text in dialogue:
            m, s = divmod(int(start), 60)
            f.write(f"[{m:02d}:{s:02d}] {speaker}: {text}\n")
    
    step9_time = time() - step9_start
    print(f"  ✓ 檔案儲存完成: {step9_time:.2f}秒")
    print(f"  → 儲存路徑: {class_folder}/voice_text/{base_name}.txt")
    
    # 總結
    total_time = time() - total_start
    print('-' * 60)
    print(f"完成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"總處理時間: {total_time:.2f}秒")
    print(f"  - Whisper轉錄: {step2_time:.2f}秒 ({step2_time / total_time * 100:.1f}%)")
    print(f"  - 輕度預處理: {step3_time:.2f}秒 ({step3_time / total_time * 100:.1f}%)")
    print(f"  - 幻覺檢測: {step4_time:.2f}秒 ({step4_time / total_time * 100:.1f}%)")
    if step5_time > 0:
        print(f"  - 重新轉錄: {step5_time:.2f}秒 ({step5_time / total_time * 100:.1f}%)")
    print(f"  - 說話人分離: {step6_time:.2f}秒 ({step6_time / total_time * 100:.1f}%)")
    if step7_time > 0:
        print(f"  - 最終去重: {step7_time:.2f}秒 ({step7_time / total_time * 100:.1f}%)")
    print(f"  - 分類意圖: {step8_time:.2f}秒 ({step8_time / total_time * 100:.1f}%)")
    print(f"  - 檔案儲存: {step9_time:.2f}秒 ({step9_time / total_time * 100:.1f}%)")
    print(f"已儲存至: {class_folder}/")
    
    return {
        'file': display_name,
        'time': total_time,
        'class': problem_type,
        'abstract': abstract,
        'intent': intent
    }


# ========== 主程式 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("語音分析系統 - 裁剪前後靜音版（Whisper + GPT）")
    print("核心改進：無預處理 + 裁剪前後靜音 + 多次重試 + 品質評估")
    print("=" * 60)
    
    audio_files = list(Path(input_folder).glob("*.wav")) + \
                  list(Path(input_folder).glob("*.mp3")) + \
                  list(Path(input_folder).glob("*.m4a"))
    
    if not audio_files:
        print(f"\n✗ 在 {input_folder} 資料夾中找不到音訊")
        print("支援格式: .wav, .mp3, .m4a")
    else:
        print(f"\n找到 {len(audio_files)} 個音訊")
        print(f"使用 Whisper API + GPT-4o-mini")
        print(f"處理後音訊將儲存至: {processed_folder}/")
        
        print("\n【核心功能】")
        print("  ✓ 無音訊預處理：直接使用原始音訊，避免過度處理導致無聲")
        print("  ✓ 280秒自動拆分：超過280秒自動拆成2段")
        print("  ✓ 輕度預處理：只去除連續3次以上完全相同的重複")
        print("  ✓ 規則幻覺檢測：4種規則快速準確")
        print("  ✓ **裁剪前後靜音**：重轉錄時去除開頭和結尾的靜音")
        print("     - 中間的靜音保留（可能只是說話停頓）")
        print("     - 例如：[1:00-1:30] 重錄，開頭 5秒靜音 → 只轉錄 [1:05-1:30]")
        print("  ✓ 多次智能重試：重轉錄時最多嘗試3次，自動選擇最佳結果")
        print("  ✓ 僅在重錄時增強：只在重新轉錄幻覺段落時才進行溫和的音訊增強")
        print("  ✓ 品質評估系統：自動評分，品質好就停止重試")
        print("  ✓ 最終去重：說話人分離後再檢查，確保無連續重複")
        print("  ✓ 兩階段OpenAI：說話人分離 → 分類意圖")
        
        print("\n" + "=" * 60)
        
        overall_start = time()
        results = []
        
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}]")
            result = process_audio_file(audio_file)
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
            print(f"  意圖: {r['intent']}")
            print(f"  摘要: {r['abstract'][:50]}...")
            print()
        
        print("=" * 60)
        print("檔案儲存結構:")
        for class_name, folder in CLASS_FOLDERS.items():
            count = sum(1 for r in results if r['class'] == class_name)
            if count > 0:
                print(f"  {class_name} ({count}個): {folder}/")
        print("=" * 60)
        print(f"\n處理後音訊: {processed_folder}/")
import torch
import whisper
from pyannote.audio import Pipeline
from collections import Counter
from pydub import AudioSegment
import os

# ========== 裝置設定 ==========
device = torch.device("cpu")

# ========== 預處理音檔（確保格式相容） ==========
audio_file = "20250605095508.wav"
temp_file = "temp_processed.wav"

print("Preprocessing audio file...")
audio = AudioSegment.from_file(audio_file)
audio = audio.set_channels(1)  # 單聲道
audio = audio.set_frame_rate(16000)  # 重新採樣為 16kHz
audio.export(temp_file, format="wav")

# ========== Whisper 辨識 ==========
print("Loading Whisper model...")
whisper_model = whisper.load_model("medium", device=device)

print("Transcribing with Whisper...")
whisper_result = whisper_model.transcribe(
    temp_file,
    language="zh",
    initial_prompt="客服對話逐字稿，請正確辨識數字。",
)

segments = whisper_result["segments"]

# ========== pyannote 說話人分離 ==========
print("Running speaker diarization with pyannote...")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1").to(device)
diarization = pipeline(temp_file)

# ========== 對齊文字與說話人 ==========
dialogue = []
for turn, _, speaker in diarization.itertracks(yield_label=True):
    start, end = turn.start, turn.end
    text = ""
    
    # 找出時間區間內的所有 segments
    for seg in segments:
        seg_start, seg_end = seg["start"], seg["end"]
        # 檢查是否有重疊
        if seg_start < end and seg_end > start:
            text += seg["text"]

    if text.strip():
        dialogue.append((start, speaker, text.strip()))

# ========== 自動判斷角色 ==========
role_map = {}
speakers = [d[1] for d in dialogue]
客服關鍵字 = ["很高興為您服務", "您好", "歡迎致電", "感謝您來電", "服務專線"]

if dialogue:
    first_text = dialogue[0][2]
    first_speaker = dialogue[0][1]

    if any(kw in first_text for kw in 客服關鍵字):
        role_map[first_speaker] = "客服"
        for spk in set(speakers):
            if spk != first_speaker:
                role_map[spk] = "客戶"
    else:
        main_speaker = Counter(speakers).most_common(1)[0][0]
        role_map = {spk: ("客戶" if spk == main_speaker else "客服") for spk in set(speakers)}

# ========== 輸出逐字稿 ==========
print("\n===== 對話逐字稿 =====")
for start, spk, text in dialogue:
    role = role_map.get(spk, spk)
    m, s = divmod(int(start), 60)
    print(f"[{m:02d}:{s:02d}] {role}: {text}")

# ========== 清理暫存檔 ==========
if os.path.exists(temp_file):
    os.remove(temp_file)
print("\nDone!")
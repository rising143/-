# realtime_speaker_recognition.py
# PC 实时多人声纹识别（Silero VAD + CAM++）
# 稳定版：注册均值 + 单中心比对 + 3 帧平滑

import queue
import numpy as np
import pyaudio
import torch
from scipy.spatial.distance import cosine
from collections import deque
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


########################
# 1. 音频参数
########################
SAMPLE_RATE = 16000
FRAME_SHIFT_MS = 10
FRAME_SHIFT = int(SAMPLE_RATE * FRAME_SHIFT_MS / 1000)

########################
# 2. Silero VAD
########################
vad_model, vad_utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False
)
(get_speech_timestamps, _, _, _, _) = vad_utils

########################
# 3. CAM++ 声纹模型
########################
spk_pipeline = pipeline(
    task=Tasks.speaker_verification,
    model='iic/speech_campplus_sv_zh-cn_16k-common'
)
########################
# 3.1 ASR 模型
########################
asr_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
)

########################
# 4. 声纹库
########################
# speaker_db = {"name": mean_embedding}
speaker_db = {}
SIM_THRESHOLD = 0.70   # ★ 实时系统阈值调低

########################
# 注册配置
########################
REGISTER_NUM_PER_SPK = 5

########################
# 工具函数
########################
def extract_embedding(wav: np.ndarray):
    if wav is None:
        return None

    wav = np.squeeze(wav)

    if len(wav) < SAMPLE_RATE * 0.5:
        return None

    if wav.dtype != np.float32:
        wav = wav.astype(np.float32)

    result = spk_pipeline([wav], output_emb=True)
    return result['embs'][0]


def cosine_sim(a, b):
    return 1 - cosine(a, b)



def identify_speaker(emb):
    best_name = None
    best_score = 0.0

    for name, ref_emb in speaker_db.items():
        score = cosine_sim(emb, ref_emb)
        if score > best_score:
            best_score = score
            best_name = name

    if best_score >= SIM_THRESHOLD:
        return best_name, best_score

    return None, best_score

########################
# 5. 音频采集
########################
audio_queue = queue.Queue()


def audio_callback(in_data, frame_count, time_info, status):
    audio = np.frombuffer(in_data, dtype=np.int16)
    audio_queue.put(audio)
    return (None, pyaudio.paContinue)


def start_audio_stream():
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=FRAME_SHIFT,
        stream_callback=audio_callback
    )
    stream.start_stream()
    return stream, pa

########################
# 6. 注册流程（同一条链路）
########################
def register_loop():
    print("\n📝 进入声纹注册模式")
    print("👉 输入说话人名字开始注册")
    print("👉 直接回车 / 输入 q / quit / exit 结束注册并进入识别\n")

    while True:
        name = input("👤 请输入说话人名字：").strip()

        if name == "" or name.lower() in ["q", "quit", "exit"]:
            break

        if name in speaker_db:
            print("⚠️ 该说话人已存在，请换一个名字")
            continue

        print(f"\n🎙 请说话人「{name}」开始说话")

        # ★ 清空残留音频
        while not audio_queue.empty():
            audio_queue.get()

        buffer = np.zeros(0, dtype=np.int16)
        embs = []

        current_speech_audio = []
        silence_frames = 0
        SILENCE_END_FRAMES = 5  # ≈ 50ms * 5

        while len(embs) < REGISTER_NUM_PER_SPK:
            chunk = audio_queue.get()
            buffer = np.concatenate([buffer, chunk])

            # ★ 至少 0.5 秒音频再做 VAD
            if len(buffer) < int(SAMPLE_RATE * 0.5):
                continue

            wav_float = buffer.astype(np.float32) / 32768.0
            buffer = buffer[-int(SAMPLE_RATE * 0.25):]

            speech_ts = get_speech_timestamps(
                wav_float, vad_model, sampling_rate=SAMPLE_RATE
            )

            if speech_ts:
                silence_frames = 0
                current_speech_audio.append(wav_float.copy())
            else:
                if current_speech_audio:
                    silence_frames += 1

                    if silence_frames >= SILENCE_END_FRAMES:
                        full_audio = np.concatenate(current_speech_audio)
                        current_speech_audio.clear()
                        silence_frames = 0

                        duration = len(full_audio) / SAMPLE_RATE
                        if duration < 1.0:
                            print(f"⚠️ 语音太短 ({duration:.2f}s)，请完整说一句")
                            continue

                        emb = extract_embedding(full_audio)
                        if emb is None:
                            print("⚠️ embedding 提取失败，重说")
                            continue

                        embs.append(emb)
                        print(f"✅ 已采集 {len(embs)}/{REGISTER_NUM_PER_SPK}")

        mean_emb = np.mean(np.stack(embs), axis=0)
        speaker_db[name] = mean_emb
        print(f"🎉 说话人「{name}」注册完成\n")

    print(f"\n✅ 注册结束，共注册 {len(speaker_db)} 人：{list(speaker_db.keys())}")
    print("➡️ 进入实时识别模式\n")

def score_with_active(emb, active_name):
    if active_name is None:
        return 0.0
    ref_emb = speaker_db.get(active_name)
    if ref_emb is None:
        return 0.0
    return cosine_sim(emb, ref_emb)

########################
# 7. 主流程（实时识别）
########################
def main():
    current_segment_audio = []

    in_speech = False
    silence_count = 0
    SILENCE_END_FRAMES = 5  # ≈ 50ms * 5
    current_segment_embs = []

    active_speaker = None
    active_hold = 0
    ACTIVE_SPK_HOLD = 3
    STRONG_SWITCH_THRESHOLD = 0.7  # ★ 强切换阈值

    stream, pa = start_audio_stream()

    # ★ 先注册
    register_loop()

    print("🎧 开始实时识别（Ctrl+C 退出）")

    buffer = np.zeros(0, dtype=np.int16)

    # ★ 3 帧历史融合
    history = deque(maxlen=3)

    try:
        while True:
            chunk = audio_queue.get()
            buffer = np.concatenate([buffer, chunk])

            # 至少 1 秒再处理
            # ★ 注册阶段：至少积累 0.5 秒再做 VAD
            if len(buffer) < int(SAMPLE_RATE * 0.5):
                continue

            # ★ 这一行必须有（你刚才缺的）
            wav_float = buffer.astype(np.float32) / 32768.0

            speech_ts = get_speech_timestamps(
                wav_float, vad_model, sampling_rate=SAMPLE_RATE
            )

            if speech_ts:
                silence_count = 0
                in_speech = True

                # ★ 不裁剪，整段加入
                current_segment_audio.append(wav_float.copy())

                emb = extract_embedding(wav_float)
                buffer = np.zeros(0, dtype=np.int16)

                if emb is not None:
                    current_segment_embs.append(emb)


            else:
                silence_count += 1
                buffer = buffer[-int(SAMPLE_RATE * 0.5):]

                # ★ 连续静音，判定说话结束
                if in_speech and silence_count >= SILENCE_END_FRAMES:
                    in_speech = False
                    silence_count = 0

                    # ===============================
                    # 1. 声纹判断（你原来就有）
                    # ===============================
                    if len(current_segment_embs) == 0:
                        current_segment_audio.clear()
                        current_segment_embs.clear()
                        active_hold = 0
                        active_speaker = None
                        continue

                    seg_emb = np.mean(np.stack(current_segment_embs), axis=0)
                    current_segment_embs.clear()

                    name, score = identify_speaker(seg_emb)
                    active_score = score_with_active(seg_emb, active_speaker)

                    # ===============================
                    # 2. ASR（语音转文字）
                    # ===============================
                    if len(current_segment_audio) > 0:
                        full_audio = np.concatenate(current_segment_audio)
                        current_segment_audio.clear()

                        asr_result = asr_pipeline(full_audio)

                        if isinstance(asr_result, list) and len(asr_result) > 0:
                            text = asr_result[0].get("text", "").strip()
                        elif isinstance(asr_result, dict):
                            text = asr_result.get("text", "").strip()
                        else:
                            text = ""


                    else:
                        text = ""

                    # ===============================
                    # 3. 说话人状态机（含“保持但强制切换”）
                    # ===============================
                    # ===============================
                    # 强切逻辑（新增，但不破坏原功能）
                    # ===============================
                    if (
                            active_speaker
                            and name
                            and name != active_speaker
                            and score >= STRONG_SWITCH_THRESHOLD
                            and score - active_score > 0.15  # ★ 关键：必须明显更像
                    ):
                        active_speaker = name
                        active_hold = ACTIVE_SPK_HOLD
                        print(f"🔁 强切换 → {name}: {text}  (score={score:.2f})")

                    # ===============================
                    # 原有正常识别逻辑（保留）
                    # ===============================
                    elif name:
                        active_speaker = name
                        active_hold = ACTIVE_SPK_HOLD
                        print(f"🗣 {name}: {text}  (score={score:.2f})")

                    # ===============================
                    # 原有保持逻辑（保留）
                    # ===============================
                    elif active_speaker and active_hold > 0:
                        active_hold -= 1
                        print(f"🗣 {active_speaker}: {text}  (保持)")

                    # ===============================
                    # 原有未知逻辑（保留）
                    # ===============================
                    else:
                        active_speaker = None
                        print(f"🗣 未知: {text}")




    except KeyboardInterrupt:
        print("\n🛑 停止识别")

    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


if __name__ == '__main__':
    main()

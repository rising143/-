🎤 Real-time Multi-Speaker Recognition (PC)

基于 Silero VAD + CAM++ + Paraformer ASR 的 PC 端实时多人声纹识别与中文转写系统。

支持声纹注册、实时说话人识别、自动语音转文字，适用于会议、多人大模型输入、人机交互等场景。

✨ Features

🎙 实时麦克风音频采集（16kHz）

🔇 Silero VAD 语音活动检测

🧠 CAM++ 中文声纹识别（均值建模）

🗣 Paraformer 中文 ASR

🔁 说话人保持 & 强制切换机制

👥 多人注册 & 实时识别

🧩 Pipeline
Mic → VAD → Speech Segment
            ↓
     Speaker Embedding (CAM++)
            ↓
      Cosine Similarity Match
            ↓
   Speaker State Machine
            ↓
        ASR (Paraformer)
            ↓
    Speaker + Transcription
📦 Requirements

Python 3.8 ~ 3.10

pip install numpy scipy torch pyaudio modelscope

⚠️ PyAudio 需系统已安装 portaudio（Windows 建议使用 whl）

🤖 Models
Task	Model
VAD	snakers4/silero-vad
Speaker Verification	iic/speech_campplus_sv_zh-cn_16k-common
ASR	iic/speech_paraformer-large_asr_nat-zh-cn-16k-common
📝 Speaker Registration

在程序启动后自动进入注册模式：

REGISTER_SPEAKERS = ["张三", "whs"]
REGISTER_NUM_PER_SPK = 5

每位说话人采集 5 次语音

每段语音 ≥ 0.5s

对 embedding 求均值作为最终声纹

⚙️ Key Parameters
SIM_THRESHOLD = 0.70          # Speaker match threshold
ACTIVE_SPK_HOLD = 3          # Speaker hold frames
STRONG_SWITCH_THRESHOLD = 0.7 # Forced speaker switch
▶️ Run
python realtime_speaker_recognition.py
📤 Output Example
🗣 张三: 我觉得这个方案可以再优化一下 (score=0.82)
🗣 张三: 然后我们再看下一步 (保持)
🔁 强切换 → whs: 我补充一点 (score=0.88)
🗣 未知: 刚才有人在说话吗
⚠️ Notes

推荐在相对安静环境下使用

注册与识别需使用同一麦克风设备

仅支持单通道 16kHz 音频输入

🚀 Future Work

Streaming ASR

Speaker overlap detection

Speaker DB persistence

GUI / Web service

📄 License

For research and educational use only.

🙏 Acknowledgements

Silero VAD

ModelScope

CAM++

Paraformer ASR
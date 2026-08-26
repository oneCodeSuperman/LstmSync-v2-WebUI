# -*- coding: utf-8 -*-
"""
VAE-LSTM-Sync WebUI  基于 Gradio
与 run.py 一致的导入方式（直接 import 根目录的 .pyd）。

数据目录约定：
  ./data/uploads/        - 用户上传的原视频（永久保留）
  ./data/<任务标题>/      - 预处理生成的 .dat/.npy/meta.json
  ./config.json          - 用户配置（API Key、默认 device 等）
"""

import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["ORT_LOG_LEVEL"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 关键：清掉 http(s)_PROXY 环境变量，否则本机代理会让
# Gradio 内部 startup-events 请求失败 (502)。
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
           "ALL_PROXY", "all_proxy"):
    os.environ.pop(_k, None)
# 强制让标准库彻底不信任系统代理
os.environ["NO_PROXY"] = "*"


import sys
import json
import shutil
import logging
import re
import uuid
import traceback
import threading
from pathlib import Path
from datetime import datetime


# ============== 全局拦截: 完全吞掉 ProactorBasePipeTransport 的无害错误 ==============
# Gradio uvicorn 在 stream 关闭后，asyncio 事件循环还会清理，这种回调在
# Windows 上抛 WinError 10054 (ConnectionResetError) 但没有任何功能影响。
# Python 3.8+ 提供了 sys.unraisablehook，专门捕获「线程结束/循环结束」时
# 漏网的异常；asyncio 内部通过 loop.call_exception_handler，但 _call_connection_lost
# 是在 proactor 内部，绕过了 loop.handler，最终落到 unraisablehook。
_SILENT_PATTERNS = (
    "ConnectionResetError",
    "WinError 10054",
    "_ProactorBasePipeTransport",
    "_call_connection_lost",
    "asyncio.proactor_events",
)


def _silent_unraisable_hook(unraisable):
    """拦截 asyncio cleanup 时抛出的无害异常，不写入 stderr / 不显示 traceback。"""
    try:
        exc = unraisable.exc_value
        msg = (str(exc) if exc else "") + " " + (unraisable.err_msg or "")
    except Exception:
        msg = repr(unraisable)
    if not any(pat in msg for pat in _SILENT_PATTERNS):
        # 其他错误仍打印（用 default hook）
        sys.__unraisablehook__(unraisable)


sys.unraisablehook = _silent_unraisable_hook


# 同步捕获 logger 走 default 的 Exception in callback ... 路径
class _SilentFilter(logging.Filter):
    def filter(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return True
        return not any(pat in msg for pat in _SILENT_PATTERNS)


logging.getLogger("asyncio").addFilter(_SilentFilter())
# 关掉 asyncio logger 自身的 propagation，免得被 root logger handler 抓到
logging.getLogger("asyncio").propagate = False

# ============== 抑制第三方日志 ==============
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)
logging.getLogger("insightface").setLevel(logging.ERROR)
logging.getLogger("kornia").setLevel(logging.ERROR)
logging.getLogger("albumentations").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

import gradio as gr
import asyncio

# ============== 全局吞掉无害的 asyncio 回调错误（不写到 stdout/stderr） ==============
# Gradio uvicorn 在 stream 关闭后还会触发 _call_connection_lost，Windows 上会
# 报 WinError 10054。这个回调的错误不影响功能，但会刷屏误导用户。
def _silent_asyncio_exception_handler(loop, context):
    msg = str(context.get("exception") or context.get("message") or "")
    if any(pat in msg for pat in (
        "WinError 10054", "ConnectionResetError", "_ProactorBasePipeTransport",
        "_call_connection_lost",
    )):
        return  # 默默吞掉
    # 其余异常仍走默认 handler（打印）
    loop.default_exception_handler(context)


try:
    asyncio.get_event_loop().set_exception_handler(_silent_asyncio_exception_handler)
except RuntimeError:
    # Python 3.10+：没 current loop 时跳过，gradio 启动后会再造 loop
    pass

# ============== 业务模块（与 run.py 一致） ==============
from preprocess import Preprocessor
from inference import Inference


# ============== 路径与配置 ==============
DATA_DIR = Path("./data").resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = Path("./config.json").resolve()
DEFAULT_CONFIG = {
    "api_key": "",
    "device": "cuda",
}


def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return {**DEFAULT_CONFIG, **json.loads(CONFIG_PATH.read_text(encoding="utf-8"))}
        except Exception:
            pass
    return dict(DEFAULT_CONFIG)


def save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def list_preprocess_dirs():
    """直接列出 data/ 下所有子目录"""
    if not DATA_DIR.exists():
        return ["(尚无预处理数据)"]
    dirs = [p.name for p in sorted(DATA_DIR.iterdir(), reverse=True) if p.is_dir()]
    return dirs or ["(尚无预处理数据)"]


# ============== 日志捕获 + 单行刷新 ==============
class LogCapture:
    """捕获 print() 输出，把 tqdm 进度条折叠成单行可滚动；
    同时识别 tqdm 的百分比，调用外部 progress 回调推到 Gradio。"""

    PROGRESS_RE = re.compile(r"^.*?(\d+)%\|[█▏▎▍▌▋▊▉ ]+\|\s*(\d+)/(\d+).*$")

    def __init__(self):
        self.lines = []
        self.lock = threading.Lock()
        self.last_progress_text = None
        # 进度回调：cb(ratio in [0, 1], label_str)
        self.progress_cb = None
        self._last_pct = -1  # 节流：相同百分比只推一次
        self._cur_desc = ""  # 当前处理阶段描述，避免 AttributeError

    # 已知无害的 asyncio / Windows stream cleanup 噪音，不写到日志面板
    _FILTER_PATTERNS = (
        "ConnectionResetError",
        "WinError 10054",
        "_ProactorBasePipeTransport",
        "asyncio.events",
        "_call_connection_lost",
    )

    def write(self, msg):
        if not msg:
            return
        # 对整段做一次过滤
        for pat in self._FILTER_PATTERNS:
            if pat in msg:
                return
        ts = datetime.now().strftime("%H:%M:%S")
        for raw_line in msg.rstrip("\n").splitlines():
            stripped = raw_line.rstrip()
            if not stripped:
                continue
            # tqdm 单行进度条 -> 折叠成一行 + 推 Gradio Progress
            if "%|" in stripped and "/" in stripped:
                m = self.PROGRESS_RE.match(stripped)
                if m:
                    pct, cur, total = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    label = f"{self._cur_desc or '处理中'} {cur}/{total}"
                    self.last_progress_text = f"[{ts}] 进度 {pct}% ({cur}/{total})"
                    self._replace_progress()
                    # 推 Gradio 进度条（节流）
                    if self.progress_cb is not None and pct != self._last_pct:
                        self._last_pct = pct
                        try:
                            self.progress_cb(pct / 100.0, label)
                        except Exception:
                            pass
                    continue
            with self.lock:
                self.lines.append(f"[{ts}] {stripped}")
                if len(self.lines) > 1500:
                    self.lines = self.lines[-1000:]

    def _replace_progress(self):
        with self.lock:
            if self.last_progress_text:
                self.lines.append(self.last_progress_text)
                self.last_progress_text = None

    def flush(self):
        pass

    def get_text(self):
        with self.lock:
            return "\n".join(self.lines[-120:])

    def clear(self):
        with self.lock:
            self.lines = []
        self._last_pct = -1

    def set_progress_cb(self, cb, desc: str = ""):
        """设置进度回调；cb(ratio, label)；desc 是这段日志的描述（preprocess/inference）"""
        self.progress_cb = cb
        self._cur_desc = desc
        self._last_pct = -1


log_capture = LogCapture()
_orig_stdout, _orig_stderr = sys.stdout, sys.stderr


class TeeWriter:
    """代理 stream，同时往控制台 + LogCapture 写"""

    def __init__(self, original, capture):
        self.original = original
        self.capture = capture

    def write(self, msg):
        try:
            self.original.write(msg)
        except Exception:
            pass
        if msg:
            self.capture.write(msg)

    def flush(self):
        try:
            self.original.flush()
        except Exception:
            pass

    # 代理常见流属性（uvicorn 等会用）
    def isatty(self):
        return getattr(self.original, "isatty", lambda: False)()

    def fileno(self):
        return self.original.fileno()

    def readable(self):
        return getattr(self.original, "readable", lambda: False)()

    def writable(self):
        return getattr(self.original, "writable", lambda: True)()

    @property
    def closed(self):
        return getattr(self.original, "closed", False)


sys.stdout = TeeWriter(_orig_stdout, log_capture)
sys.stderr = TeeWriter(_orig_stderr, log_capture)


# ============== 工具函数 ==============
def _resolve_uploaded_path(file_obj) -> str:
    if file_obj is None:
        return ""
    if isinstance(file_obj, str):
        return file_obj
    if hasattr(file_obj, "name"):
        return file_obj.name
    return str(file_obj)


def save_uploaded_file(file_obj, sub_dir: str, filename: str = None) -> str:
    src = _resolve_uploaded_path(file_obj)
    if not src or not os.path.exists(src):
        raise ValueError(f"上传文件路径无效: {src}")
    target_dir = DATA_DIR / sub_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = Path(src).name
    target_path = target_dir / filename
    shutil.copy(src, target_path)
    return str(target_path)


def copy_uploaded_to(file_obj, target_dir: Path, filename: str = None) -> str:
    """把上传文件复制到指定目录（不会保留在 data/uploads/）。
    target_dir 由调用方控制，必须是 data/ 之外的目录，避免被 Preprocessor.run rmtree 误删。
    """
    src = _resolve_uploaded_path(file_obj)
    if not src or not os.path.exists(src):
        raise ValueError(f"上传文件路径无效: {src}")
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = Path(src).name
    target_path = target_dir / filename
    shutil.copy(src, target_path)
    return str(target_path)


_TMP_UPLOADS = Path(__file__).parent.resolve() / "temp" / "uploads"


def stage_uploaded_video(file_obj) -> str:
    """把上传的视频暂存到项目根目录的 temp/uploads/（不会被 rmtree），返回绝对路径。
    文件名用 uuid 重新生成，避免中文/括号/空格路径导致 soundfile / cv2 / ffmpeg 失败。
    """
    src = _resolve_uploaded_path(file_obj)
    if not src or not os.path.exists(src):
        raise ValueError(f"上传文件路径无效: {src}")
    suffix = Path(src).suffix or ".mp4"
    target_filename = f"video_{uuid.uuid4().hex}{suffix}"
    return copy_uploaded_to(file_obj, _TMP_UPLOADS, filename=target_filename)


def stage_uploaded_audio(file_obj, task_dir_name: str) -> str:
    """把音频暂存到 data/<任务>/temp/ 下，保证和 Preprocessor 数据共存以便管理。
    文件名用 uuid 重新生成，避免中文/括号/空格路径导致 soundfile 失败。
    """
    src = _resolve_uploaded_path(file_obj)
    if not src or not os.path.exists(src):
        raise ValueError(f"上传文件路径无效: {src}")
    suffix = Path(src).suffix or ".wav"
    target_filename = f"audio_{uuid.uuid4().hex}{suffix}"
    return copy_uploaded_to(file_obj, DATA_DIR / task_dir_name / "temp", filename=target_filename)


def sanitize_title(title: str) -> str:
    """去掉不安全字符，作为目录名"""
    if not title:
        return ""
    cleaned = re.sub(r"[\\/:*?\"<>|\s]", "_", title.strip())
    return cleaned[:64] if cleaned else ""


def list_cuda_devices():
    """列出可用推理设备，与 inference.py 的 _parse_device 保持一致。

    - 有 CUDA: ['cuda', 'cuda:0', 'cuda:1', ..., 'cpu', 'mps'?]
    - 仅 MPS（Apple Silicon）: ['mps', 'cpu']
    - 都没有: ['cpu']
    """
    devices = []
    try:
        import torch
        if torch.cuda.is_available():
            devices.append("cuda")
            for i in range(torch.cuda.device_count()):
                devices.append(f"cuda:{i}")
        # MPS（Apple Silicon）
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append("mps")
    except Exception:
        pass
    if "cpu" not in devices:
        devices.append("cpu")
    return devices


def pick_folder(initial_dir: str = None) -> str:
    """弹出 Windows / macOS / Linux 原生文件夹选择对话框，返回选中路径。
    Gradio 后端进程必须和 tkinter 同一显示器才能弹窗（一般是 OK 的）。
    用户取消返回空串。
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        # tk 在 Windows 上图标不显示；用 attributes 置顶
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        # 优先使用输入框里已填的目录；否则从项目根目录开始（不是 data/）
        if initial_dir and os.path.isdir(initial_dir):
            initial = initial_dir
        else:
            initial = str(Path(__file__).parent.resolve())
        path = filedialog.askdirectory(
            title="选择输出目录",
            initialdir=initial,
            mustexist=False,
        )
        root.destroy()
        return path or ""
    except Exception as e:
        print(f"[WebUI] 弹文件夹选择失败: {e}")
        return ""


# ============== 移除 tqdm monkey-patch：进度由日志捕获器 LogCapture 解析 ==============
# 直接复用原生 tqdm，让日志里有原始 tqdm 行（方便用户观察）
# 进度通过 LogCapture.set_progress_cb 推给 Gradio。


# ============== 配置保存相关 ==============
def save_api_key_action(api_key):
    cfg = load_config()
    cfg["api_key"] = (api_key or "").strip()
    save_config(cfg)
    return gr.Info(f"已保存到 {CONFIG_PATH}（重启后仍生效）")


def save_api_key_action_no_toast(api_key):
    """纯保存，不弹 Toast（保留给旧接口调用）"""
    cfg = load_config()
    cfg["api_key"] = (api_key or "").strip()
    save_config(cfg)


def list_preprocess_dirs_ui():
    return gr.update(choices=list_preprocess_dirs())


# ============== 任务 1：预处理 ==============
def run_preprocess_ui(
    title,
    video,
    face_size,
    frame_batch_size,
    device,
):
    log_capture.clear()

    title_clean = sanitize_title(title)
    if not title_clean:
        return ("请填写任务标题！", gr.update(), gr.update(), "")
    if video is None:
        return ("请先上传视频！", gr.update(), gr.update(), "")

    cfg = load_config()
    api_key = cfg.get("api_key", "").strip()
    if not api_key:
        return ("API Key 未配置，请先在『设置』中保存 API Key。",
                gr.update(), gr.update(), "")

    try:
        # 1. 把上传视频暂存到系统临时目录（不会被 rmtree），不再归档到 data/uploads/
        # video_filename = Path(_resolve_uploaded_path(video)).name
        video_save_path = stage_uploaded_video(video)
        # 如果用户传的文件没有扩展名，强制补 .mp4 防止 cv2 失败
        if not Path(video_save_path).suffix:
            mp4_path = video_save_path + ".mp4"
            os.rename(video_save_path, mp4_path)
            video_save_path = mp4_path
            # video_filename = Path(video_save_path).name
        print(f"[WebUI] 已暂存上传视频: {video_save_path}")

        # 2. 校验 cv2 能读
        import cv2
        cap = cv2.VideoCapture(video_save_path)
        ok, frame = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        print(f"[WebUI] 视频诊断: size={os.path.getsize(video_save_path)} bytes, "
              f"cv2.read_ok={ok}, fps={fps}")
        if not ok:
            return (f"上传视频 cv2 无法读取（可能损坏）：{video_save_path}",
                    gr.update(), gr.update(), "")

        # 3. 模型路径
        face_size = int(face_size)
        vae_encoder_path = f"./checkpoints/{face_size}.encoder.onnx"
        if not os.path.exists(vae_encoder_path):
            return f"找不到 VAE Encoder 模型: {vae_encoder_path}", gr.update(), gr.update(), ""

        # 4. 输出目录：data/<标题>/
        output_dir = str(DATA_DIR / title_clean)
        # 注意：Preprocessor.run 第一步会 rmtree(output_dir)，数据放这里安全

        print(f"[WebUI] 开始预处理: title={title_clean}, video={video_save_path}, output={output_dir}")

        pre = Preprocessor(
            face_size=face_size,
            vae_encoder_path=vae_encoder_path,
            device=device,
            api_key=api_key,
        )

        pre.run(
            video_path=video_save_path,
            output_dir=output_dir,
            frame_batch_size=int(frame_batch_size),
        )

        print(f"[WebUI] 预处理完成！输出目录: {output_dir}")
        return (
            f"预处理完成！\n标题: {title_clean}\n输出: {output_dir}",
            gr.update(choices=list_preprocess_dirs(), value=title_clean),
            gr.update(value=title_clean),
            output_dir,
        )
    except Exception as e:
        traceback.print_exc()
        return f"预处理失败: {e}", gr.update(), gr.update(), ""


# ============== 任务 2：推理 ==============
def run_inference_ui(
    preprocess_dir,
    audio,
    face_size,
    batch_size,
    sync_offset,
    audio_mode,
    data_load_mode,
    device,
    output_dir,
):
    log_capture.clear()

    if not preprocess_dir or preprocess_dir == "(尚无预处理数据)":
        return "请先选择预处理数据目录！", None
    if audio is None:
        return "请先上传音频！", None

    cfg = load_config()
    api_key = cfg.get("api_key", "").strip()

    try:
        data_dir = str(DATA_DIR / preprocess_dir)
        if not os.path.isdir(data_dir):
            return f"预处理目录不存在: {data_dir}", None

        # 1. 音频暂存到 data/<任务>/ 下，方便和产物放一起管理
        audio_save_path = stage_uploaded_audio(audio, preprocess_dir)
        print(f"[WebUI] 已暂存上传音频: {audio_save_path}")

        face_size = int(face_size)
        human_path = f"./checkpoints/{face_size}.m.onnx"
        hubert_path = "./checkpoints/chinese-hubert-large/"
        if not os.path.exists(human_path):
            return f"找不到 Human 模型: {human_path}", None
        if not os.path.exists(hubert_path):
            return f"找不到 Hubert 模型: {hubert_path}", None

        # 2. 输出路径：优先用户指定，否则 data/<任务>/
        if output_dir and output_dir.strip():
            out_dir = Path(output_dir.strip())
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = DATA_DIR / preprocess_dir
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        output_video = str(out_dir / f"{preprocess_dir}_{ts}.mp4")
        if os.path.exists(output_video):
            os.remove(output_video)

        # 把中间临时 avi / 16k wav 放到项目根目录的 temp/ 下，
        # 避免 soundfile 在 C:\Users\Administrator\AppData\Local\Temp\
        # 这种路径下偶发 LibsndfileError: System error
        # 文件名用 uuid，完全避免中文/括号/空格
        project_root = Path(__file__).parent.resolve()
        tmp_dir = project_root / "temp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        run_uid = uuid.uuid4().hex
        video_temp_path = str(tmp_dir / f"{run_uid}_temp.avi")
        audio_temp_path = str(tmp_dir / f"{run_uid}_audio_16k.wav")
        # 清掉可能存在的旧临时
        for f in [video_temp_path, audio_temp_path]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

        print("[WebUI] 正在初始化 Inference ...")

        infer = Inference(
            human_path=human_path,
            hubert_path=hubert_path,
            batch_size=int(batch_size),
            sync_offset=int(sync_offset),
            device=device,
            data_load_mode=data_load_mode,
            audio_mode=audio_mode,
            api_key=api_key,
        )

        print(f"[WebUI] 开始推理: data_dir={data_dir}, audio={audio_save_path}")
        try:
            infer.run(
                data_dir=data_dir,
                audio_path=audio_save_path,
                video_out_path=output_video,
                audio_temp_path=audio_temp_path,
                video_temp_path=video_temp_path,
            )
        finally:
            # 清理临时 avi / 16k wav
            for f in [video_temp_path, audio_temp_path]:
                try:
                    if f and os.path.exists(f):
                        os.remove(f)
                except Exception:
                    pass

        print(f"[WebUI] 推理完成！输出: {output_video}")
        return f"推理完成！\n输出: {output_video}", output_video
    except Exception as e:
        traceback.print_exc()
        return f"推理失败: {e}", None


# ============== Gradio 界面 ==============
def build_ui():
    cfg = load_config()
    cuda_choices = list_cuda_devices()

    custom_css = """
    /* 全局字体 */
    body, .gradio-container, .prose, .markdown, p, span, button, label, input, textarea {
        font-family: 'Microsoft YaHei UI', 'Microsoft YaHei', 'PingFang SC',
                     'Hiragino Sans GB', 'Source Han Sans SC', 'Noto Sans CJK SC',
                     'WenQuanYi Micro Hei', system-ui, sans-serif !important;
        font-feature-settings: 'liga' 0, 'calt' 0;
    }
    textarea, .log-box textarea, pre, code {
        font-family: 'JetBrains Mono', 'Cascadia Code', Consolas,
                     'Microsoft YaHei UI', 'Source Han Mono SC',
                     'Noto Sans Mono CJK SC', monospace !important;
        font-size: 13px !important;
        line-height: 1.45 !important;
        letter-spacing: 0 !important;
        word-break: break-all;
    }
    /* 日志面板：最低高度 = 8 行（约 160px），有内容自然增长，超过 360px 出滚动条 */
    .log-box textarea {
        min-height: 160px !important;
        max-height: 360px !important;
        overflow-y: auto !important;
        resize: none !important;
        font-size: 12.5px !important;
    }
    .log-box {
        min-height: 180px !important;
    }
    /* ====== 输出目录 + 选择按钮对齐 ====== */
    /* 让两者顶部对齐，即按钮顶部 = 输入框顶部 */
    .gradio-row.folder-row {
        align-items: flex-start !important;
    }
    .folder-pick-btn {
        /* 按钮顶上和 textarea 容器顶上齐平，要腾出 label 高度 */
        margin-top: 30px !important;       /* ≈ Gradio Textbox label 高度 */
        height: 40px !important;
        min-height: 40px !important;
        max-height: 40px !important;
        font-size: 13px !important;
        line-height: 1 !important;
        padding: 0 14px !important;
        white-space: nowrap !important;
    }
    /* 隐藏 Gradio 顶部 "Use via API" / "Settings" 等按钮 */
    .built-in { display: none !important; }
    """

    with gr.Blocks(
        title="VAE-LSTM-Sync WebUI",
        theme=gr.themes.Soft(primary_hue="blue"),
        css=custom_css,
    ) as demo:
        gr.Markdown(
            """
            # 🎬 VAE-LSTM-Sync 数字人 WebUI
            基于 Gradio 的可视化操作界面，支持**视频预处理**和**音频驱动推理**两步式生成数字人视频。
            ---
            """
        )

        with gr.Tabs():
            # ----------- 设置 -----------
            with gr.Tab("⚙️ 设置"):
                gr.Markdown(
                    "### API Key 配置（保存到本地 `config.json`，下次自动加载）\n\n"
                    "👉 **注册地址：[https://lstmsync.andclaw.cn/](https://lstmsync.andclaw.cn/)**"
                )
                api_key_in = gr.Textbox(
                    label="API Key（注册地址 https://lstmsync.andclaw.cn/）",
                    value=cfg.get("api_key", ""),
                    type="password",
                    placeholder="sk-...",
                )
                save_cfg_btn = gr.Button("💾 保存配置", variant="primary")
                save_cfg_btn.click(
                    save_api_key_action,
                    inputs=[api_key_in],
                )
                gr.Markdown(
                    f"> 配置文件路径：`{CONFIG_PATH}`\n\n"
                    "> 注册时一般会拿到一个 `sk-...` 开头的 API Key，复制粘贴到上面的输入框即可。"
                )

            # ----------- Tab 1：预处理 -----------
            with gr.Tab("① 视频预处理"):
                gr.Markdown("### 上传原始视频，生成 VAE 编码后的预处理数据")
                with gr.Row():
                    with gr.Column(scale=1):
                        title_in = gr.Textbox(
                            label="📌 任务标题 (必填，作为 data/ 下的子目录名)",
                            placeholder="例如: my_demo",
                            max_lines=1,
                        )
                        video_in = gr.Video(
                            label="上传视频 (mp4 / avi)",
                            sources=["upload"],
                            height=280,
                        )
                        face_size_pre = gr.Radio(
                            choices=[256, 384],
                            value=256,
                            label="人脸尺寸 face_size (与模型匹配)",
                        )
                        frame_batch_size = gr.Slider(
                            minimum=4,
                            maximum=128,
                            value=32,
                            step=4,
                            label="每批处理帧数 (显存不足调小)",
                        )
                        device_pre = gr.Radio(
                            choices=cuda_choices,
                            value=cfg.get("device", "cuda"),
                            label="推理设备（预处理需要 CUDA / cuda:N）",
                        )
                        pre_btn = gr.Button("🚀 开始预处理", variant="primary")

                    with gr.Column(scale=1):
                        pre_status = gr.Textbox(
                            label="预处理状态",
                            interactive=False,
                            lines=3,
                        )
                        pre_output_dir = gr.Textbox(
                            label="预处理输出目录 (data/标题/)",
                            interactive=False,
                        )
                        gr.Markdown("#### 处理日志")
                        pre_log = gr.Textbox(
                            label="日志",
                            interactive=False,
                            lines=1,
                            autoscroll=True,
                            max_lines=20,
                            elem_classes=["log-box"],
                        )

                pre_state = gr.State(value=None)

            # ----------- Tab 2：推理 -----------
            with gr.Tab("② 音频驱动推理"):
                gr.Markdown("### 上传音频，选择已预处理的数据，生成数字人视频")
                with gr.Row():
                    with gr.Column(scale=1):
                        preprocess_dir_in = gr.Dropdown(
                            choices=list_preprocess_dirs(),
                            label="选择已预处理的数据 (data/<标题>/)",
                            value=list_preprocess_dirs()[0],
                            allow_custom_value=True,
                        )
                        refresh_btn = gr.Button("🔄 刷新列表")

                        audio_in = gr.Audio(
                            label="上传驱动音频 (wav / mp3)",
                            type="filepath",
                        )
                        face_size_inf = gr.Radio(
                            choices=[256, 384],
                            value=256,
                            label="人脸尺寸 (与预处理数据一致)",
                        )
                        batch_size_inf = gr.Slider(
                            minimum=1,
                            maximum=512,
                            value=2,
                            step=1,
                            label="推理 batch_size (显存不足调小，默认 2)",
                        )
                        sync_offset = gr.Slider(
                            minimum=-20,
                            maximum=20,
                            value=0,
                            step=1,
                            label="音视频同步偏移 (帧)",
                        )
                        audio_mode = gr.Radio(
                            choices=["full", "streaming"],
                            value="full",
                            label="音频处理模式 (长音频 -> streaming)",
                        )
                        data_load_mode_in = gr.Radio(
                            choices=["auto", "full", "streaming"],
                            value="auto",
                            label="数据加载方式  auto=自动 / full=全量 / streaming=流式",
                        )
                        device_inf = gr.Radio(
                            choices=cuda_choices,
                            value=cfg.get("device", "cuda"),
                            label="推理设备 (cuda / cuda:N / cpu / mps)",
                        )
                        with gr.Row(elem_classes=["folder-row"]):
                            output_dir_in = gr.Textbox(
                                label="📁 自定义输出目录（默认项目根目录，留空则用 data/<任务>/）",
                                value=str(Path(__file__).parent.resolve()),
                                placeholder="例如: D:\\videos\\output",
                                max_lines=1,
                                scale=5,
                                container=True,
                            )
                            output_dir_btn = gr.Button(
                                "📂 选择文件夹",
                                scale=0,
                                min_width=140,
                                elem_classes=["folder-pick-btn"],
                            )
                        output_dir_btn.click(
                            pick_folder,
                            inputs=[output_dir_in],
                            outputs=output_dir_in,
                        )
                        infer_btn = gr.Button("🎬 开始推理", variant="primary")

                    with gr.Column(scale=1):
                        infer_status = gr.Textbox(
                            label="推理状态",
                            interactive=False,
                            lines=2,
                        )
                        output_video = gr.Video(
                            label="生成的数字人视频",
                            interactive=False,
                            height=400,
                        )
                        gr.Markdown("#### 处理日志")
                        infer_log = gr.Textbox(
                            label="日志",
                            interactive=False,
                            lines=1,
                            autoscroll=True,
                            max_lines=20,
                            elem_classes=["log-box"],
                        )

            # ----------- Tab 3：使用说明 -----------
            with gr.Tab("📖 使用说明"):
                gr.Markdown(
                    f"""
                    ### 流程
                    1. **⚙️ 设置**：填入 API Key、选择推理设备，点击「保存配置」（保存到 `{CONFIG_PATH}`）。
                    2. **① 视频预处理**：填任务标题（将作为 `data/<标题>/` 目录名），上传视频，开始预处理。
                    3. **② 音频驱动推理**：选择 `data/<标题>/`，上传音频，开始推理。

                    ### 关于 API Key
                    - **注册地址：[https://lstmsync.andclaw.cn/](https://lstmsync.andclaw.cn/)**
                    - 注册后会获得一个 `sk-...` 开头的 Key，填到「⚙️ 设置」页保存即可。
                    - 没有 Key 时预处理/推理会拒绝执行。

                    ### 数据目录约定
                    - `data/<标题>/` 预处理生成的 `.dat/.npy/meta.json`
                    - `data/<标题>/<标题>_时间戳.mp4` 默认推理产物
                    - 自定义输出目录可通过 📂 按钮选择

                    ### 注意事项
                    - `face_size` 必须**前后保持一致**（256 ↔ 256 或 384 ↔ 384）。
                    - 长音频（> 30s）建议 `audio_mode=streaming`。
                    - 大量帧数据建议 `data_load_mode=streaming`，避免一次性占用太多内存。
                    """
                )

        # ----------- 事件绑定 -----------
        pre_btn.click(
            fn=run_preprocess_ui,
            inputs=[title_in, video_in, face_size_pre, frame_batch_size, device_pre],
            outputs=[pre_status, preprocess_dir_in, pre_state, pre_output_dir],
        )

        infer_btn.click(
            fn=run_inference_ui,
            inputs=[
                preprocess_dir_in,
                audio_in,
                face_size_inf,
                batch_size_inf,
                sync_offset,
                audio_mode,
                data_load_mode_in,
                device_inf,
                output_dir_in,
            ],
            outputs=[infer_status, output_video],
        )

        refresh_btn.click(
            list_preprocess_dirs_ui,
            outputs=preprocess_dir_in,
        )

        # ----------- 页面加载时刷新预处理列表（避免 Timer.tick 在某些环境下导致组件渲染异常） -----------
        demo.load(
            list_preprocess_dirs_ui,
            outputs=preprocess_dir_in,
        )

        # ----------- 定时刷新日志面板 -----------
        timer = gr.Timer(value=2.0, active=True)

        # Gradio 4.x 的 autoscroll 属性在 Textbox 上不稳定，这里用 then + js 强制滚动到底
        scroll_js = """
        (text) => {
            try {
                // 找到所有 .log-box 下的 textarea，把滚动位置放到最底
                const boxes = document.querySelectorAll('.log-box textarea');
                boxes.forEach(el => { el.scrollTop = el.scrollHeight; });
            } catch (e) {}
            return text;
        }
        """
        timer.tick(
            fn=lambda: (log_capture.get_text(), log_capture.get_text()),
            outputs=[pre_log, infer_log],
            js=scroll_js,
        )

        gr.Markdown(
            """
            ---
            启动命令：`python webui.py`　|　默认端口：`7860`
            """
        )

    return demo


# ============== 入口 ==============
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VAE-LSTM-Sync Gradio WebUI")
    parser.add_argument("--host", default=os.environ.get("WEBUI_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("WEBUI_PORT", "7860")))
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--no-browser", action="store_true",
                        help="不要自动打开浏览器（默认会开）")
    args = parser.parse_args()

    print(f"[WebUI] 数据目录: {DATA_DIR}")
    print(f"[WebUI] 配置文件: {CONFIG_PATH}")
    print(f"[WebUI] CUDA 可用: {__import__('torch').cuda.is_available()}  设备: {list_cuda_devices()}")
    print(f"[WebUI] 启动 Gradio: http://{args.host}:{args.port}")

    demo = build_ui()

    # Gradio 启动时会新建 event loop；把异常 handler 在 launch 时也挂上
    try:
        _orig_launch = demo.queue().launch
        def _launch_with_hook(*a, **kw):
            try:
                loop = asyncio.get_event_loop_policy().get_event_loop()
                loop.set_exception_handler(_silent_asyncio_exception_handler)
            except Exception:
                pass
            return _orig_launch(*a, **kw)
        # 在 demo 上 monkey-patch launch，等到调用时才 hook
        demo.queue().launch = _launch_with_hook
    except Exception:
        pass

    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
    )

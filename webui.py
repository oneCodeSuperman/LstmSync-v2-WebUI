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


import os
import sys
import json
import shutil
import logging
import re
import uuid
import traceback
import threading
import urllib.request
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

# 直播预处理数据根目录
DATA_LIVE_DIR = Path("./data_live").resolve()
DATA_LIVE_DIR.mkdir(parents=True, exist_ok=True)

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


def list_live_data_dirs():
    """列出 data_live/ 下直播预处理生成的目录（含 meta.json + video_25fps.mp4）"""
    if not DATA_LIVE_DIR.exists():
        return ["(尚无直播预处理数据，请先运行『③ 直播预处理』)"]
    dirs = []
    for p in sorted(DATA_LIVE_DIR.iterdir(), reverse=True):
        if not p.is_dir():
            continue
        if (p / "meta.json").exists() and (p / "video_25fps.mp4").exists():
            dirs.append(p.name)
    return dirs or ["(尚无直播预处理数据，请先运行『③ 直播预处理』)"]


# ============== 日志捕获 + 单行刷新 ==============
class LogCapture:
    """捕获 print() 输出，把 tqdm 进度条折叠成单行可滚动；
    同时识别 tqdm 的百分比，调用外部 progress 回调推到 Gradio。"""

    # tqdm "%|...| N/T" 格式：含百分数 + 进度条字符
    PROGRESS_RE = re.compile(r"^.*?(\d+)%\|[█▏▎▍▌▋▊▉ ]+\|\s*(\d+)/(\d+).*$")
    # tqdm "Nit [elapsed, X.XXit/s]" 格式：tqdm 在 total 未知 / 终端太窄时会切成这种格式
    # 例: Inference (full mode): 11it [00:01,  8.36it/s]
    #     Encoding audio (full mode): 5/30 [00:02,  2.34it/s]
    PROGRESS_IT_RE = re.compile(
        r"^(?P<desc>.+?):\s*(?P<cur>\d+)(?:it)?\s*\[(?P<elapsed>[^,]+),\s*(?P<rate>[\d.]+)it/s\]"
        r"(?:\s*\[(?P<post>.*?)\])?\s*$"
    )

    def __init__(self):
        self.lines = []
        self.lock = threading.Lock()
        self.last_progress_text = None
        # 进度回调：cb(ratio in [0, 1], label_str)
        self.progress_cb = None
        self._last_pct = -1  # 节流：相同百分比只推一次
        self._cur_desc = ""  # 当前处理阶段描述，避免 AttributeError
        # it/s 格式进度折叠：把同一 desc 的最近一行存这里，下一帧到达时替换而不是追加
        self._last_it_line = None
        self._last_it_idx = -1

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
                    # it/s 格式计数清零
                    self._last_it_line = None
                    continue
            # tqdm "Nit [t, X.XXit/s]" 格式（total 未知或终端太窄时）
            if "it/s]" in stripped:
                m = self.PROGRESS_IT_RE.match(stripped)
                if m:
                    desc = m.group("desc").strip()
                    cur = int(m.group("cur"))
                    rate = float(m.group("rate"))
                    line = f"[{ts}] {desc}: {cur}it [{m.group('elapsed')}, {rate:.2f}it/s]"
                    with self.lock:
                        if self._last_it_line is not None:
                            # 用最新一行替换上一行：先去掉上一行
                            try:
                                self.lines.remove(self._last_it_line)
                            except ValueError:
                                pass
                        self.lines.append(line)
                        if len(self.lines) > 1500:
                            self.lines = self.lines[-1000:]
                    self._last_it_line = line
                    continue
            with self.lock:
                self.lines.append(f"[{ts}] {stripped}")
                if len(self.lines) > 1500:
                    self.lines = self.lines[-1000:]
            # 出现非 tqdm 输出，重置 it/s 折叠状态
            self._last_it_line = None

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


def list_live_data_dirs_ui():
    return gr.update(choices=list_live_data_dirs())


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


# ============== 任务 3：直播预处理 ==============
# 常见音频后缀（选择目录时用于过滤无关文件）
_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".wma"}


def _audio_sort_key(name: str):
    """音频排序键：优先按文件名中出现的数字排序（与 extract_number 逻辑一致），无数字按名称"""
    m = re.search(r"\d+", name)
    if m:
        return (0, int(m.group()), name.lower())
    return (1, 0, name.lower())


def run_live_preprocess_ui(
    title,
    video,
    audios,
    face_size,
    frame_batch_size,
    res_preset_label,
    sync_offset,
    device,
):
    log_capture.clear()

    title_clean = sanitize_title(title)
    if not title_clean:
        return "请填写直播标题！", gr.update()
    if video is None:
        return "请先上传视频！", gr.update()
    audios_dir = (audios or "").strip().strip('"')
    if not audios_dir:
        return "请输入或选择音频目录！", gr.update()
    if not os.path.isdir(audios_dir):
        return f"音频目录不存在: {audios_dir}", gr.update()

    cfg = load_config()
    api_key = cfg.get("api_key", "").strip()
    if not api_key:
        return "API Key 未配置，请先在『设置』中保存 API Key。", gr.update()

    try:
        from preprocess_live import PreprocessLive

        # 1. 暂存视频（不会被 rmtree）
        video_save_path = stage_uploaded_video(video)
        if not Path(video_save_path).suffix:
            mp4_path = video_save_path + ".mp4"
            os.rename(video_save_path, mp4_path)
            video_save_path = mp4_path
        print(f"[WebUI] 已暂存上传视频: {video_save_path}")

        # 2. 暂存所选目录中的音频到 temp/live_audios_<标题>/（不能放 output_dir 内，run_video 会 rmtree）
        #    按原文件名中的数字排序后重命名为纯数字，保证 extract_number 顺序正确，避免中文/空格导致 ffmpeg 失败
        project_root = Path(__file__).parent.resolve()
        audios_stage_dir = project_root / "temp" / f"live_audios_{title_clean}"
        if audios_stage_dir.exists():
            shutil.rmtree(audios_stage_dir, ignore_errors=True)
        audios_stage_dir.mkdir(parents=True, exist_ok=True)

        audio_files = []
        for root, _, names in os.walk(audios_dir):
            for name in names:
                src = os.path.join(root, name)
                if Path(src).suffix.lower() not in _AUDIO_EXTS:
                    continue
                audio_files.append(src)
        if not audio_files:
            return "音频目录中没有找到音频文件（支持 " + "/".join(sorted(_AUDIO_EXTS)) + "）！", gr.update()
        audio_files.sort(key=lambda p: _audio_sort_key(os.path.basename(p)))
        for idx, src in enumerate(audio_files):
            suffix = Path(src).suffix or ".wav"
            shutil.copy(src, audios_stage_dir / f"{idx}{suffix}")
        print(f"[WebUI] 已暂存 {len(audio_files)} 个音频: {audios_dir} -> {audios_stage_dir}")

        # 3. 模型路径
        face_size = int(face_size)
        vae_encoder_path = f"./checkpoints/{face_size}.encoder.onnx"
        hubert_path = "./checkpoints/chinese-hubert-large/"
        if not os.path.exists(vae_encoder_path):
            return f"找不到 VAE Encoder 模型: {vae_encoder_path}", gr.update()
        if not os.path.exists(hubert_path):
            return f"找不到 Hubert 模型: {hubert_path}", gr.update()

        # 4. 输出目录：data_live/<标题>/
        dir_name = title_clean
        output_dir = str(DATA_LIVE_DIR / dir_name)

        res_map = {
            "不压缩(原分辨率)": None,
            "480p": "480p",
            "720p": "720p",
            "1080p": "1080p",
        }
        res_preset = res_map.get(res_preset_label, "1080p")

        print(f"[WebUI] 开始直播预处理: title={dir_name}, output={output_dir}")

        pre = PreprocessLive(
            face_size=face_size,
            vae_encoder_path=vae_encoder_path,
            hubert_path=hubert_path,
            device=device,
            api_key=api_key,
            sync_offset=int(sync_offset),
        )
        pre.run(
            video_path=video_save_path,
            audios_dir=str(audios_stage_dir),
            output_dir=output_dir,
            fps=25,
            frame_batch_size=int(frame_batch_size),
            res_preset=res_preset,
        )

        print(f"[WebUI] 直播预处理完成！输出目录: {output_dir}")
        return (
            f"直播预处理完成！\n标题: {dir_name}\n输出: {output_dir}",
            gr.update(choices=list_live_data_dirs(), value=dir_name),
        )
    except Exception as e:
        traceback.print_exc()
        return f"直播预处理失败: {e}", gr.update()


# ============== 任务 4：直播推流 ==============
# 全局状态：start(block=False) 返回的 handle，含 stop/wait/is_running
_live_state = {"handle": None}


def _live_running() -> bool:
    """判断推流是否仍在运行（handle 存在且未 stop）。"""
    h = _live_state.get("handle")
    if h is None:
        return False
    try:
        return bool(h["is_running"]())
    except Exception:
        return False


def _live_thread_snapshot() -> str:
    """返回当前 handle 中所有工作线程的快照文本，用于日志展示。"""
    h = _live_state.get("handle")
    if h is None:
        return ""
    threads = h.get("threads") or {}
    if not threads:
        return ""
    lines = ["工作线程："]
    for name, t in threads.items():
        try:
            lines.append(f"  - {name}: ident={t.ident}, alive={t.is_alive()}")
        except Exception:
            lines.append(f"  - {name}: <unknown>")
    return "\n".join(lines)


def run_live_inference_ui(
    live_dir,
    face_size,
    video_load_mode,
    audio_loop_mode,
    batch_size,
    sync_offset,
    frame_w,
    frame_h,
    port,
    cam_backend,
    reverse_random_prob,
    device,
):
    log_capture.clear()

    if _live_running():
        return ("推流正在运行中，请先点击「停止推流」。",
                gr.update(interactive=False), gr.update(interactive=True))

    if not live_dir or live_dir.startswith("(尚无直播预处理数据"):
        return ("请先选择直播预处理数据！",
                gr.update(interactive=True), gr.update(interactive=False))

    cfg = load_config()
    api_key = cfg.get("api_key", "").strip()
    if not api_key:
        return ("API Key 未配置，请先在『设置』中保存 API Key。",
                gr.update(interactive=True), gr.update(interactive=False))

    data_dir = str(DATA_LIVE_DIR / live_dir)
    if not os.path.isdir(data_dir):
        return (f"数据目录不存在: {data_dir}",
                gr.update(interactive=True), gr.update(interactive=False))

    face_size = int(face_size)
    human_path = f"./checkpoints/{face_size}.m.onnx"
    hubert_path = "./checkpoints/chinese-hubert-large/"
    if not os.path.exists(human_path):
        return (f"找不到 Human 模型: {human_path}",
                gr.update(interactive=True), gr.update(interactive=False))
    if not os.path.exists(hubert_path):
        return (f"找不到 Hubert 模型: {hubert_path}",
                gr.update(interactive=True), gr.update(interactive=False))

    try:
        from inference_live import InferenceLive

        frame_w = int(frame_w) if frame_w else None
        frame_h = int(frame_h) if frame_h else None
        port = int(port) if port else 8886

        # 在主回调线程构造实例，初始化报错可以立即反馈到界面
        print("[WebUI] 正在初始化 InferenceLive ...")
        # pyd 打包后的 InferenceLive 不再允许 vae_decoder_path=None，
        # 显式从 human_path 同目录推断 decoder 路径
        vae_decoder_path = f"./checkpoints/{face_size}.decoder.onnx"
        infer = InferenceLive(
            human_path=human_path,
            vae_decoder_path=vae_decoder_path,
            hubert_path=hubert_path,
            device=device,
            video_load_mode=video_load_mode,
            audio_loop_mode=audio_loop_mode,
            frame_w=frame_w,
            frame_h=frame_h,
            sync_offset=int(sync_offset),
            batch_size=int(batch_size),
            api_key=api_key,
            port=port,
            cam_backend=cam_backend,
            reverse_random_prob=float(reverse_random_prob),
        )

        # 用 block=False 启动，立刻拿到 handle；
        # 内部 4 个工作线程 + api_server 线程都登记在 handle["threads"]
        # handle["stop"]() / handle["wait"]() 是优雅停止入口
        handle = infer.start(data_dir=data_dir, block=False)
        _live_state["handle"] = handle

        snapshot = _live_thread_snapshot()
        print(f"[WebUI] 推流已启动: data_dir={data_dir}, /set_audio 端口={port}\n{snapshot}")
        msg = (
            f"推流已启动！\n数据目录: {data_dir}\n"
            f"虚拟摄像头已打开，/set_audio 服务端口: {port}\n"
            f"外部可 POST http://127.0.0.1:{port}/set_audio 切换音频。\n\n{snapshot}"
        )
        return (msg, gr.update(interactive=False), gr.update(interactive=True))
    except Exception as e:
        traceback.print_exc()
        _live_state["handle"] = None
        return (f"推流启动失败: {e}",
                gr.update(interactive=True), gr.update(interactive=False))


def stop_live_inference_ui():
    """优雅停止推流：handle.stop() → handle.wait()。"""
    h = _live_state.get("handle")
    if h is None or not _live_running():
        _live_state["handle"] = None
        return ("当前没有在运行的推流。",
                gr.update(interactive=True), gr.update(interactive=False))

    print("[WebUI] 收到停止推流请求，正在关闭...")
    try:
        h["stop"]()
    except Exception as e:
        print(f"[stop] stop() 异常: {e}")

    try:
        h["wait"](timeout=10)
    except Exception as e:
        print(f"[stop] wait() 异常: {e}")

    _live_state["handle"] = None
    print("[WebUI] 推流已停止。")
    return ("推流已停止。",
            gr.update(interactive=True), gr.update(interactive=False))


# ============== 任务 5：发送 /set_audio 请求 ==============
def send_set_audio_ui(audio, host, port, interrupt, is_save):
    """上传音频 → 暂存为无中文/空格路径 → POST /set_audio {"path":..., "interrupt":..., "is_save":...}"""
    if not audio:
        return "请先上传要发送的音频！"
    src = _resolve_uploaded_path(audio)
    if not src or not os.path.exists(src):
        return "音频文件不存在！"

    host = (host or "http://127.0.0.1").strip().rstrip("/")
    try:
        port = int(port) if port else 8886
    except Exception:
        return "端口必须是数字！"

    try:
        # 暂存音频到本地无中文/空格路径（推流端用 shell=True 调 ffmpeg，路径必须干净）
        stage_dir = Path(__file__).parent.resolve() / "temp" / "set_audio"
        stage_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(src).suffix or ".wav"
        staged = stage_dir / f"{uuid.uuid4().hex}{suffix}"
        shutil.copy(src, staged)

        url = f"{host}:{port}/set_audio"
        payload = json.dumps(
            {"path": str(staged), "interrupt": bool(interrupt), "is_save": bool(is_save)}).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"}, method="POST")
        print(f"[WebUI] 发送 /set_audio: {url}, path={staged}, interrupt={bool(interrupt)}")
        with urllib.request.urlopen(req, timeout=60) as resp:
            text = resp.read().decode("utf-8", "ignore")
        return f"发送成功 (HTTP {resp.status}): {text}"
    except Exception as e:
        return f"发送失败: {e}"


# ============== Gradio 界面 ==============
def build_ui():
    cfg = load_config()
    cuda_choices = list_cuda_devices()
    # 直播模式仅支持 CUDA
    cuda_only_choices = [d for d in cuda_choices if d.startswith("cuda")] or ["cuda"]

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

            # ----------- Tab 3：直播预处理 -----------
            with gr.Tab("③ 直播预处理"):
                gr.Markdown(
                    "### 上传视频 + 选择音频目录，生成直播推流数据（`data_live/<标题>/`）\n"
                    "> 视频会转成 25fps + 人脸检测 + VAE 编码；音频走 HuBERT 切片 + 44.1kHz 立体声 wav。"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        live_title_in = gr.Textbox(
                            label="📌 直播标题 (必填，输出目录为 data_live/<标题>/)",
                            placeholder="例如: live_demo",
                            max_lines=1,
                        )
                        live_video_in = gr.Video(
                            label="上传视频 (mp4 / avi)",
                            sources=["upload"],
                            height=280,
                        )
                        live_audios_in = gr.Textbox(
                            label="🎵 音频目录（本地路径，音频文件按文件名数字排序）",
                            placeholder="输入目录路径，或点击下方按钮选择",
                            max_lines=1,
                        )
                        live_audios_pick_btn = gr.Button("📂 选择音频目录")
                        live_face_size = gr.Radio(
                            choices=[256, 384],
                            value=256,
                            label="人脸尺寸 face_size (与模型匹配)",
                        )
                        live_res_preset = gr.Radio(
                            choices=["不压缩(原分辨率)", "480p", "720p", "1080p"],
                            value="1080p",
                            label="分辨率压缩档位",
                        )
                        live_frame_batch = gr.Slider(
                            minimum=4,
                            maximum=128,
                            value=64,
                            step=4,
                            label="每批处理帧数 (显存不足调小)",
                        )
                        live_sync_offset_pre = gr.Slider(
                            minimum=-20,
                            maximum=20,
                            value=0,
                            step=1,
                            label="音视频同步偏移 (帧)",
                        )
                        live_device_pre = gr.Radio(
                            choices=cuda_only_choices,
                            value=cuda_only_choices[0],
                            label="计算设备（直播模式仅支持 CUDA）",
                        )
                        live_pre_btn = gr.Button("🚀 开始直播预处理", variant="primary")

                    with gr.Column(scale=1):
                        live_pre_status = gr.Textbox(
                            label="直播预处理状态",
                            interactive=False,
                            lines=3,
                        )
                        gr.Markdown("#### 处理日志")
                        live_pre_log = gr.Textbox(
                            label="日志",
                            interactive=False,
                            lines=1,
                            autoscroll=True,
                            max_lines=20,
                            elem_classes=["log-box"],
                        )

            # ----------- Tab 4：直播推流 -----------
            with gr.Tab("④ 直播推流"):
                gr.Markdown(
                    "### 选择直播预处理数据，实时推理并推送到虚拟摄像头 + 扬声器\n"
                    "> 内置 FastAPI 服务（`POST /set_audio`）可接收外部音频切换播放；"
                    "> 停止推流会销毁后台线程并释放虚拟摄像头/音频。"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        live_dir_in = gr.Dropdown(
                            choices=list_live_data_dirs(),
                            label="选择直播预处理数据 (data_live/<标题>/)",
                            value=list_live_data_dirs()[0],
                            allow_custom_value=True,
                        )
                        live_refresh_btn = gr.Button("🔄 刷新列表")
                        live_face_size_inf = gr.Radio(
                            choices=[256, 384],
                            value=256,
                            label="人脸尺寸 (与预处理数据一致)",
                        )
                        live_video_load_mode = gr.Radio(
                            choices=["full", "streaming"],
                            value="full",
                            label="视频数据加载方式  full=全量 / streaming=流式",
                        )
                        live_audio_loop_mode = gr.Radio(
                            choices=["random", "sequential"],
                            value="random",
                            label="音频播放顺序  random=随机 / sequential=顺序",
                        )
                        live_batch = gr.Slider(
                            minimum=1,
                            maximum=512,
                            value=2,
                            step=1,
                            label="推理 batch_size (实时建议 1)",
                        )
                        live_sync_offset = gr.Slider(
                            minimum=-20,
                            maximum=20,
                            value=0,
                            step=1,
                            label="音视频同步偏移 (帧)",
                        )
                        with gr.Row():
                            live_frame_w = gr.Number(
                                label="虚拟摄像头宽度 (留空=按预处理数据)",
                                value=None,
                                precision=0,
                            )
                            live_frame_h = gr.Number(
                                label="虚拟摄像头高度 (留空=按预处理数据)",
                                value=None,
                                precision=0,
                            )
                        with gr.Row():
                            live_port = gr.Number(
                                label="/set_audio 服务端口",
                                value=8886,
                                precision=0,
                            )
                            live_cam_backend = gr.Radio(
                                choices=["obs", "unitycapture"],
                                value="obs",
                                label="虚拟摄像头后端",
                            )
                        live_reverse_prob = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0,
                            step=0.05,
                            label="随机反转人脸关键点概率",
                        )
                        live_device_inf = gr.Radio(
                            choices=cuda_only_choices,
                            value=cuda_only_choices[0],
                            label="计算设备（直播模式仅支持 CUDA）",
                        )
                        with gr.Row():
                            live_start_btn = gr.Button("🎥 开始推流", variant="primary")
                            live_stop_btn = gr.Button("⏹ 停止推流", variant="stop")

                    with gr.Column(scale=1):
                        live_infer_status = gr.Textbox(
                            label="推流状态",
                            interactive=False,
                            lines=4,
                        )
                        gr.Markdown("#### 推流日志")
                        live_infer_log = gr.Textbox(
                            label="日志",
                            interactive=False,
                            lines=1,
                            autoscroll=True,
                            max_lines=20,
                            elem_classes=["log-box"],
                        )

                # ----------- 独立块：发送 /set_audio 请求 -----------
                with gr.Group():
                    gr.Markdown(
                        "#### 📤 发送 /set_audio 请求（向运行中的推流切换播放音频）"
                    )
                    with gr.Row():
                        set_audio_file = gr.Audio(
                            label="上传音频 (wav / mp3)",
                            sources=["upload"],
                            type="filepath",
                        )
                    with gr.Row():
                        set_audio_host = gr.Textbox(
                            label="服务地址",
                            value="http://127.0.0.1",
                        )
                        set_audio_port = gr.Number(
                            label="端口",
                            value=8886,
                            precision=0,
                        )
                        set_audio_interrupt = gr.Checkbox(
                            label="打断当前播放 (interrupt)",
                            value=True,
                        )
                        set_audio_is_save = gr.Checkbox(
                            label="保存 wav + npy 到数据目录 (is_save)",
                            value=False,
                        )
                        set_audio_btn = gr.Button("📨 发送 POST 请求", variant="secondary")
                    set_audio_status = gr.Textbox(
                        label="发送结果",
                        interactive=False,
                        lines=2,
                    )

            # ----------- Tab 5：使用说明 -----------
            with gr.Tab("📖 使用说明"):
                gr.Markdown(
                    f"""
                    ### 流程
                    1. **⚙️ 设置**：填入 API Key、选择推理设备，点击「保存配置」（保存到 `{CONFIG_PATH}`）。
                    2. **① 视频预处理**：填任务标题（将作为 `data/<标题>/` 目录名），上传视频，开始预处理。
                    3. **② 音频驱动推理**：选择 `data/<标题>/`，上传音频，开始推理。

                    ### 直播流程
                    4. **③ 直播预处理**：填直播标题（输出到 `data_live/<标题>/`），上传视频并选择音频目录，开始预处理。
                    5. **④ 直播推流**：选择直播预处理数据，点「开始推流」实时输出到虚拟摄像头 + 扬声器；
                       点「停止推流」销毁后台线程并释放资源。
                    6. **发送 /set_audio**：推流页底部独立块，上传音频并填写服务地址/端口/是否打断，
                       点「发送 POST 请求」即可向运行中的推流切换播放音频。

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

        live_audios_pick_btn.click(
            fn=pick_folder,
            inputs=[live_audios_in],
            outputs=live_audios_in,
        )

        live_pre_btn.click(
            fn=run_live_preprocess_ui,
            inputs=[
                live_title_in,
                live_video_in,
                live_audios_in,
                live_face_size,
                live_frame_batch,
                live_res_preset,
                live_sync_offset_pre,
                live_device_pre,
            ],
            outputs=[live_pre_status, live_dir_in],
        )

        live_start_btn.click(
            fn=run_live_inference_ui,
            inputs=[
                live_dir_in,
                live_face_size_inf,
                live_video_load_mode,
                live_audio_loop_mode,
                live_batch,
                live_sync_offset,
                live_frame_w,
                live_frame_h,
                live_port,
                live_cam_backend,
                live_reverse_prob,
                live_device_inf,
            ],
            outputs=[live_infer_status, live_start_btn, live_stop_btn],
        )

        live_stop_btn.click(
            fn=stop_live_inference_ui,
            outputs=[live_infer_status, live_start_btn, live_stop_btn],
        )

        live_refresh_btn.click(
            list_live_data_dirs_ui,
            outputs=live_dir_in,
        )

        set_audio_btn.click(
            fn=send_set_audio_ui,
            inputs=[set_audio_file, set_audio_host, set_audio_port, set_audio_interrupt, set_audio_is_save],
            outputs=set_audio_status,
        )

        # ----------- 页面加载时刷新预处理列表（避免 Timer.tick 在某些环境下导致组件渲染异常） -----------
        demo.load(
            list_preprocess_dirs_ui,
            outputs=preprocess_dir_in,
        )
        demo.load(
            list_live_data_dirs_ui,
            outputs=live_dir_in,
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
            fn=lambda: (log_capture.get_text(), log_capture.get_text(),
                        log_capture.get_text(), log_capture.get_text()),
            outputs=[pre_log, infer_log, live_pre_log, live_infer_log],
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

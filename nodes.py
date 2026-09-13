from __future__ import annotations

import base64
import io as bytes_io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import torch
import torchaudio
from comfy_api.v0_0_2 import ComfyExtension, io


PROTOCOL_VERSION = "1.0"
AUK_LOCAL_CONNECTION = io.Custom("AUK_LOCAL_CONNECTION")
TASK_OPTIONS = [
    "描述生成语音",
    "参考声音克隆",
    "语音文字编辑",
    "歌词编辑",
    "音高编辑",
    "速度编辑",
    "音量编辑",
    "情绪编辑",
    "音色编辑",
    "去口音",
    "非语言声音编辑",
    "耳语转换",
    "语音增强",
    "说话人分离",
    "音乐人声提取",
    "指定说话人提取",
]
TASK_KEYS = [
    "instruct_tts",
    "zero_shot_tts",
    "content_edit",
    "lyric_edit",
    "pitch",
    "speed",
    "volume",
    "emotion",
    "timbre",
    "deaccent",
    "nonverbal",
    "whisper",
    "enhance",
    "speech_separate",
    "music_separate",
    "target_speaker",
]


class HttpResponseError(RuntimeError):
    def __init__(self, status_code: int, detail: str):
        super().__init__(f"AuK Local HTTP {status_code}: {detail}")
        self.status_code = status_code


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "AuK Local 不接受 HTTP 重定向", headers, fp)


def validate_loopback_url(base_url: str) -> str:
    candidate = str(base_url or "").rstrip("/")
    try:
        parsed = urllib.parse.urlsplit(candidate)
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("AuK Local 服务地址无效") from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("AuK Local 服务地址只能是本机 loopback HTTP 地址")
    return candidate


def resolve_token_file(setting: str) -> Path:
    if setting.strip():
        return Path(os.path.expandvars(setting)).expanduser().resolve()
    environment = os.environ.get("AUK_LOCAL_TOKEN_FILE")
    if environment:
        return Path(environment).expanduser().resolve()
    home = os.environ.get("AUK_LOCAL_HOME")
    if home:
        return Path(home).expanduser().resolve() / "data" / "session-token"
    config_file = Path(__file__).with_name("auk-local-config.json")
    if config_file.is_file():
        config = json.loads(config_file.read_text(encoding="utf-8"))
        configured = Path(config["token_file"]).expanduser()
        if not configured.is_absolute():
            configured = config_file.parent / configured
        return configured.resolve()
    raise ValueError("找不到 AuK Local 服务令牌；请运行整合包里的“安装ComfyUI节点.cmd”")


class Client:
    def __init__(self, base_url: str, token_file: Path):
        self.base_url = validate_loopback_url(base_url)
        self.token_file = token_file
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json", "X-AuK-Token": self.token_file.read_text(encoding="utf-8").strip()}

    def json_request(self, method: str, path: str, payload: dict[str, Any] | None = None, timeout: float = 30):
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None,
            method=method,
            headers=self._headers(),
        )
        try:
            with self._opener.open(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise HttpResponseError(exc.code, detail) from exc

    def health(self):
        with self._opener.open(self.base_url + "/api/v1/health", timeout=5) as response:
            health = json.loads(response.read().decode("utf-8"))
        if str(health.get("protocol_version", "")).split(".")[0] != PROTOCOL_VERSION.split(".")[0]:
            raise RuntimeError(f"协议不兼容：节点 {PROTOCOL_VERSION}，服务 {health.get('protocol_version')}")
        return health

    def download(self, path: str, timeout: float = 60) -> bytes:
        request = urllib.request.Request(self.base_url + path, headers=self._headers())
        with self._opener.open(request, timeout=timeout) as response:
            return response.read()


def submit_with_recovery(client: Client, request_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    last_error: BaseException | None = None
    for attempt in range(3):
        try:
            return client.json_request("POST", "/api/v1/tasks", payload)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last_error = exc
            try:
                return client.json_request("GET", f"/api/v1/tasks/{request_id}", timeout=10)
            except HttpResponseError as status_error:
                if status_error.status_code != 404:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
                pass
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"提交 AuK 任务后无法确认服务状态：{last_error}") from last_error


class AuKLocalConnection(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AuKLocalConnection",
            display_name="AuK Local 连接",
            category="AuK Local",
            description="连接独立的 AuK Local 本机服务；不会在 ComfyUI 进程加载模型。",
            inputs=[
                io.String.Input("service_url", default="http://127.0.0.1:7860"),
                io.String.Input(
                    "token_file",
                    default="",
                    optional=True,
                    advanced=True,
                    tooltip="留空时读取安装脚本生成的本地配置；工作流不保存令牌内容。",
                ),
                io.Combo.Input("model", options=["flash", "base"], default="flash"),
                io.Boolean.Input("cpu_offload", default=True, optional=True),
                io.Boolean.Input("keep_loaded", default=False, optional=True, advanced=True),
            ],
            outputs=[AUK_LOCAL_CONNECTION.Output("connection", display_name="连接")],
        )

    @classmethod
    def execute(
        cls,
        service_url: str,
        token_file: str = "",
        model: str = "flash",
        cpu_offload: bool = True,
        keep_loaded: bool = False,
    ) -> io.NodeOutput:
        token_path = resolve_token_file(token_file)
        client = Client(service_url, token_path)
        client.health()
        return io.NodeOutput(
            {
                "service_url": service_url.rstrip("/"),
                "token_file": str(token_path),
                "model": model,
                "cpu_offload": bool(cpu_offload),
                "keep_loaded": bool(keep_loaded),
            }
        )


def normalize_audio(audio: dict | None) -> tuple[dict[str, Any] | None, float]:
    if audio is None:
        return None, 0.0
    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if not torch.is_tensor(waveform) or waveform.ndim != 3 or waveform.shape[0] != 1:
        raise ValueError("输入 AUDIO 必须是 [1, channels, samples]")
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("输入采样率无效")
    waveform = waveform[0].detach().to(device="cpu", dtype=torch.float32)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if waveform.numel() == 0 or not torch.isfinite(waveform).all():
        raise ValueError("输入音频为空或含 NaN/Inf")
    mono = waveform.squeeze(0).contiguous().numpy().astype("<f4", copy=False)
    return (
        {
            "encoding": "f32le",
            "sample_rate": sample_rate,
            "channels": 1,
            "frames": int(mono.size),
            "data": base64.b64encode(mono.tobytes()).decode("ascii"),
        },
        mono.size / sample_rate,
    )


def check_interrupted() -> None:
    try:
        from comfy import model_management

        model_management.throw_exception_if_processing_interrupted()
    except ModuleNotFoundError:
        return


class AuKLocalGenerateEdit(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AuKLocalGenerateEdit",
            display_name="AuK Local 生成 / 编辑",
            category="AuK Local",
            description="通过本机服务执行 AuK 的16类生成、编辑、增强和分离任务。",
            inputs=[
                AUK_LOCAL_CONNECTION.Input("connection"),
                io.Combo.Input("task", options=TASK_OPTIONS, default=TASK_OPTIONS[0]),
                io.String.Input("primary", display_name="主要内容", multiline=True, default="你好，欢迎使用 AuK。"),
                io.String.Input("secondary", display_name="声音描述 / 附加要求", multiline=True, default="自然、清晰、温暖"),
                io.Float.Input(
                    "generation_seconds",
                    default=3.0,
                    min=0.2,
                    max=30.0,
                    step=0.1,
                    display_mode=io.NumberDisplay.slider,
                ),
                io.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=0x7FFFFFFFFFFFFFFF,
                    control_after_generate=io.ControlAfterGenerate.fixed,
                ),
                io.Audio.Input("input_audio", optional=True),
                io.Int.Input(
                    "refresh",
                    default=0,
                    min=0,
                    max=0x7FFFFFFF,
                    control_after_generate=io.ControlAfterGenerate.increment,
                    advanced=True,
                    tooltip="递增时强制创建新任务；不影响模型随机种子。",
                ),
                io.Int.Input("nfe_steps", default=32, min=4, max=64, advanced=True),
                io.Float.Input("cfg_strength", default=2.0, min=0.0, max=5.0, step=0.1, advanced=True),
                io.Float.Input("sway_sampling_coef", default=-1.0, min=-1.0, max=1.0, step=0.1, advanced=True),
            ],
            outputs=[
                io.Audio.Output("generated_audio", display_name="生成音频"),
                io.String.Output("instruction", display_name="最终指令"),
                io.String.Output("metadata", display_name="运行参数 JSON"),
            ],
        )

    @classmethod
    def execute(
        cls,
        connection: dict[str, Any],
        task: str,
        primary: str,
        secondary: str,
        generation_seconds: float,
        seed: int,
        input_audio: dict | None = None,
        refresh: int = 0,
        nfe_steps: int = 32,
        cfg_strength: float = 2.0,
        sway_sampling_coef: float = -1.0,
    ) -> io.NodeOutput:
        if task not in TASK_OPTIONS:
            raise ValueError(f"未知任务：{task}")
        task_key = TASK_KEYS[TASK_OPTIONS.index(task)]
        model = connection["model"]
        if model == "flash":
            nfe_steps, cfg_strength, sway_sampling_coef = 4, 0.0, -1.0
        encoded_audio, source_seconds = normalize_audio(None if task_key == "instruct_tts" else input_audio)
        if source_seconds + float(generation_seconds) > 30.0 + 1e-9:
            raise ValueError(
                f"输入 {source_seconds:.2f}s + 输出 {generation_seconds:.2f}s 超过 30s 限制"
            )
        client = Client(connection["service_url"], Path(connection["token_file"]))
        client.health()
        payload = {
            "task_key": task_key,
            "primary": primary,
            "secondary": secondary,
            "generation_seconds": float(generation_seconds),
            "model": model,
            "seed": int(seed),
            "nfe_steps": int(nfe_steps),
            "cfg_strength": float(cfg_strength),
            "sway_sampling_coef": float(sway_sampling_coef),
            "cpu_offload": connection["cpu_offload"],
            "keep_loaded": connection["keep_loaded"],
            "client": "comfyui",
            "audio": encoded_audio,
        }
        # One node execution owns one remote task. Network retries below keep this
        # ID, while separate workflow executions cannot cancel or reuse each other.
        request_id = str(uuid.uuid4())
        payload["request_id"] = request_id
        try:
            submitted = submit_with_recovery(client, request_id, payload)
            disconnected_at = None
            while submitted["state"] not in {"succeeded", "failed", "cancelled", "interrupted"}:
                scheduler = submitted.get("scheduler") or {}
                if (
                    scheduler.get("state") in {"paused", "stopping", "stopped"}
                    or scheduler.get("dispatcher_alive") is False
                ):
                    detail = scheduler.get("error") or "任务调度不可用"
                    raise RuntimeError(
                        f"AuK Local 任务 {request_id} 已暂停：{detail}。请检查服务诊断，恢复后重启服务并重试。"
                    )
                check_interrupted()
                time.sleep(0.4)
                try:
                    submitted = client.json_request("GET", f"/api/v1/tasks/{request_id}", timeout=10)
                    disconnected_at = None
                except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                    disconnected_at = disconnected_at or time.monotonic()
                    if time.monotonic() - disconnected_at > 30:
                        raise RuntimeError("AuK Local 服务断线超过 30 秒") from exc
        except BaseException:  # Comfy interrupts also require remote task cancellation
            try:
                client.json_request("POST", f"/api/v1/tasks/{request_id}/cancel", {}, timeout=5)
            except Exception:  # noqa: BLE001, S110 - preserve the original failure if service cancellation fails
                pass
            raise
        if submitted["state"] != "succeeded":
            raise RuntimeError(f"AuK任务{submitted['state']}：{submitted.get('error') or ''}")
        wav_bytes = client.download(f"/api/v1/tasks/{request_id}/audio")
        waveform, sample_rate = torchaudio.load(bytes_io.BytesIO(wav_bytes))
        metadata = client.json_request("GET", f"/api/v1/tasks/{request_id}/metadata")
        return io.NodeOutput(
            {"waveform": waveform.unsqueeze(0).to(torch.float32).cpu(), "sample_rate": int(sample_rate)},
            metadata["instruction"],
            json.dumps(metadata, ensure_ascii=False, indent=2),
        )


class AuKLocalExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [AuKLocalConnection, AuKLocalGenerateEdit]


async def comfy_entrypoint() -> AuKLocalExtension:
    return AuKLocalExtension()

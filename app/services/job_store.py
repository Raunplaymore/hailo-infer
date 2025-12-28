import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from app.core.config import Settings


@dataclass
class JobInfo:
    job_id: str
    status: str
    mode: str
    result_path: Path
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    canceled: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class JobStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._jobs: Dict[str, JobInfo] = {}
        self._lock = threading.Lock()

    def _analysis_dir(self) -> Path:
        return self._settings.data_dir / "analysis"

    def result_path(self, job_id: str) -> Path:
        return self._analysis_dir() / f"{job_id}.json"

    def get(self, job_id: str) -> Optional[JobInfo]:
        with self._lock:
            return self._jobs.get(job_id)

    def upsert(self, info: JobInfo) -> None:
        with self._lock:
            self._jobs[info.job_id] = info

    def set_status(self, job_id: str, status: str, error_code: Optional[str] = None, error_message: Optional[str] = None) -> None:
        info = self.get(job_id)
        if not info:
            return
        with info.lock:
            info.status = status
            info.error_code = error_code
            info.error_message = error_message

    def mark_canceled(self, job_id: str) -> bool:
        info = self.get(job_id)
        if not info:
            return False
        with info.lock:
            info.canceled = True
            info.status = "canceled"
        return True

    def is_canceled(self, job_id: str) -> bool:
        info = self.get(job_id)
        if not info:
            return False
        with info.lock:
            return info.canceled

    def load_cached_result(self, job_id: str) -> Optional[dict]:
        path = self.result_path(job_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return None

    def save_result(self, job_id: str, result: dict) -> None:
        self._analysis_dir().mkdir(parents=True, exist_ok=True)
        path = self.result_path(job_id)
        path.write_text(json.dumps(result, ensure_ascii=True, indent=2))

    def init_job(self, job_id: str, mode: str, status: str) -> JobInfo:
        info = JobInfo(job_id=job_id, status=status, mode=mode, result_path=self.result_path(job_id))
        self.upsert(info)
        return info

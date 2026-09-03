import os
from pathlib import Path


class Settings:
    def __init__(self) -> None:
        self.port = int(os.getenv("PORT", "3002"))
        self.upload_dir = Path(os.getenv("UPLOAD_DIR", "/home/ray/uploads"))
        self.meta_dir = Path(os.getenv("META_DIR", "/tmp"))
        self.data_dir = Path(os.getenv("DATA_DIR", "/home/ray/data"))
        self.camera_base_url = os.getenv("CAMERA_BASE_URL", "http://127.0.0.1:3001")
        self.hailo_hef_path = os.getenv("HAILO_HEF_PATH", "")
        self.body_pose_target_fps = max(0.0, float(os.getenv("BODY_POSE_TARGET_FPS", "0")))

    @property
    def hailo_available(self) -> bool:
        if not self.hailo_hef_path:
            return False
        return Path(self.hailo_hef_path).exists()

# hailo-infer

추론 결과 기반 스윙 코칭 연산 전담 Python 마이크로서비스입니다.

## 역할

- Hailo 추론 메타 기반 스윙 이벤트 분할
- 코칭 지표 산출
- 분석 결과 JSON 생성

`hailo-camera`는 GStreamer + hailonet으로 `.meta.json`을 생성하며,
본 서비스는 메타 기반 분석(A)을 우선으로 수행합니다.
필요 시 영상 → 메타 생성(B, 선택 기능)을 제공합니다.

## 개발 전제

- 로컬 macOS에서 Python venv 기반 개발
- Raspberry Pi 전용 의존성(Hailo SDK/HailoRT)은 로컬에서 필요하지 않음

## 실행

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 3002
```

## 환경 변수

- `PORT=3002`
- `UPLOAD_DIR=/home/ray/uploads`
- `META_DIR=/tmp`
- `DATA_DIR=/home/ray/data`
- `CAMERA_BASE_URL=http://127.0.0.1:3001`
- 선택: `HAILO_HEF_PATH=/usr/share/hailo-models/yolov8s_h8.hef`

## API

### Health Check

`GET /health`

### Job 생성

`POST /v1/jobs`

```json
{
  "mode": "coach_from_meta",
  "jobId": "session_YYYY",
  "source": {
    "filename": "session_YYYY.mp4",
    "videoPath": "/home/ray/uploads/session_YYYY.mp4",
    "metaPath": "/tmp/session_YYYY.meta.json"
  },
  "options": {
    "force": false,
    "tailFramesForLive": 30
  }
}
```

### 상태/취소/결과

- `GET /v1/jobs/{jobId}`
- `POST /v1/jobs/{jobId}/cancel`
- `GET /v1/jobs/{jobId}/result`

## 결과 저장

`DATA_DIR/analysis/<jobId>.json`

## 배포

GitHub Actions로 Raspberry Pi에 배포합니다.

필수 시크릿:

- `PI_HOST`
- `PI_USER`
- `PI_PATH`
- `PI_SSH_KEY`

systemd 서비스는 `systemd/hailo-infer.service` 참고.

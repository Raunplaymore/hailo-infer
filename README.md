# hailo-infer

추론 결과 기반 스윙 코칭 연산 전담 Python 마이크로서비스입니다.

## 역할

- Hailo 추론 메타 기반 스윙 이벤트 분할
- 코칭 지표 산출
- 분석 결과 JSON 생성
- `pi_service`는 `INFER_BASE_URL`로 본 서비스를 호출합니다.
- 업로드 body bootstrap 단계는 `BODY_ANALYZER_BASE_URL`로 같은 서비스의 `/v1/body/from-video`를 호출할 수 있습니다.

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
- 선택: `BODY_POSE_TARGET_FPS=30` — body pose를 원본 FPS 기준 약 30fps로 분석합니다. 기본값 `0`은 기존 frame-count sampling입니다.
- 선택: `VISIBLE_GRIP_TRACK_ENABLED=1` — dominant visible wrist identity를 고정하고 가까운 club handle을 corroboration으로 사용합니다. 기본값 `0`은 기존 wrist track입니다.

두 pose refresh 옵션은 서로 독립적으로 되돌릴 수 있습니다. 운영 문제 발생 시 두 값을 각각 `0`으로
설정하고 서비스를 재시작하면 코드 rollback 없이 기존 동작으로 복귀합니다. 가려진 손목의 장기
구간은 생성하거나 보간하지 않으며, 양손 관측이 필요한 지표만 별도로 보류합니다.

## API

### Health Check

`GET /health`

systemd 헬스체크 타이머가 이 엔드포인트를 주기적으로 확인합니다.

### Job 생성

`POST /v1/jobs`

```json
{
  "mode": "coach_from_meta",
  "jobId": "session_YYYY",
  "source": {
    "filename": "session_YYYY.mp4",
    "videoPath": "/home/ray/uploads/session_YYYY.mp4",
    "metaPath": "/tmp/session_YYYY.meta.json",
    "bodyPath": "/home/ray/data/body/session_YYYY.json"
  },
  "options": {
    "force": false,
    "tailFramesForLive": 30
  }
}
```

`bodyPath`가 있으면 body artifact의 pose wrist keypoints를 top 이벤트 산정에 우선 사용합니다.
`mediapipe`를 사용할 수 없는 환경에서는 기존 OpenCV HOG body bootstrap만 생성되고, 이벤트는 club
trajectory fallback으로 계산됩니다.

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
- `TAILSCALE_AUTHKEY` (Tailscale 사용 시)

systemd 서비스는 `systemd/hailo-infer.service` 참고.

### systemd 설치/부팅 자동 복구

Pi에서 최초 1회 설정:

```bash
sudo cp /home/ray/hailo-infer/systemd/hailo-infer.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now hailo-infer
```

헬스체크 타이머(자동 재시작 포함):

```bash
sudo cp /home/ray/hailo-infer/systemd/hailo-infer-healthcheck.service /etc/systemd/system/
sudo cp /home/ray/hailo-infer/systemd/hailo-infer-healthcheck.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now hailo-infer-healthcheck.timer
```

### venv 주의사항

`/home/ray/hailo-infer/.venv`가 없으면 `ExecStart`가 실패합니다. 배포 후 venv를 보장하세요:

```bash
cd /home/ray/hailo-infer
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

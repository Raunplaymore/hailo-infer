# hailo-infer Context

## 프로젝트 개요
Python FastAPI 기반 스윙 코칭 분석 마이크로서비스. Hailo 메타 기반 이벤트 분할 및 코칭 지표 산출.

## 자주 수정하는 파일

### 핵심 로직
- `app/main.py` - FastAPI 앱, 라우팅, Job 관리
- `app/services/coach_pipeline.py` - 코칭 분석 파이프라인
- `app/services/meta_loader.py` - 메타 파일 로딩

### 데이터 관리
- `app/services/job_store.py` - Job 상태 저장소
- `app/schemas.py` - Pydantic 스키마

### 설정
- `app/core/config.py` - 환경변수 설정

## 분석 파이프라인

### 입력
```json
{
  "mode": "coach_from_meta",
  "jobId": "session_123",
  "source": {
    "filename": "session_123.mp4",
    "videoPath": "/home/ray/uploads/session_123.mp4",
    "metaPath": "/tmp/session_123.meta.json"
  },
  "options": {
    "force": false
  }
}
```

### 처리 순서
1. 메타 파일 로드 (`load_meta()`)
2. 프레임 배열 파싱
3. 이벤트 분할 (`analyze_meta()`)
   - Address (t=0)
   - Top (백스윙 정점)
   - Impact (임팩트)
   - Finish (피니시)
4. 메트릭 계산
   - Swing Plane (inside-out/square/outside-in)
   - Tempo (backswing/downswing ratio)
   - Impact Stability (안정성 점수)
5. 요약 생성

### 출력
```json
{
  "ok": true,
  "jobId": "session_123",
  "status": "succeeded",
  "events": {
    "addressMs": 0,
    "topMs": 820,
    "impactMs": 1220,
    "finishMs": 1600
  },
  "metrics": {
    "swingPlane": { "label": "inside-out", "confidence": 0.62 },
    "tempo": { "backswingMs": 820, "downswingMs": 400, "ratio": 2.05 },
    "impactStability": { "label": "stable", "score": 0.74 }
  },
  "summary": "Swing plane inside-out. Impact stability stable. Tempo 2.05:1.",
  "meta": {
    "fps": 60,
    "width": 1456,
    "height": 1088,
    "durationMs": 2000,
    "analysisVersion": "hailo-coach-v1"
  }
}
```

## 주의사항

### 메타 파일 형식
```json
{
  "frames": [
    {
      "t": 1766000000000,
      "frame": 123,
      "detections": [
        { "label": "golf_ball", "classId": 0, "conf": 0.9, "bbox": [x, y, w, h] },
        { "label": "clubhead", "classId": 1, "conf": 0.85, "bbox": [x, y, w, h] },
        { "label": "person", "classId": 66, "conf": 0.92, "bbox": [x, y, w, h] }
      ]
    }
  ]
}
```

### 캐싱
- 결과 저장: `DATA_DIR/analysis/<jobId>.json`
- `force=true`로 캐시 무시 가능
- 실패한 결과도 캐싱 (재분석 방지)

### 에러 처리
- `CoachError` - 분석 로직 에러 (errorCode 포함)
- `MetaLoadError` - 메타 파일 로드 실패
- 예외 발생 시 `status=failed` + 에러 정보 저장

## 환경변수

### 필수
- `UPLOAD_DIR=/home/ray/uploads` - 비디오 파일 경로
- `META_DIR=/tmp` - 메타 파일 경로
- `DATA_DIR=/home/ray/data` - 분석 결과 저장 경로

### 선택
- `PORT=3002` - 서버 포트
- `CAMERA_BASE_URL=http://127.0.0.1:3001` - 카메라 서버 주소
- `HAILO_HEF_PATH=/usr/share/hailo-models/yolov8s_h8.hef` - Hailo 모델 (미사용)

## API 엔드포인트

### Health Check
- `GET /health`
```json
{
  "ok": true,
  "version": "0.1.0",
  "hailoAvailable": false
}
```

### Job 생성
- `POST /v1/jobs`
- Request: `JobCreateRequest`
- Response: `JobCreateResponse`

### Job 상태
- `GET /v1/jobs/{jobId}`
- Response: `JobStatusResponse`

### Job 취소
- `POST /v1/jobs/{jobId}/cancel`

### Job 결과
- `GET /v1/jobs/{jobId}/result`
- 완료 시 전체 분석 결과 반환
- 진행 중/실패 시 상태만 반환

## 디버깅 팁

### 메타 파일 로드 실패
1. 파일 존재 확인: `ls -la /tmp/<jobId>.meta.json`
2. JSON 포맷 확인: `cat /tmp/<jobId>.meta.json | jq`
3. 권한 확인: 읽기 권한 필요

### 이벤트 검출 실패
1. `coach_pipeline.py` 로그 확인
2. 메타에 충분한 프레임 있는지 확인
3. detection 라벨 확인 (golf_ball, clubhead, person)

### 의존성 에러
1. venv 활성화 확인: `which python`
2. 패키지 설치 확인: `pip list`
3. requirements.txt 재설치

### uvicorn 시작 실패
1. 포트 충돌 확인: `lsof -i :3002`
2. venv 경로 확인
3. import 에러 확인

## 알려진 제약사항

### Hailo 의존성
- 로컬(macOS) 개발 시 Hailo SDK 불필요
- `infer_meta_from_video` 모드는 Raspberry Pi 전용
- `coach_from_meta` 모드만 로컬 개발 가능

### 분석 정확도
- 단일 카메라 DTL 촬영 전제
- 정밀 물리값(스핀/캐리) 계산 불가
- 휴리스틱 기반: 검출 실패 시 `null` 반환

### Job 저장소
- 현재 메모리 기반 (재시작 시 유실)
- 결과는 파일로 영구 저장
- 추후 DB 연동 필요

## 코딩 컨벤션

### 에러 처리
```python
try:
    result = analyze_meta(meta)
except CoachError as exc:
    store.set_status(job_id, "failed",
                     error_code=exc.code,
                     error_message=str(exc))
```

### 백그라운드 태스크
- FastAPI `BackgroundTasks` 사용
- Job 상태 업데이트: `queued` → `running` → `succeeded`/`failed`
- 예외 발생 시 반드시 `failed` 상태로 전환

### 타입 힌트
- Pydantic 모델 사용
- 모든 함수에 타입 힌트 명시
- `Optional[T]` 명시적 사용

## 개발 워크플로

### 로컬 개발
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 3002
```

### 테스트
```bash
# Health check
curl http://localhost:3002/health

# Job 생성
curl -X POST http://localhost:3002/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"mode":"coach_from_meta","jobId":"test","source":{...},"options":{}}'

# 상태 확인
curl http://localhost:3002/v1/jobs/test

# 결과 조회
curl http://localhost:3002/v1/jobs/test/result
```

### 배포 (Raspberry Pi)
- GitHub Actions 자동 배포
- systemd 서비스 관리: `systemd/hailo-infer.service`
- 헬스체크 타이머: `systemd/hailo-infer-healthcheck.timer`

## 추후 개선 사항
- [ ] Job 큐 고도화 (Redis/Celery)
- [ ] DB 연동 (Job 상태 영구 저장)
- [ ] 분석 정확도 개선 (ML 모델 업그레이드)
- [ ] 다중 카메라 지원 검토

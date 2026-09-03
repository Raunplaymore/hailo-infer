# Pose and Technology Refresh Refactoring Plan

- 문서 상태: Active
- 작성일: 2026-09-03
- 대상 서비스: `hailo-infer` 중심, 필요 시 `hailo-model`, `hailo-back`, `hailo-front`, Pi runtime
- 운영 원칙: 검증되지 않은 지표를 더 많이 표시하는 것보다 근거가 약한 지표를 보류하는 것을 우선한다.
- 진행 기록: [pose-technology-refresh-progress.md](pose-technology-refresh-progress.md)

## 1. 목적

이 계획은 프로젝트 초기에 선택한 pose, video, runtime 및 dependency 구성을 현재 기준으로 다시
평가하고, 동일 영상 반복 분석에서도 변하지 않는 골프 스윙 지표를 만드는 것을 목적으로 한다.

목표는 최신 라이브러리를 무조건 적용하는 것이 아니다. 다음 네 조건을 기존보다 잘 만족하는 후보만
채택한다.

1. 동일 입력에 대한 재현성
2. 골프 동작에 필요한 관절·클럽 관측 정확도
3. Raspberry Pi 5와 Hailo-8에서의 처리시간 및 자원 사용량
4. 근거가 부족할 때 안전하게 `withheld`하는 능력

## 2. 확인된 현재 상태

### 2.1 운영 환경

| 항목 | 현재 상태 |
| --- | --- |
| 보드 | Raspberry Pi 5 16GB |
| 가속기 | Hailo-8, firmware/HailoRT 4.23.0 |
| OS | Raspberry Pi OS Bookworm, Linux 6.12 계열 |
| Python | 3.11.2 |
| Node.js | 18.20.4 |
| FFmpeg | 5.1.8 |
| Pose | MediaPipe 0.10.14 legacy `mp.solutions.pose.Pose` |
| OpenCV | `opencv-python` 4.12와 `opencv-contrib-python` 4.13 동시 설치 |
| Front build | React 19, Vite 7 |
| Local LLM | llama.cpp 0.3.0 + Qwen3 4B Q4_K_M |

### 2.2 현재 pose 처리

- `BODY_SAMPLE_TARGET = 24`
- 약 197프레임 영상은 stride 8로 25프레임만 처리한다.
- MediaPipe는 `static_image_mode=False`, model complexity 1로 실행한다.
- 어깨/골반 회전은 좌우 keypoint를 잇는 선의 `atan2` 각도 차이로 계산한다.
- 계산 window가 address/top/impact event에 의존한다.
- pose coverage는 샘플 25개 중 keypoint가 존재하는 비율이며 전체 프레임 추적 품질이 아니다.

### 2.3 재현성 증거

최신 `IMG_5755.mov`와 2026-08-30 업로드는 container hash는 다르지만 다음이 동일하다.

- H.264, 1920x1080, 60fps
- duration 3.276667초
- 197 video frames
- 디코딩한 전체 frame MD5 stream의 SHA-256 동일

그런데 같은 분석 버전 `hailo-coach-service7-v15`에서 다음 차이가 발생했다.

| 지표 | 이전 | 최신 |
| --- | ---: | ---: |
| top | 2129ms | 1730ms |
| impact | 2528ms | 2262ms |
| tempo | 2.93 | 1.47 |
| shaft angle | 42.5° | 63.3° |
| backswing | low_top | adequate |
| shoulder proxy | 7.8° | 173.6° |
| hip proxy | 13.4° | 167.1° |

이는 플레이어 변화가 아니라 분석 재현성 문제다.

## 3. 필수 범위

아래 항목은 제품의 사실성 확보에 필요하며 옵션 모델 평가보다 먼저 수행한다.

### M1. 잘못된 관절 코칭 차단

목표:

- 불안정한 shoulder/hip/head proxy가 확정 코칭 finding으로 승격되지 않게 한다.
- API 필드 구조는 유지한다.

변경 후보:

- `app/services/coach_commentary.py`
- `app/services/coach_pipeline.py`
- `app/services/result_completion.py`

정책:

- shoulder/hip proxy는 검증 완료 전 `reference` 또는 `withheld`
- LLM consumer context에서 검증되지 않은 body finding 제외
- 기존 값은 debug/research payload에 보존 가능

통과 기준:

- 낮은 confidence 또는 temporal quality 미달 입력에서 body correction finding 0개
- 기존 API consumer가 schema 오류 없이 동작
- withheld 이유 code가 결과에 존재

rollback:

- body evidence gate commit만 revert 가능해야 한다.

### M2. 축 각도와 좌우 identity 수정

원인:

어깨선과 골반선은 방향 벡터가 아니라 축이다. 따라서 `θ`와 `θ + 180°`는 같은 자세다. 기존
360° 차이는 좌우 keypoint 순서 변화 또는 선 방향 변화에 의해 작은 회전을 170° 이상으로 오인한다.

구현:

- axial difference: 결과 범위를 0~90°로 제한
- frame sequence에서 180° periodic unwrap
- left/right swap 후보 감지 및 continuity가 작은 orientation 선택
- visibility threshold 미달 frame 제외
- 직전 유효 frame 대비 비현실적 점프를 quality error로 기록
- raw angle과 corrected axial angle을 debug에 별도 저장

변경 후보:

- `app/services/coach_pipeline.py`
- 필요 시 `app/services/body_pipeline.py`
- 신규 `scripts/check_body_pose_metrics.py`

통과 기준:

- `7.8°`와 `173.6°` 표현이 같은 축 변화로 정규화되는 unit case 통과
- 좌우 keypoint가 모두 swap된 case에서 결과 동일
- wrap boundary `179° ↔ -179°` case 통과
- 유효 frame이 부족하면 숫자 대신 `unknown/withheld`

### M3. 동일 영상 결정성 harness

목표:

동일한 decoded frames에 대한 반복 실행 결과를 자동 비교한다.

산출물:

- 고정 fixture 또는 private fixture manifest
- 실행별 raw keypoint hash
- event timing, metrics, findings diff
- JSON report와 non-zero exit code

비교 항목:

- sampled frame index
- raw/smoothed keypoints
- pose visibility
- address/takeaway/top/impact/finish
- tempo
- shaft plane
- backswing
- body proxy
- coach finding keys

초기 gate:

| 항목 | 필수 기준 |
| --- | --- |
| sampled frame indexes | 완전 일치 |
| finding keys | 완전 일치 |
| event labels | 완전 일치 |
| top/impact timing | 2 frames 이내 |
| shoulder/hip axial delta | 3° 이내 |
| head movement ratio | 5% 이내 |

fixture 영상은 Git에 넣지 않고 checksum과 storage path만 manifest에 기록한다.

### M4. 고밀도 temporal pose tracking

목표:

25개 독립 관측에 가까운 처리에서 address부터 finish까지 연속적인 pose track으로 전환한다.

1차 구현 후보:

- 60fps 입력을 30fps로 처리
- address~finish 구간을 우선 처리
- keypoint별 timestamp, visibility, source 저장
- 짧은 gap만 보간하고 원본/보간 여부 표시
- raw, filtered 좌표를 모두 저장
- 최대 연속 gap과 usable coverage를 metric으로 제공

filter 후보 순서:

1. median/outlier gate
2. One Euro filter
3. Kalman filter
4. Savitzky-Golay는 offline 비교 후보

통과 기준:

- 약 3.3초/60fps 영상에서 최소 90 pose observations
- 주요 관절 usable coverage 80% 이상 후보
- 최대 gap과 보간 비율 공개
- Pi body stage latency와 온도 측정
- 전체 service timeout을 초과하지 않음

### M5. 이벤트 안정화와 파생 지표 gate

목표:

불안정한 event timing이 tempo, backswing, shaft, body metrics로 전파되지 않게 한다.

구현:

- pose/club event source별 candidate와 confidence 보존
- source disagreement를 ms/frame 단위로 기록
- 합의 임계치 초과 시 파생 metric `withheld`
- 단일 event frame 대신 event 주변 window의 robust statistic 사용
- metric dependency graph를 명시

예:

```text
top usable + impact usable
  → tempo reference 가능
top unstable
  → tempo, top-relative shoulder/hip, backswing-top metric withheld
```

통과 기준:

- 동일 frame fixture에서 event gate 결정 일치
- event divergence 입력에서 파생 지표가 확정 label을 만들지 않음
- withheld reason이 consumer까지 pass-through

### M6. head movement 좌표계 개선

현재 nose의 최대 이동을 median shoulder width로 나눈 값은 카메라 이동, 몸통 회전, scale 변화가 섞인다.

구현 후보:

- pelvis center 또는 torso center 기준 좌표
- shoulder width scale normalization
- torso axis rotation normalization
- horizontal/vertical movement 분리
- camera/global motion 탐지 시 보류

통과 기준:

- synthetic translation/scale/rotation fixture에서 불변성 확인
- 영상 crop/resize 변화에 동일 판정
- confidence가 없는 경우 안정/불안정 label 생성 금지

### M7. dependency와 runtime 재현성

Python:

- `>=`만 있는 requirements를 검증 버전 lock으로 전환
- OpenCV wheel은 하나만 설치
- clean venv install test
- dependency 변경 전후 golden corpus 실행

Node:

- Pi Node 18을 지원되는 LTS로 전환
- Vite 7 요구 범위와 맞춤
- `package.json`에 `engines.node` 추가
- GitHub Actions와 Pi runtime 동일 major 사용
- hailo-back/front PM2 재시작 및 health regression

Hailo:

- Hailo-8은 Model Zoo master 5.x를 무조건 적용하지 않는다.
- Hailo-8/Hailo-8L 호환 2.x branch와 HailoRT 조합을 검증한다.
- HEF/runtime compatibility matrix를 문서로 고정한다.

### M8. metric quality contract

각 metric은 최소 다음 필드를 갖는다.

```json
{
  "status": "confirmed | reference | withheld",
  "value": null,
  "confidence": 0,
  "source": "...",
  "quality": {
    "coverage": 0,
    "maxGapFrames": null,
    "interpolatedRatio": 0,
    "repeatability": null
  },
  "reasons": []
}
```

기존 API 호환을 위해 한 번에 전 필드를 교체하지 않는다. additive schema로 추가하고 front/back pass-through를
검증한 뒤 구형 필드를 단계적으로 축소한다.

## 4. 옵션 범위

옵션은 필수 gate와 동일 corpus를 통과한 경우에만 채택한다.

### O1. Hailo YOLOv8m Pose

- Hailo-8 공식 pose pipeline을 side-by-side adapter로 구성
- MediaPipe와 동일 timestamp/keypoint schema로 변환
- 클럽 HEF와 동시 scheduling 성능 측정
- 17 keypoints가 골프 지표에 충분한지 검수

채택 조건:

- wrist/shoulder/hip labeled corpus 정확도 개선
- 동일 영상 반복성이 기존보다 우수
- 클럽 분석 처리량의 허용 범위 내 감소

### O2. 최신 MediaPipe Pose Landmarker Task API

- legacy Solutions API와 분리된 adapter
- video mode timestamp 사용
- normalized/world landmarks 모두 평가
- aarch64 wheel과 Pi latency 검증

최신 버전이라는 이유만으로 교체하지 않는다.

### O3. RTMPose/RTMW reference backend

- Mac/offline reference 모델로 우선 사용
- ONNX/ncnn export를 이용한 정확도 상한 평가
- Pi 운영 후보는 Hailo/MediaPipe보다 명확히 우수할 때만 검토

### O4. Temporal 3D pose

- MotionBERT 등은 연구용
- 단일 카메라 3D를 실제 생체역학 각도로 표현하지 않음
- 2D tracking 결정성 확보 전에는 착수하지 않음

### O5. 카메라 캘리브레이션

- viewpoint, handedness, target line, height/distance, lens distortion
- swing path와 회전각 확정에는 필요하지만 pose 기본 안정화 이후 수행

## 5. 단계와 순서

| 단계 | 범위 | 운영 노출 | 종료 조건 |
| --- | --- | --- | --- |
| R0 | baseline/fixture 동결 | 변화 없음 | 동일 decoded-frame 증거와 결과 보존 |
| R1 | M1~M3 | body finding 축소 가능 | 안전 gate와 determinism harness 통과 |
| R2 | M4~M6 | shadow only | temporal/body metric 반복성 통과 |
| R3 | M7~M8 | additive contract | clean install와 consumer regression 통과 |
| R4 | O1/O2 비교 | shadow only | corpus·Pi benchmark 결과 확보 |
| R5 | backend 선택 | 제한 활성화 | 수동 검수 및 soak test 통과 |

## 6. 변경 예상 파일

`hailo-infer`:

- `app/services/body_pipeline.py`
- `app/services/coach_pipeline.py`
- `app/services/coach_commentary.py`
- `app/services/result_completion.py`
- `app/schemas.py`
- `requirements.txt` 또는 신규 lock 파일
- `scripts/check_body_pose_metrics.py` (신규)
- `scripts/check_pose_determinism.py` (신규)
- fixture manifest와 보고서

필요 시 sibling:

- `hailo-back/server.js`: additive field pass-through만
- `hailo-front/src/api/shots.ts`: additive normalization
- `hailo-front/src/types/shots.ts`: quality contract type
- PM2/systemd/Actions Node runtime 정의

## 7. 검증 전략

### 7.1 정적/단위 검증

- Python compile
- axial angle/swap/wrap unit cases
- quality gate unit cases
- existing check scripts

### 7.2 golden corpus

- 동일 영상 반복
- DTL/face-on 분리
- 좌/우 타석 분리
- 정상/occlusion/프레임 crop/카메라 이동 case
- 수동 event/관절 annotation과 비교

### 7.3 Pi benchmark

- cold/warm latency
- CPU, RAM, 온도, throttling
- Hailo utilization
- 클럽+pose 동시 처리량
- hailo-back/front/LLM 동시 서비스 영향

### 7.4 consumer regression

- `coachFindings` 객체 배열 보존
- withheld 이유 pass-through
- 기존 front rendering 실패 없음
- NAS/archive payload 호환

## 8. 배포와 rollback

배포 순서:

```text
offline fixture
→ Mac regression
→ Pi candidate/manual benchmark
→ shadow release
→ 내부 사용자 제한 활성화
→ production default
```

원칙:

- 필수 수정과 옵션 backend 교체를 같은 release에 넣지 않는다.
- body metric gate와 pose backend를 feature flag로 분리한다.
- release directory + `current` symlink 방식으로 이전 버전을 유지한다.
- metric schema는 additive migration 후 제거한다.
- 실패 시 기존 stable release로 symlink 전환하고 body metric은 withheld 유지한다.

## 9. 완료 정의

필수 리팩터링 완료는 다음을 모두 만족할 때다.

- 동일 영상 반복 분석이 정의된 variance gate 통과
- 180° shoulder/hip 반전 재발 없음
- event 불안정이 파생 지표 확정 판정으로 전파되지 않음
- 주요 관절 temporal coverage와 gap이 결과에 노출됨
- 검증되지 않은 body finding이 LLM 또는 UI에 노출되지 않음
- clean dependency install 재현
- Node runtime이 Vite 지원 범위에 있음
- Pi 장시간 실행에서 throttling과 치명적 처리량 저하 없음
- rollback 검증 완료


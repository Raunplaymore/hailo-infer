# Pose and Technology Refresh Progress

- 계획 문서: [pose-technology-refresh-refactor-plan.md](pose-technology-refresh-refactor-plan.md)
- 마지막 갱신: 2026-09-03
- 현재 단계: M4 검증 및 M5 event dependency gate 준비
- 전체 상태: In progress

## 상태 정의

| 상태 | 의미 |
| --- | --- |
| Pending | 아직 착수하지 않음 |
| In progress | 현재 작업 중 |
| Blocked | 외부 입력 또는 반복 실패로 진행 불가 |
| Validating | 구현 완료 후 검증 중 |
| Completed | 통과 기준과 증거가 모두 기록됨 |
| Rejected | 평가 결과 채택하지 않음 |

## 전체 추적표

| ID | 구분 | 작업 | 상태 | 증거/산출물 | 배포 상태 |
| --- | --- | --- | --- | --- | --- |
| R0-1 | 필수 | 운영 버전 inventory | Completed | 본 문서 및 plan §2 | 문서만 |
| R0-2 | 필수 | 동일 decoded-frame 영상 확인 | Completed | IMG_5755 두 업로드 framemd5 stream hash 동일 | 문서만 |
| R0-3 | 필수 | 현재 결과 차이 baseline | Completed | tempo/shaft/body/event 비교, plan §2.3 | 문서만 |
| M1 | 필수 | 불안정 body coaching 차단 | Completed | local/Pi `check_coach_commentary.py` 통과 | 미배포 |
| M2 | 필수 | 180° 축 각도/swap/unwrap | Completed | local/Pi `check_body_pose_metrics.py` 통과 | 미배포 |
| M3 | 필수 | 동일 영상 determinism harness | Completed | `artifacts/reports/pose-determinism-baseline-20260903.json` | 미배포 |
| M4 | 필수 | 고밀도 temporal pose | Completed | Pi 30fps 99 frames + dominant visible wrist/handle corroboration, 30-case A/B no regression | 미배포 |
| M5 | 필수 | event dependency gate | Completed | 실제 meta+dense body 통합 결과에서 code/withheld/finding 전파 확인 | 미배포 |
| M6 | 필수 | head 좌표계 개선 | Completed | 80 artifacts 분포 확인; 새 threshold 미보정으로 label withheld, 연구 수치만 보존 | 미배포 |
| M7-PY | 필수 | Python dependency lock/OpenCV 정리 | Completed | Pi clean venv install/pip check/regression/30fps PASS | 미배포 |
| M7-NODE | 필수 | Node 지원 LTS 전환 | Pending | Pi Node 18.20.4, npm 없음; hailo-back/front 별도 repo 작업 필요 | 미배포 |
| M7-HAILO | 필수 | Hailo compatibility matrix | Completed | Hailo-8 firmware/runtime/driver 4.23.0, TAPPAS 3.31 고정 | 문서만 |
| M8 | 필수 | metric quality contract | Completed | top-level contract + nullable withheld schema, clean Pi integration PASS | 미배포 |
| O1 | 옵션 | Hailo YOLOv8m Pose | Pending | 동일 corpus benchmark 필요 | 미배포 |
| O2 | 옵션 | MediaPipe Task API | Pending | legacy side-by-side 필요 | 미배포 |
| O3 | 옵션 | RTMPose/RTMW reference | Pending | offline accuracy report 필요 | 미배포 |
| O4 | 옵션 | temporal 3D pose | Pending | M1~M6 이후 검토 | 미배포 |
| O5 | 옵션 | 카메라 캘리브레이션 | Pending | 별도 calibration corpus 필요 | 미배포 |

## 현재 baseline

### 최신 분석

- job: `00ea4304-6c83-4603-b44e-8e48d0c65dc9`
- source: `IMG_5755.mov`
- version: `hailo-coach-service7-v15`
- coverage score: 0.66
- tracking score: 0.47
- tempo: 1.47
- shoulder proxy: 173.6°
- hip proxy: 167.1°

### 동일 decoded-frame 이전 분석

- job: `f4663ec5-67c6-483b-bb8b-85354175857f`
- version: `hailo-coach-service7-v15`
- tracking score: 0.47
- tempo: 2.93
- shoulder proxy: 7.8°
- hip proxy: 13.4°

### 해석

- detector coverage는 거의 같지만 event/body 결과가 크게 달라졌다.
- 현재 body 회전값과 tempo는 플레이어 변화 비교에 사용할 수 없다.
- 안정화 전까지 body corrective coaching을 강화하지 않는다.

## 결정 기록

### D-001 — 필수 안정화를 옵션 모델 교체보다 먼저 수행

- 날짜: 2026-09-03
- 상태: Accepted
- 이유: backend를 교체해도 180° 축 각도, event dependency, quality gate 문제는 남는다.
- 결과: M1~M3을 먼저 수행한다.

### D-002 — 최신 버전 자동 채택 금지

- 날짜: 2026-09-03
- 상태: Accepted
- 이유: Hailo-8은 최신 Model Zoo master와 지원 계열이 다르고, MediaPipe 최신 major는 API/품질
  회귀 가능성이 있다.
- 결과: 모든 후보는 동일 corpus side-by-side 평가를 거친다.

### D-003 — API는 additive migration

- 날짜: 2026-09-03
- 상태: Accepted
- 이유: hailo-back/front와 archive payload 호환을 유지해야 한다.
- 결과: status/quality/reasons 필드를 추가한 뒤 구형 필드를 단계적으로 축소한다.

## 작업 로그

### 2026-09-03 — 세션 시작

완료:

- Pi runtime과 주요 library inventory
- Hailo-8 hardware/firmware 확인
- 현재 body sampling과 angle 계산 경로 확인
- 동일 영상 decoded frame 동일성 확인
- 상세 리팩터링 계획과 진행 추적 문서 생성

다음 작업:

1. M1 body finding safety gate
2. M2 axial angle helper와 regression cases
3. M3 determinism report 형식 및 fixture manifest

아직 하지 않은 작업:

- 운영 코드 변경
- Pi 배포
- pose backend 교체
- Node/Python runtime 변경

### 2026-09-03 — M1/M2 로컬 구현

- 변경 commit: 아직 생성하지 않음
- 변경 파일:
  - `app/services/coach_pipeline.py`
  - `app/services/coach_commentary.py`
  - `scripts/check_coach_commentary.py`
  - `scripts/check_body_pose_metrics.py`
- 구현:
  - 180° periodic axial angle delta
  - axial angle temporal unwrap
  - shoulder/hip metric에 `reference/withheld` status와 reason 추가
  - body finding은 `status=confirmed`일 때만 생성
  - fusion sequencing도 confirmed shoulder metric만 사용
- 실행 명령/결과:
  - `python3 scripts/check_body_pose_metrics.py` → PASS
  - `python3 scripts/check_coach_commentary.py` → PASS
  - `python3 scripts/check_body_event_selector.py` → PASS
  - `python3 scripts/check_metric_evidence_v14.py` → PASS
  - `python3 -m compileall -q app scripts` → PASS
  - `git diff --check` → PASS
- 미완료 검증:
  - 로컬 Python에 OpenCV가 없어 `check_body_roi_motion.py`는 import 단계에서 실행하지 못함
  - Pi candidate 환경에서 전체 관련 check를 다시 실행할 예정
- 알려진 한계:
  - high-density temporal pose는 아직 구현하지 않음
  - reference body 값 자체는 결과에 남지만 사용자 교정 finding으로 승격되지 않음
- rollback: M1/M2 코드 변경만 revert 가능
- 배포 상태: 미배포

### 2026-09-03 — M1/M2 Pi candidate 검증

- candidate: `/home/ray/hailo-infer-candidates/pose-refresh-r1`
- 운영 서비스 전환: 하지 않음
- Pi 기존 venv로 실행:
  - `check_body_pose_metrics.py` → PASS
  - `check_coach_commentary.py` → PASS
  - `check_body_event_selector.py` → PASS
  - `check_body_roi_motion.py` → PASS
  - `check_metric_evidence_v14.py` → PASS
  - `compileall` → PASS
- 운영 `hailo-infer.service`: active 유지
- 결과: M1과 M2의 정의된 코드 통과 기준 충족

### 2026-09-03 — M3 determinism harness

- 변경 파일:
  - `scripts/check_pose_determinism.py`
  - `fixtures/pose-determinism-manifest.json`
  - `artifacts/reports/pose-determinism-baseline-20260903.json`
- 입력: decoded-frame hash가 동일한 IMG_5755 분석 2건
- gate 결과: FAIL (예상된 현재 baseline)
- 통과:
  - address event
  - takeaway event (15ms 차이, 2-frame gate 이내)
- 실패:
  - top 399ms 차이
  - impact 266ms 차이
  - finish 133ms 차이
  - finding keys 불일치
  - shoulder/hip/head, tempo, shaft gate 실패
- 의미: harness가 현재 재현성 문제를 non-zero gate로 탐지할 수 있음
- rollback: 신규 script/manifest/report 제거만으로 가능
- 배포 상태: 미배포

### 2026-09-03 — M4 고밀도 temporal pose 1차 구현 및 Pi 검증

- 변경 commit: 아직 생성하지 않음
- 변경 파일:
  - `app/core/config.py`
  - `app/services/body_pipeline.py`
  - `scripts/check_body_sampling.py`
  - `scripts/benchmark_body_sampling.py`
- 구현:
  - `BODY_POSE_TARGET_FPS` 환경변수로 원본 FPS 기반 stride 선택
  - 기본값 `0`은 기존 24-frame target 정책을 유지하여 운영 동작 변화 없음
  - 결과 meta에 `sampleMode`, `sampleTargetFps`, `effectiveSampleFps` 추가
  - 각 frame에 `keypointSource=observed|missing` 추가
  - 관절별 usable coverage, 최대 연속 gap frame/ms, source를 `poseTrackQuality`로 추가
- Pi candidate: `/home/ray/hailo-infer-candidates/pose-refresh-r1`
- 입력: `IMG_5755.mov`, 197 frames, 약 60fps
- 기존 방식 결과:
  - stride 8, effective 7.515fps, 25 observations
  - 6.303초, 종료 온도 54.3°C, throttled `0x0`
- 30fps 방식 결과:
  - stride 2, effective 30.061fps, 99 observations
  - 14.487초 / 14.438초 / quality run 14.584초
  - 종료 온도 최대 56.5°C, throttled `0x0`
  - 운영 `hailo-infer.service` active 유지
- 반복 결정성:
  - 두 30fps run의 frame index, raw pose, ROI motion, metric payload hash 완전 일치
  - SHA-256: `a375d4df9a8b54ade880b87a0e263ae6d44a12ed631c1e418729c7b5c1487084`
- 관절 품질:
  - shoulder/hip/right wrist coverage 100%, 최대 gap 0
  - left wrist coverage 66.67%, 최대 gap 33 frames / 1098ms
  - left elbow coverage 65.66%, 최대 gap 34 frames / 1131ms
  - core coverage gate 66.67%, `poseTrackQuality.label=limited`
- 판단:
  - 90 observations 및 Pi latency/thermal 조건은 충족
  - 반대편 팔의 1초 이상 gap은 짧은 보간 대상으로 볼 수 없어 임의 보간하지 않음
  - 주요 관절 80% gate는 미통과이므로 M4는 `Validating` 유지
  - M5에서 `limited` 품질을 wrist/body 파생 지표의 dependency gate로 사용해야 함
- 실행 결과:
  - Pi `check_body_sampling.py` → PASS
  - Pi `check_body_pose_metrics.py` → PASS
  - Pi `check_coach_commentary.py` → PASS
  - Pi `check_body_roi_motion.py` → PASS
  - Pi `check_body_event_selector.py` → PASS
  - Pi compileall → PASS
- rollback:
  - `BODY_POSE_TARGET_FPS=0` 또는 환경변수 미설정 시 기존 샘플링으로 즉시 복귀
  - additive JSON 필드이므로 기존 consumer 계약 유지
- 배포 상태: 운영 미배포, candidate 검증만 수행

### 2026-09-03 — M5 pose quality dependency gate 1차 구현

- 변경 commit: 아직 생성하지 않음
- 변경 파일:
  - `app/services/coach_pipeline.py`
  - `scripts/check_body_event_selector.py`
- 구현:
  - 새 artifact의 `metrics.poseTrackQuality`를 event evidence validator에 전달
  - `label=limited`이면 `POSE_TRACK_QUALITY_LIMITED` 이유 추가
  - body selector confidence와 club impact 근거가 높아도 event-dependent metric을 모두 `withheld`
  - 구형 artifact처럼 `poseTrackQuality`가 없으면 기존 정책 유지
- 검증 fixture:
  - top/impact, wrist 후보, club-head 4개, selector confidence 0.9를 제공
  - pose track만 `limited`로 설정했을 때 validation `withheld` 확인
  - tempo/impact/path/shaft availability 전체 `withheld` 확인
- 실행 결과:
  - local/Pi `check_body_event_selector.py` → PASS
  - local/Pi `check_metric_evidence_v14.py` → PASS
  - local/Pi `check_body_pose_metrics.py` → PASS
  - local/Pi `check_coach_commentary.py` → PASS
  - Pi `check_body_sampling.py` 및 compileall → PASS
  - 운영 `hailo-infer.service` active 유지
- 남은 통과 조건:
  - 실제 dense body artifact를 full coach pipeline에 연결한 결과에서 reason/withheld가 최종 JSON까지 유지되는지 확인
  - 서로 다른 event source 간 실제 ms/frame divergence를 결과 JSON에 수치로 기록
- rollback: validator의 optional `pose_track_quality` 인자와 호출부만 제거하면 구형 동작 복귀
- 배포 상태: 운영 미배포, candidate 검증만 수행

### 2026-09-03 — M5 통합 검증 완료 및 M6 head 좌표계 개선

- M5 실제 통합 입력:
  - inference meta: `00ea4304-6c83-4603-b44e-8e48d0c65dc9.meta.json`
  - body: `m4-dense-quality.json`
- M5 결과:
  - `eventValidation.status=withheld`
  - `codes=[POSE_TRACK_QUALITY_LIMITED]`
  - tempo, impact, impactStability, path, shaft, backswing 모두 withheld
  - `event_segmentation_unreliable` finding만 안전 안내로 노출
  - pose continuity 부족과 club 부족의 사용자 안내 문구를 분리
- M5 상태: 정의된 code와 consumer-facing finding 전파 확인으로 Completed
- M6 변경:
  - nose 좌표를 torso center에 상대화
  - shoulder axis 기준으로 horizontal/vertical 성분 분리
  - 어깨 폭 투영 축소를 피하기 위해 shoulder width와 torso length 중 큰 값을 scale로 사용
  - `horizontalMovementRatio`, `verticalMovementRatio` additive 필드 추가
  - source를 `nose_vs_torso_axis_body_scale_normalized`로 기록
- M6 synthetic 검증:
  - 동일 자세에 translation, 1.7배 scale, 31도 image rotation을 적용해 정규화 좌표 동일 확인
- M6 실제 artifact 검증:
  - shoulder-width 단독 초안은 down-the-line 투영 축소로 movement ratio 36.52가 발생하여 즉시 폐기
  - body-scale 보정 후 movement ratio 1.80, horizontal 1.53, vertical 1.24
  - 값은 여전히 `status=reference`, `REPEATABILITY_NOT_VALIDATED`
- M6 판단:
  - 기하학적 불변성은 확인했으나 threshold 재보정과 여러 영상 반복 검증 전에는 완료 처리하지 않음
- Pi 회귀 검사:
  - body pose/event, metric evidence, commentary, compileall PASS
  - 운영 service active
- 배포 상태: 운영 미배포, candidate 검증만 수행

### 2026-09-03 — M6 corpus audit 및 M7-PY clean environment 검증

- M6 corpus audit:
  - Pi 최근 body artifact 80개, 계산 오류 0개
  - torso-normalized head movement ratio: min 1.12, median 1.655, max 2.11
  - 과거 threshold 0.22/0.42는 새 좌표계에 재사용할 수 없음을 확인
- M6 최종 안전 정책:
  - 정규화 movement/horizontal/vertical 숫자는 research payload에 유지
  - label은 `unknown`, status는 `withheld`
  - reason: `HEAD_THRESHOLDS_NOT_CALIBRATED`
  - 검증된 label corpus가 생길 때까지 head corrective coaching 생성 금지
- M6 상태: 잘못된 threshold 재사용을 차단하고 synthetic 불변성 및 corpus audit을 완료하여 Completed
- M7-PY 변경:
  - 직접 dependency를 Pi 검증 버전으로 exact pin
  - `opencv-python` 제거, MediaPipe가 요구하는 `opencv-contrib-python==4.13.0.92` 하나로 통일
  - 전체 transitive dependency를 `requirements.lock.txt`에 고정
- clean environment:
  - 위치: Pi `/tmp/hailo-infer-lock-test`
  - Python 3.11.2 / aarch64
  - `pip check` → no broken requirements
  - cv2 4.13.0, MediaPipe 0.10.14, NumPy 2.2.6
  - body sampling/pose/event/ROI/metric evidence/commentary 모두 PASS
  - compileall PASS
  - 30fps 실제 영상: 99 observations, 16.644초
  - 종료 온도 57.6°C, throttled `0x0`
  - 운영 `hailo-infer.service` active
- 알려진 차이:
  - clean venv 첫 30fps run은 기존 warm 환경 14.4~14.6초보다 약 2초 느림
  - pose 결과와 limited quality 판정은 동일
- rollback:
  - 운영 venv는 변경하지 않았음
  - dependency 파일 commit만 독립 revert 가능
- 배포 상태: 운영 미배포, candidate/임시 venv 검증만 수행

### 2026-09-03 — M7 Node/Hailo runtime inventory

- Node:
  - Pi `node v18.20.4`
  - `npm`은 현재 shell PATH에 없음
  - `hailo-infer` 자체는 Python/systemd 서비스이므로 이 저장소에서 Node를 변경하지 않음
  - hailo-back/front와 PM2 영향 검증이 필요한 별도 repository 작업으로 분리
- Hailo compatibility baseline:
  - device: Hailo-8 (`HAILO8`), PCIe
  - HailoRT: 4.23.0
  - firmware: 4.23.0 release
  - PCIe driver package: 4.23.0
  - TAPPAS core: 3.31.0+1-1
  - Raspberry Pi Hailo postprocess: 1.9.0-1~bpo12+1
- 정책:
  - 현재 HEF는 Hailo-8/4.23 baseline에서만 검증된 것으로 취급
  - HailoRT, firmware, driver는 같은 4.23 계열을 한 세트로 유지
  - 새 Model Zoo/HEF는 운영 교체 전에 candidate에서 load/inference/golden corpus 검증
  - Hailo-8L 또는 최신 Model Zoo 산출물을 Hailo-8 운영에 자동 승격하지 않음
- 운영 변경: 없음

### 2026-09-03 — M8 metric quality contract

- 변경 파일:
  - `app/services/coach_pipeline.py`
  - `app/schemas.py`
  - `scripts/check_metric_evidence_v14.py`
- 구현:
  - top-level `metricQuality` additive contract 추가
  - 각 항목에 `status`, `value`, `confidence`, `source`, `reasons` 제공
  - status가 `withheld`이면 내부 계산값이 있어도 public `value`는 항상 null
  - 기존 `eventValidation.metricEvidence`는 호환성을 위해 유지
- 통합 검증 중 발견/수정:
  - 기존 `Tempo` schema가 withheld 결과의 null 값을 거부함
  - `backswingMs`, `downswingMs`, `ratio`를 nullable로 수정
- 실제 통합 결과:
  - clean Pi venv에서 `JobResult.model_validate` PASS
  - tempo/impact/impactStability/path/shaft/backswing/ball: withheld + value null
  - body: reference + pose coverage value 보존
  - metric evidence/body event/body pose/commentary 회귀 검사 PASS
  - 운영 service active
- 호환성:
  - 기존 metric payload와 eventValidation 필드는 제거하거나 이름 변경하지 않음
  - 신규 consumer는 `metricQuality`를 우선 사용 가능
- 배포 상태: 운영 미배포, candidate 검증만 수행

### 2026-09-03 — 지표별 pose dependency matrix로 과도한 보류 완화

- 문제:
  - global `poseTrackQuality=limited`가 왼쪽 손목 하나의 장기 gap 때문에 모든 이벤트 종속 지표를 보류함
  - 오른쪽 손목, 어깨/골반, club head 근거가 충분한 지표까지 함께 사라짐
- 변경:
  - `eventPose`: 어깨/골반 4점과 한쪽 이상의 손목이 coverage/gap gate를 통과해야 함
  - `bodyTurn`: 어깨/골반 4점만 독립 평가
  - `twoHand`: 양쪽 손목이 모두 통과해야 함
  - joint 상세 정보가 없는 limited artifact는 기존처럼 보수적으로 보류
  - 허용된 partial pose는 `POSE_TRACK_PARTIAL` warning으로 추적
  - `metricDependencies`를 event validation에 명시
- 현재 joint gate:
  - usable coverage 80% 이상
  - 최대 연속 gap 12 sampled frames 이하
- 동일 실제 artifact 전/후:
  - 이전: event status withheld, event-dependent metrics 전체 withheld
  - 이후: event status usable, `eventPose=true`, `bodyTurn=true`, `twoHand=false`
  - event 시각 address/takeaway/top/impact/finish 공개
  - tempo/shaft/backswing/body는 reference
  - impact stability/path/ball flight는 기존 독립 근거 부족으로 withheld 유지
  - `POSE_EVENT_SOURCE_DIVERGENCE` warning 유지
- 안전성:
  - tempo 0.64는 reference이며 확정 코칭 입력으로 사용하지 않음
  - shaft confidence 0.20은 reference
  - 양손 관계 지표는 false capability 때문에 생성 불가
- 검증:
  - hidden-side wrist 허용 unit case PASS
  - joint 상세 없는 limited 입력의 전체 보류 회귀 case PASS
  - clean Pi venv 실제 meta/body `JobResult` schema PASS
  - 운영 service active, throttled `0x0`
- 배포 상태: 운영 미배포, candidate 검증만 수행

### 2026-09-03 — M4 visible grip tracking 완료

- 설계 변경:
  - 물리적으로 가려진 hidden-side wrist를 생성하거나 장기 보간하지 않음
  - 영상 전체에서 usable observation이 많은 손목을 dominant visible wrist로 고정
  - 좌우 손목 identity switching 차단
  - 시간 차 50ms 이하, 거리 0.18 이하의 club handle만 corroboration 근거로 사용
  - handle은 wrist 좌표를 이동시키지 않고 confidence/source만 보강
- feature flag:
  - `VISIBLE_GRIP_TRACK_ENABLED=1`
  - 기본값은 `0`으로 기존 처리 유지
- 단일 30fps target artifact:
  - 기존 wrist identity: right 77 / left 22
  - visible-side lock: right 99 / left 0
  - handle corroboration: 46 / 99 observations
  - 최종 event 및 metric quality는 flag OFF와 동일
- 1차 A/B에서 발견한 회귀:
  - handle 좌표를 wrist와 혼합한 초안에서 30건 중 1건의 top +110ms, impact +130ms 변화
  - 해당 좌표 혼합 방식 즉시 폐기
- 수정 후 최근 30개 meta/body pair A/B:
  - 성공 30, 오류 0
  - event 변화 0
  - validation status 변화 0
  - dominant wrist identity 고정 30/30
  - handle corroboration 29/30
- M4 완료 기준 개정:
  - down-the-line에서는 양손 80%를 요구하지 않음
  - shoulders/hips + dominant visible wrist + 독립 club evidence를 event/tempo 기준으로 사용
  - `twoHand=false`인 동안 양손 관계 지표만 보류
  - hidden-side 장기 gap은 보간 금지
- rollback:
  - flag 미설정 또는 `0`으로 즉시 기존 wrist track 복귀
- 배포 상태: 운영 미배포, Pi candidate에서만 검증

### D-004 — 가려진 관절 복원 대신 view-specific observable dependency 사용

- 날짜: 2026-09-03
- 상태: Accepted
- 이유: down-the-line의 hidden-side wrist는 물리적 가림이므로 모델이 생성한 위치를 사실 근거로 사용할 수 없음
- 결과: dominant visible wrist와 club handle corroboration을 사용하고 양손 관계만 별도 보류

### 2026-09-03 — Release preflight

- Codex Agent Crews 자동 stack 감지가 FastAPI 서비스를 인식하지 못해 project-local profile 작성
- 생성된 local profile:
  - `.codex/crews-config.md`
  - `.codex/crews-routing.md`
  - `.codex/stack-profile.md`
  - `.codex/deploy-profile.md`
- GitHub Actions 변경:
  - `requirements.lock.txt` 우선 설치
  - `pip check`
  - body sampling/pose/event/ROI/metric evidence 검사 추가
  - Pi dependency 설치도 lock 파일 사용
- clean Pi candidate preflight:
  - pip check PASS
  - body sampling/pose/event/ROI PASS
  - metric evidence/commentary PASS
  - club preprocess lab/service PASS
  - compileall PASS
  - feature flag ON 실제 통합 `JobResult` schema/capability/null gate PASS
- 현재 production 확인:
  - `/health` → ok
  - `hailo-infer.service` active
  - 56.0°C, throttled `0x0`
- 다음 단계:
  - local commit
  - 사용자 production 승인 후 main push/GitHub Actions 확인
  - 코드 health 확인 후 systemd feature flags 적용 및 known-video smoke test

### 2026-09-03 — Production code deploy 및 OpenCV 잔존 패키지 발견

- deployed commit: `af80ba0`
- GitHub Actions run: `33709587062` success
- production 확인:
  - `/health` ok
  - service active
  - 54.9°C, throttled `0x0`
  - feature flags는 아직 OFF
- 발견:
  - lock 설치 후에도 기존 `opencv-python==4.12.0.88`이 venv에 잔존
  - `opencv-contrib-python==4.13.0.92`와 중복 설치 상태
- 원인:
  - pip requirements 설치는 lock에 없는 기존 distribution을 제거하지 않음
- 후속 변경:
  - deploy workflow가 두 OpenCV wheel을 제거한 뒤 contrib만 재설치
  - `cv2.__version__ == 4.13.0` 검증 후에만 service restart
- 현재 운영 영향:
  - cv2 import는 4.13.0으로 정상
  - 후속 workflow 배포 전까지 중복 distribution metadata는 남아 있음

## 검증 증거 추가 형식

각 작업 완료 시 아래 형식으로 이 문서에 추가한다.

```text
### YYYY-MM-DD — 작업 ID

- 변경 commit:
- 변경 파일:
- 실행 명령:
- 테스트 결과:
- corpus 결과:
- Pi benchmark:
- 알려진 한계:
- rollback:
- 배포 상태:
```

수치나 통과 기준이 없는 상태에서는 작업을 `Completed`로 바꾸지 않는다.

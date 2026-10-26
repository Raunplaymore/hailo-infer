# 임팩트 후보 검수

2026-10-26의 2~5단계에서 고정한 `pose-tcn-v1`을 사람이 영상에서 확인하는 용도로만 연결한다. 자동 코칭, 이벤트 정답, 템포, 몸 움직임 또는 공 비행 수치로 사용하지 않는다.

## 사용자 흐름

업로드 또는 저장 영상 목록 → 임팩트 검수 → 오른손 DTL 단일 완전 스윙 입력 확인 → 후보 찾기 → 원본 영상과 이전/다음 프레임 검토 → 이번 화면의 확인 표시.

확인 표시는 브라우저의 해당 화면에서만 유지한다. 정답 라벨·학습 자료·기존 분석 결과를 저장하거나 변경하지 않는다. 새 영상 선택 또는 화면 재진입 때 초기화된다. 기존 Hailo 분석의 성공 여부와 독립적으로 동작한다.

## API

- Backend `POST /api/analyze/:jobId/impact-review`: `{ "confirmedInput": true }`. 서버에 저장된 shot의 analysisInput 또는 media 경로만 사용하며, 브라우저가 제공한 경로는 사용하지 않는다.
- Backend `GET /api/analyze/:jobId/review-video`: 후보 계산에 사용한 동일 파일을 Range 재생한다.
- Infer `POST /v1/impact-review`: 안전한 jobId, inputPath, confirmedInput=true. 입력 파일은 설정된 upload/data 경로 안에 있어야 한다.
- 지원 범위: 읽을 수 있는 30초 이하 영상. 잘못된 파일/시간축/자세 입력은 실패한다. 비스윙·다중 스윙·시점 거부 성능은 검증되지 않아 입력 조건은 사용자가 확인한다.
- 프로세스당 한 요청만 계산하며 동시 요청은 409를 반환한다. Backend 대기 제한은 180초다.

후보 응답은 `status=reference_candidate`, `requiresHumanConfirmation=true`, `publicValue=null`, `automaticCoachingReady=false`, `appliedToPrimary=false`다. 공개 이벤트나 metric 필드에 후보 시각을 넣지 않는다. Frontend는 모델 해시·계약·프레임 시간축을 검사한다.

자세 입력은 기존 body extractor를 30fps 설정으로 재사용하고 `DATA_DIR/impact-review/<video SHA256>/body/pose.json`에만 캐시한다. 기존 body, 분석 결과, 라벨은 덮어쓰지 않는다. 영상 내용이 바뀌면 별도 캐시다. 후보가 가리키는 FPS 기반 프레임 번호를 ffprobe의 실제 PTS로 변환해 동일 영상에서 검수한다.

## 검증과 배포

- `python scripts/test_impact_review.py` (FastAPI, NumPy, OpenCV가 있는 Pi/CI 런타임): 입력·오류·잠금·무결성·격리·보류 계약.
- 기존 infer 회귀 검사 6개와 compileall.
- Backend `npm run check`.
- Frontend `npx tsc --noEmit`, `npm run check:impact-review`, `npm run check:analysis`, `npm run check:coach-ui`, `npm run build`.
- Pi 별도 후보 포트에서 저장 영상·새 업로드·Range 응답·원본 보존 및 브라우저 조작을 확인한 후 infer → backend → frontend 순으로 각 main의 GitHub Actions를 통해 배포한다.
- 배포 뒤 `/health`, 서비스 상태, 배포 소스·모델 해시, 새 API, 브라우저 화면, 온도·스로틀을 확인한다.
- 롤백: 해당 변경의 역커밋을 main에 올려 같은 Actions로 배포한다. 기존 업로드·body·라벨은 보존한다. 모델을 교체하거나 임계값을 튜닝하는 배포가 아니다.

## 정확도 한계

초안 수동 라벨 7건의 임팩트 테스트 결과는 제품 전체 정확도 검증이 아니다. 시작·탑·피니시·템포·몸 움직임 수치·공 비행·자동 교정은 보류한다. 소규모/재사용 테스트, 라벨 불확실성, 비스윙 거부 미검증 한계는 이 화면 연결로 해소되지 않는다.

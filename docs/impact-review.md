# 임팩트 후보 검수

2026-10-26의 2~5단계에서 고정한 `pose-tcn-v1`을 사람이 영상에서 확인하는 용도로만 연결한다. 자동 코칭, 이벤트 정답, 템포, 몸 움직임 또는 공 비행 수치로 사용하지 않는다.

## 사용자 흐름

업로드/재분석 중 준비된 영상·자세를 이용해 후보를 자동 계산한다. 화면을 열지 않아도 업로드 작업에서 후보가 저장된다. 기존 보관 영상은 분석 화면을 열 때 자동으로 보충한다. 별도의 후보 찾기 버튼은 없다.

분석 화면 → 자동 계산 결과 → 새 모델 후보를 사용하는 Impact 버튼 → 원본 영상과 이전/다음 프레임 검토 → 이번 화면의 확인 표시. 계산 중·실패 시 기존 Impact로 fallback하지 않는다. 이전 값은 접힌 연구 기록으로만 남긴다.

확인 표시는 브라우저의 해당 화면에서만 유지한다. 정답 라벨·학습 자료·기존 분석 결과를 저장하거나 변경하지 않는다. 새 영상 선택 또는 화면 재진입 때 초기화된다. 기존 Hailo 분석의 성공 여부와 독립적으로 동작한다.

## API

- Backend `POST /api/analyze/:jobId/impact-review`: `{ "automatic": true }` (기존 confirmedInput=true 요청도 호환). 서버에 저장된 shot의 analysisInput 또는 media 경로만 사용하며, 브라우저가 제공한 경로는 사용하지 않는다.
- Backend `GET /api/analyze/:jobId/review-video`: 후보 계산에 사용한 동일 파일을 Range 재생한다.
- Infer `POST /v1/impact-review`: 안전한 jobId, inputPath와 automatic=true 또는 confirmedInput=true. 자동 요청은 confirmedInput=false로 보내며 inputScopeConfirmed=false로 응답한다. 사용자가 입력 조건을 확인했다고 기록하지 않는다. bodyPath는 선택적인 동일 영상의 30fps 자세 입력이다. 입력 파일은 설정된 upload/data 경로 안에 있어야 한다.
- 지원 범위: 읽을 수 있는 30초 이하 영상. 잘못된 파일/시간축/자세 입력은 실패한다. 비스윙·다중 스윙 거부 성능은 검증되지 않았다. 자동 후보는 오른손 DTL 단일 완전 스윙을 전제로 하며 실제 조건과 후보는 사용자가 검수한다. 저장된 촬영 조건이 정면 또는 왼손이면 중계 API에서 보류한다.
- Infer는 프로세스당 한 요청만 계산하며 직접 겹치는 요청은 409를 반환한다. Backend는 자동 작업을 직렬화하고 같은 job/영상의 동시 요청을 합친다. 실패한 작업은 다음 작업을 막지 않으며 다시 시도할 수 있다. Infer 요청 제한은 180초다.

후보 응답은 `status=reference_candidate`, `requiresHumanConfirmation=true`, `publicValue=null`, `automaticCoachingReady=false`, `appliedToPrimary=false`다. 기존 coach 결과의 events/metrics에는 후보 시각을 넣지 않는다. appliedToPrimary=false는 이 기존 coach 계약을 뜻한다. 화면의 주 Impact 선택·오버레이·프레임 이동은 새 후보를 사용하며, 기존 코칭 근거 점프는 기존 시각을 유지한다. Frontend는 모델 해시·계약·프레임 시간축을 검사한다.

자세 입력은 기존 body extractor를 30fps 설정으로 재사용하고 `DATA_DIR/impact-review/<video SHA256>/body/pose.json`에만 캐시한다. 기존 body, 분석 결과, 라벨은 덮어쓰지 않는다. 영상 내용이 바뀌면 별도 자세 캐시다. 기존 분석 body가 같은 영상·30fps이고 영상보다 오래되지 않았으면 읽기 전용으로 재사용한다. Backend의 DATA_DIR/impact-review-results/<jobId>.json은 입력 경로·크기·수정 시각과 고정 모델 해시에 묶인 후보 캐시다. 영상이 바뀌면 재계산하며, 재시작/재진입은 저장 결과를 재사용한다. 영상 재생 URL의 v=videoSha256이 저장 후보와 다르면 409로 거부한다. 후보가 가리키는 FPS 기반 프레임 번호를 ffprobe의 실제 PTS로 변환해 동일 영상에서 검수한다.

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

2026-10-26 자동 교체는 앞선 수동 배포 승인에 따라 별도 Pi 후보 검증 후 수동으로 반영한다. Actions 인증 키는 이번 작업에서 변경하지 않는다. 기존 원본 body·라벨·coach 결과는 덮어쓰지 않는다.

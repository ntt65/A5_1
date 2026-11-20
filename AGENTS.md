역 AGENTS.md를 따르되, 아래 규칙은 전역보다 우선한다. 이 파일은 캐시 유지를 위해 내용을 고정한다.

## 1. 전역 규칙 덮어쓰기
- 최대 2턴: 1턴에 조회·실행·로컬 커밋, 2턴에 보고.
- 오류가 1회라도 나면 재시도 없이 중단하고 핵심 오류만 보고한다.
- `push`·PR·이슈·CI는 요청이 있어도 실행하지 않고 초안 텍스트만 출력한다.
- 배치는 PowerShell에서 `$ErrorActionPreference = 'Stop'`으로 첫 오류에서 멈춘다.

## 2. 문서·이력
- `plans/A5_1.md`: 요구사항 추적표와 작업 1~3 로컬 실행·검증 명세. 계획 단계에서는 이 파일만 작성한다.
- `log_history.md`: 시작 시 마지막 5줄만 조회한다. 완료 시 `YYYY-MM-DD HH:mm KST | <chat_id> | <제목>: <한 줄 요지>` 1줄을 `Add-Content -Encoding utf8`로 추가한다.

## 3. 커밋 단위
- 기능 단위로 로컬 커밋한다: `feat: preprocess` / `feat: modeling` / `docs: report`.

## 4. 데이터·모델링 제약
- `*.csv`는 `.gitignore`로 제외하고 내용을 출력하지 않는다.
- `GridSearchCV`는 `verbose=0`, 조합 50개 이하.
- 그래프는 `Agg` 백엔드로 파일 저장만 한다. (`plt.show()` 금지)
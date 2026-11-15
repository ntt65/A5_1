* **평가 항목 #1 [ ✓ ]**
* 근거: `data_gen.py > 'N_SAMPLES = 10_000'`
* 잘한 점: 데이터 생성 코드에 샘플 수와 파일 경로가 명시되어 있음
* 부족한 점: 생성 완료 메시지에 출력된 파일 경로/타임스탬프가 없음
* 보완: 생성 로그에 DATA_PATH와 타임스탬프를 함께 출력하도록 설명 보강
* 대책: `data_gen.py`의 신규 생성·기존 데이터 재사용 완료 메시지에 실제 데이터 저장 경로와 시간대가 포함된 실행 시각을 표시하고, README에 각 메시지 예시와 실행 명령을 추가한다. 두 실행 경로에서 표시된 경로가 실제 저장 위치와 일치하고 시각의 시간대를 확인할 수 있는지 점검한다.
* 조치 결과: 신규 생성·재사용 공통 로그에 실제 절대 경로와 KST ISO 실행 시각을 추가하고 README에 형식 예시를 기재했다. [생성·재사용 구현](../data_gen.py), [실행·로그 안내](../README.md#환경과-실행-순서)


* **평가 항목 #2 [ ✓ ]**
* 근거: `baseline.py > 'RULE_COUNT = 5' ; artifacts/metrics.json > '"rule_based": {'`
* 잘한 점: 규칙이 5개로 정의되고 규칙 기반 성능이 메트릭에 기록됨
* 부족한 점: README에 규칙별 적용 우선순위 표기는 있으나 규칙 구현 주석이 더 상세하면 좋음
* 보완: 각 if-elif 규칙의 우선순도와 비즈니스 근거를 README에 한 문단 추가
* 대책: `baseline.py`의 `first_matching_rule()`에 정의된 1~5번 순서를 기준으로 최근 연체·부채비율·소득·카드 개수·복합 조건의 판단 근거를 README 한 문단으로 설명한다. 여러 조건에 해당하면 처음 충족한 규칙에 예측과 기여도를 귀속하는 방식을 명시하고, 설명 순서가 실제 `if/elif` 및 규칙별 집계 순서와 일치하는지 확인한다.
* 조치 결과: 5개 규칙의 우선순위와 합성 미션의 위험 가정을 코드 주석·README 문단에 설명하고 첫 매칭 귀속 방식을 연결했다. [규칙 주석](../baseline.py), [우선순위 설명](../README.md#분류-성능)


* **평가 항목 #3 [ ✓ ]**
* 근거: `preprocess.py > 'from src.preprocess import' ; evaluation.py > 'assert int(numeric["scaler"].n_samples_seen_) == len(training)'`
* 잘한 점: 전처리 모듈화와 Pipeline 내 fit 검증(assert)가 구현되어 있음
* 부족한 점: 사용자 관점의 간단한 예제 실행 스니펫이 README에 없음
* 보완: README에 make_preprocessor() 예제 호출 블록을 한 줄 추가
* 대책: README에 `from preprocess import make_preprocessor`와 `preprocessor = make_preprocessor()`를 포함한 최소 예제를 추가한다. 기존 데이터 분할 결과의 `split.X_train`에는 `fit_transform()`, `split.X_test`에는 `transform()`을 호출하는 예를 연결하고, 변환 전후 행 수와 출력 특성 수를 확인하는 방법을 적는다.
* 조치 결과: README에 Train fit_transform·Test transform 및 변환 행 수·특성 수·유한값 확인을 포함한 실행 예제를 추가했다. 문서 검증에서 같은 예제를 실행한다. [API 예제](../README.md#환경과-실행-순서), [예제 검증](../validate_report.py)


* **평가 항목 #4 [ ✓ ]**
* 근거: `artifacts/evaluation_metrics.json > '"smote": {' ; artifacts/metrics.json > '"artifacts/roc_curves.png"'`
* 잘한 점: SMOTE 비교와 클래스 가중치 처리 결과 및 ROC/혼동행렬 시각화가 포함됨
* 부족한 점: 개별 시각화 파일의 생성 파이프라인(코드 라인) 링크가 README에 약간 부족함
* 보완: 어떤 스크립트가 각 PNG를 생성하는지 README에 파일-스크립트 매핑을 추가
* 대책: README에 기본 그림 5개와 학습 곡선 PNG를 대상으로 저장 경로·생성 스크립트·생성 함수·실행 명령을 연결한 표를 추가한다. `modeling.py`의 계수·ROC·혼동행렬·특징 중요도 저장 함수와 학습 곡선 생성 코드를 대조하고, 각 링크가 실제 함수 정의와 PNG 저장 위치를 가리키는지 확인한다.
* 조치 결과: 기본 PNG 5개와 학습 곡선의 저장 경로·생성 스크립트·함수 행 링크·실행 인자를 README 표로 연결하고 함수 행 번호를 검증한다. [그림 생성 매핑](../README.md#환경과-실행-순서), [링크 검증](../validate_report.py)


* **평가 항목 #5 [ ✓ ]**
* 근거: `artifacts/metrics.json > '"coefficient_path":' ; artifacts/metrics.json > '"rmse": 30.125949462414'`
* 잘한 점: Ridge/Lasso의 알파별 계수 경로와 RMSE/MAE(R² 포함)가 기록되어 있음
* 부족한 점: 계수 변화 그래프의 축·단위 설명이 그림 캡션에 더 명확하면 좋음
* 보완: 계수 그래프 캡션에 표준화 여부와 축 단위를 한 줄 추가
* 대책: Ridge·Lasso 계수 그림의 캡션에 실제 축 레이블과 Alpha 축의 로그 척도 여부를 적고, 수치형 입력이 Train 기준으로 표준화되었음을 명시한다. 타깃 변환 여부를 소스에서 확인한 뒤 계수의 단위를 신용 점수 단위/입력 1표준편차로 설명하고, 그래프의 특성 이름·축·캡션이 일치하는지 확인한다.
* 조치 결과: Ridge·Lasso 캡션에 실제 Alpha 로그축·표준화 입력 계수축과 입력 1표준편차당 신용 점수 단위를 명시했다. 타깃 원척도와 예측 clipping 단계도 설명했다. [회귀 그림·캡션](../README.md#회귀-성능과-규제)


* **평가 항목 #6 [ ✓ ]**
* 근거: `artifacts/metrics.json > '"random_forest": {' ; artifacts/metrics.json > '"absolute_gain": 0.34305219114732444'`
* 잘한 점: Random Forest 튜닝 결과와 ML 성능 향상 수치(절대·상대)가 기록됨
* 부족한 점: GridSearchCV 탐색 범위(후보 수) 요약이 README에 간단 표로 없음
* 보완: GridSearch 탐색 후보 수와 총 조합 수를 README에 표기해 두기
* 대책: README에 Ridge·Lasso의 Alpha 후보 5개와 RF의 `n_estimators` 2개·`max_depth` 2개·`min_samples_leaf` 3개를 표로 정리한다. 탐색별 전체 조합 수를 회귀 모델별 5개, RF 12개로 표시하고 5-fold 평가 횟수와 최종 재학습을 구분한다. 실제 `ParameterGrid`의 조합 수 및 `verbose=0` 설정과 대조한다.
* 조치 결과: Ridge·Lasso 각 5개와 RF 12개 조합, 파라미터별 후보 수, 5-fold 평가 횟수, 최종 재학습 횟수를 표로 추가했다. [탐색 요약](../README.md#재현성과-데이터-누수-방지), [조합 수·verbose 검증](../modeling.py)


* **평가 항목 #7 [ ✓ ]**
* 근거: `README.md > '# A5_1' ; requirements.txt > 'numpy==1.26.4'`
* 잘한 점: 프로젝트 설명 README와 고정된 requirements.txt가 존재함
* 부족한 점: requirements.txt에 dev/test 도구(예: black, pytest) 표기는 없음(필수는 아님)
* 보완: 실행 재현을 돕는 최소 개발 도구 목록을 README나 requirements-dev에 추가 고려
* 대책: README의 개발·검증 안내에 Python 3.11·`uv`·고정 의존성 설치 방법과 현재 사용하는 `--validate` 및 `validate_report.py` 실행 명령을 정리한다. `pytest`나 `black`은 실제 도입 목적이 생기면 사용할 버전·명령과 함께 선택 개발 도구로 추가하는 방안을 적고, 현재 실행·검증에 필요한 도구가 안내에 포함되는지 확인한다.
* 조치 결과: Python 3.11·uv·고정 의존성·기존 --validate·문서 검증 명령을 최소 개발·검증 도구 표와 단일 실행 블록으로 정리했다. 추가 도구 도입 시 목적·버전·명령의 명세 방법도 적었다. [개발·검증 안내](../README.md#환경과-실행-순서)


* **평가 항목 #8 [ ✓ ]**
* 근거: `README.md > '| 단계 | 수행 내용 | 구현·산출물 |'`
* 잘한 점: README에 전처리→학습→평가 단계별 파일·산출물이 표로 정리되어 있음
* 부족한 점: 함수/스크립트별 역할(입력·출력 예시)은 요약은 있으나 예시 호출문 추가로 이해도를 높일 수 있음
* 보완: 각 주요 함수(make_preprocessor, make_model, predict_*)의 한줄 역할·입출력 예시를 README에 표기
* 대책: README에 `make_preprocessor()`, `make_model(estimator)`, `predict_credit_score(model, features)`, `predict_overdue_probability(model, features)`의 import 경로·한 줄 역할·입력·반환값·최소 호출 예시를 표로 추가한다. `make_model()`이 반환한 Pipeline을 학습하는 단계까지 연결하고, 회귀 예측 배열과 클래스 1의 확률 배열이 입력 행 수와 같은 길이를 갖는지 실제 함수 정의와 대조한다.
* 조치 결과: 4개 주요 API의 import·역할·입력·출력·호출 예시와 Pipeline 학습 단계를 추가했다. 회귀 점수·양성 확률의 길이·유한값·범위를 예제에서 확인한다. [함수 표·실행 예제](../README.md#환경과-실행-순서), [예제 검증](../validate_report.py)


* **평가 항목 #9 [ ✓ ]**
* 근거: `README.md > '모든 CV가 전처리를 포함한 전체 Pipeline을 탐색하여 전처리 통계가 각 학습 폴드에서 fit되게 한다.' ; evaluation.py > 'assert training.index.intersection(split.X_test.index).empty'`
* 잘한 점: CV 내부에서 전처리·SMOTE가 수행되고 Test 분리는 코드로 검증됨
* 부족한 점: 데이터 누수 관련 주요 assert 로그가 실패 시 사용자 안내 메시지는 한줄로 부족함
* 보완: 데이터 누수 관련 assert 실패 시 어떻게 재현/디버그할지 README에 짧게 안내 추가
* 대책: README에 누수 검증 실패 시 traceback의 assert 위치, 재현 명령, `random_state=42` 분할 설정, Train/Validation/Test 인덱스 중복, scaler 학습 표본 수, imputer의 학습 중앙값을 순서대로 확인하는 절차를 추가한다. `modeling.py`와 `evaluation.py`의 주요 assert를 해당 점검 항목에 연결하고, 실패 원인을 확인한 뒤 학습 폴드의 fit 범위를 수정하는 방법을 설명한다.
* 조치 결과: 첫 traceback·재현 명령·시드·인덱스 교집합·scaler 표본 수·imputer 중앙값·CV Pipeline을 순서대로 점검하는 표를 추가했다. [누수 실패 점검](../README.md#재현성과-데이터-누수-방지)


* **평가 항목 #10 [ ✓ ]**
* 근거: `README.md > '| 공통 ML | 6개 입력, 두 타겟 제외, 동일 Train/Test 인덱스, 학습 폴드 전용 대치·스케일링 | 검증·Test에는 학습된 변환 적용 |'`
* 잘한 점: 공통 전처리와 회귀/분류의 개별 처리(알파 CV·SMOTE/가중치)가 문서화되어 있음
* 부족한 점: 회귀·분류별 입력 파이프라인 차이를 코드에서 한눈에 보는 요약이 있으면 좋음
* 보완: README에 회귀용/분류용 파이프라인 흐름도(한줄 요약)를 추가
* 대책: README에 회귀의 공통 전처리→Ridge/Lasso→KFold 기반 Alpha 선택→신용 점수 예측 흐름과 분류의 공통 전처리→클래스 가중치 모델→층화 CV→양성 확률·임계값 판정 흐름을 한 줄씩 추가한다. SMOTE 비교 경로는 학습 폴드 전처리→SMOTE→분류기 순서로 따로 표시하고, 각 요약을 실제 Pipeline 및 CV 구성과 대조한다.
* 조치 결과: 회귀·기본 분류·SMOTE 비교의 전처리→학습→설정 선택→예측 흐름을 실제 Pipeline과 연결한 문장으로 추가했다. [파이프라인 흐름](../README.md#재현성과-데이터-누수-방지)


* **평가 항목 #11 [ ✓ ]**
* 근거: `README.md > 'Precision·Recall·F1·AUC를 함께 해석한다.' ; artifacts/metrics.json > '"train_overdue_rate": 0.120125'`
* 잘한 점: F1/AUC 선택 근거와 불균형 수치(약 12%)가 명시되어 있음
* 부족한 점: 지표 선택에 대한 간단 수학적 근거(예: F1이 불균형에서 왜 유의미한지 한 문장)가 더 있으면 좋음
* 보완: README에 F1/AUC 선택 이유를 1문단으로 더 명확히 기술
* 대책: README에 양성 클래스가 약 12%인 상황에서 `F1 = 2 × Precision × Recall / (Precision + Recall)`이 오탐과 미탐의 균형을 함께 반영한다는 설명을 추가한다. ROC-AUC는 여러 임계값에 걸친 양성·음성의 점수 순위 구분력을 평가한다고 적고, 실제 심사 판단에는 Precision·Recall과 FP/FN 비용을 함께 사용하는 이유를 한 문단으로 연결한다.
* 조치 결과: F1·Precision·Recall 수식과 약 12% 양성 비율에서의 해석, ROC-AUC의 순위 구분력, 운영 비용과 임계값의 관계를 설명했다. [지표 선택 근거](../README.md#분류-성능)


* **평가 항목 #12 [ ✓ ]**
* 근거: `README.md > 'Ridge의 L2 규제는 계수 제곱합을, Lasso의 L1 규제는 계수 절댓값 합을 벌점으로 사용한다.'`
* 잘한 점: L1/L2 차이와 Alpha 영향에 대한 설명과 계수 예시가 제공됨
* 부족한 점: 규제가 실제 예측오차에 미친 영향(숫자 비교)은 있으나 직관적 요약 문장이 추가되면 이해도 향상
* 보완: Alpha 증감이 RMSE/MAE에 미친 방향을 요약 문장으로 README에 추가
* 대책: 기존 Alpha별 Train CV RMSE 표를 근거로 Ridge는 Alpha 0.01~10에서 오차 변화가 작고 Lasso는 Alpha 10·100에서 오차가 커지는 흐름을 README에 요약한다. MAE의 변화 방향은 같은 Train CV 폴드에서 Alpha별 MAE 평균·표준편차를 추가 산출하여 확인한 뒤 수치와 함께 적고, 선택 Alpha의 최종 Test RMSE·MAE와 연결한다.
* 조치 결과: 기존 회귀 CV의 동일 학습에서 Alpha별 MAE 원점수·평균·표준편차를 추가 저장한다. README의 RMSE·MAE 표와 규제 변화 요약은 저장 결과에 맞춰 대조한다. [회귀 구현](../modeling.py), [Alpha별 결과](../artifacts/metrics.json), [오차 비교](../README.md#회귀-성능과-규제)


* **평가 항목 #13 [ ✓ ]**
* 근거: `README.md > 'RF는 단일 트리보다 Validation F1이 0.5531→0.6757로 높고 Train−Validation 격차는 0.1379→0.1147로 작았다.'`
* 잘한 점: 앙상블의 편향/분산 설명과 실험 결과(Train/Validation·격차)가 함께 제시되어 있음
* 부족한 점: 편향·분산 감소에 대한 정량적 분해는 없으나 관찰 근거는 충분함
* 보완: 결과 근거로 폴드간 분산 변화를 한 문장으로 요약해 추가
* 대책: `artifacts/evaluation_metrics.json`에서 같은 5개 검증 폴드로 평가한 단일 트리와 RF의 Validation F1 원점수 및 표준편차를 확인한다. README에 모델별 평균·표준편차와 표준편차의 변화량을 한 문장으로 추가하고, 관측 결과에 따라 폴드 간 성능 변동이 얼마나 줄거나 늘었는지 기술한다. 표준편차의 계산 기준은 `evaluation.py`의 `np.std` 설정과 맞춘다.
* 조치 결과: 동일 5-fold Validation F1 표준편차의 실제 변화량과 np.std(ddof=0) 기준을 추가했다. RF의 폴드 간 변동 증가와 검증 평균·일반화 격차의 개선을 관측값에 맞춰 설명한다. [앙상블 해석](../README.md#앙상블의-편향과-분산), [폴드 원점수](../artifacts/evaluation_metrics.json)


* **평가 항목 #14 [ ✓ ]**
* 근거: `README.md > '리샘플링을 적용할 때는 각 CV 학습 폴드 내부에서 수행해 검증·Test 데이터로 정보가 유입되는 것을 방지한다.'`
* 잘한 점: SMOTE vs class_weight 비교와 Test에 미적용 이유가 문서와 코드에서 명확히 설명됨
* 부족한 점: SMOTE 적용 시 발생하는 잠재적 데이터 분포 변화에 대한 경고 문구가 짧게 추가되면 좋음
* 보완: README에 'SMOTE는 학습전용, Test에는 미적용' 문장을 더 눈에 띄게 표시
* 대책: README의 불균형 처리 설명과 SMOTE 예제 바로 앞에 각 CV 학습 폴드 내부에서만 SMOTE를 적용하고 검증·Test는 원래 표본 구성과 클래스 비율로 평가한다는 문장을 굵게 표시한다. 합성 표본이 학습 분포와 예측확률에 영향을 줄 수 있음을 덧붙이고, 원래 분포의 검증 데이터에서 Precision·Recall·F1·AUC와 비용을 확인하도록 안내한다.
* 조치 결과: SMOTE의 CV 학습 폴드 적용과 검증·Test 원분포 평가를 본문·예제 앞에서 강조하고 합성 표본의 분포·확률 영향과 확인 지표를 추가했다. [SMOTE 적용 범위](../README.md#불균형-처리-비교)


* **평가 항목 #15 [ ✓ ]**
* 근거: `README.md > '운영 가정은 대출 사전심사에서 연체 미탐 손실을 우선 줄이면서 응답시간과 자원 한도를 만족하는 것이다.'`
* 잘한 점: 정확도·속도·자원 제약을 고려한 운영 우선순위와 선택 기준(비용) 제시
* 부족한 점: 운영 제약 수치(예: p95·모델 크기)는 있으나 의사결정 트레이드오프 요약이 한 문단 정도 더 필요함
* 보완: 운영 우선순위(정확도 vs 속도) 선택 사례를 짧은 표로 README에 추가
* 대책: README에 연체 미탐 비용을 우선하는 사전심사와 응답시간을 우선하는 대량 요청 상황을 예시 행으로 둔 선택 표를 추가한다. 후보 모델의 Recall·예상 비용·단건 p95·모델 크기를 기존 실측값과 연결하고, 사전에 정한 응답시간·자원 한도를 만족하는 후보 중 업무 우선순위에 따라 선택하는 근거를 적는다. 가정한 제약과 측정 환경을 함께 표시한다.
* 조치 결과: 비용 우선 사전심사·응답시간 우선 대량 요청 사례에서 LR/RF의 OOF 비용·Recall·임계값·p95·모델 크기와 선택을 표로 제시한다. 예시 한도와 측정 환경도 연결했다. [운영 사례·실측값](../README.md#운영-선택과-비용-임계값), [운영 결과](../artifacts/evaluation_metrics.json)


* **평가 항목 #16 [ ✓ ]**
* 근거: `README.md > '예시 비용을 FP 1단위·FN 10단위로 두면 임계값 0.5인 RF의 총비용은 142 × 1 + 52 × 10 = 662단위다.'`
* 잘한 점: FP·FN의 비즈니스 비용 비교와 임계값 조정 방식(비용 최소화)이 구현·문서화됨
* 부족한 점: 실제 업무 적용 시 비용 민감도(다른 비용 비율 테스트)는 보완 가능
* 보완: 임계값 민감도(FP/FN 비용 비율 변화)에 대한 짧은 예시를 추가하여 의사결정 입력을 강화
* 대책: 예시 FP/FN 비용 비율을 `1:5`, `1:10`, `1:20`으로 두고 동일한 Train 내부 OOF 확률과 임계값 후보에서 `FP × FP비용 + FN × FN비용`을 최소화하는 임계값을 비율별로 선택한다. 비용 비율·선택 임계값·FP·FN·Precision·Recall·총비용을 표로 비교하고 각 비율의 기본 임계값 0.5 결과도 함께 제시한다. Test 평가는 비율별 임계값을 확정한 뒤 수행하며, 비용 비율은 업무 가정임을 명시한다.
* 조치 결과: FP/FN 1:5·1:10·1:20을 동일 Train OOF 확률·후보로 비교하고 모든 임계값을 Test 평가 전에 확정한다. 기본 0.5·선택 임계값의 OOF/Test 비용·FP·FN·Precision·Recall을 저장·문서화한다. [민감도 구현](../evaluation.py), [비율별 결과](../artifacts/evaluation_metrics.json), [비교표](../README.md#운영-선택과-비용-임계값)
# 모델 실행 환경

## 준비 상태

| 모델 | 체크포인트 | 공식 코드/기반 모델 | 현재 연결 |
|---|---|---|---|
| TruFor | `weights/trufor/finetuned/best.pth.tar` | `vendor/trufor/TruFor_train_test` | 파인튜닝 가중치 adapter 연결 완료 |
| FUSED | `weights/fused/finetuned/latest.pth` | `vendor/fused` | 공식 OpenSDI adapter 연결 완료 |
| Recapture | backbone 마지막 2개 layer + MLP head | 로컬 DINOv2-with-registers-base | 공식 adapter 연결 완료 |

체크포인트와 소스는 모두 준비되었습니다. 실제 공식 추론은 NVIDIA CUDA가 노출되는 WSL 환경에서 실행해야 합니다.

## 환경 분리 원칙

세 모델은 라이브러리 버전이 다르므로 하나의 Python 환경에 합치지 않습니다.

### TruFor

```bash
conda env create -p .venvs/trufor -f vendor/trufor/TruFor_train_test/trufor_conda.yaml
```

### FUSED

```bash
python3.12 -m venv .venvs/fused
.venvs/fused/bin/pip install --upgrade pip
.venvs/fused/bin/pip install -e vendor/fused
```

### Recapture

```bash
python3.11 -m venv .venvs/recapture
.venvs/recapture/bin/pip install --upgrade pip
.venvs/recapture/bin/pip install -r models/recapture/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

## 앱에 환경 경로 연결

```bash
export CG_TRUFOR_PYTHON="$PWD/.venvs/trufor/bin/python"
export CG_FUSED_PYTHON="$PWD/.venvs/fused/bin/python"
export CG_RECAPTURE_PYTHON="$PWD/.venvs/recapture/bin/python"
export CG_TRUFOR_CHECKPOINT="$PWD/weights/trufor/finetuned/best.pth.tar"
export CG_FUSED_CHECKPOINT="$PWD/weights/fused/finetuned/latest.pth"
```

`python scripts/check_weights.py`는 소스와 체크포인트 준비 상태를 확인합니다. 실제 추론 결과의 `meta.is_official_model`이 `true`인지도 함께 확인해야 합니다.

## Streamlit 화면의 실제 모델 연결

```bash
.venvs/trufor/bin/python -m pip install -r models/trufor/requirements-inference.txt
.venv/bin/python -m streamlit run app/streamlit_app.py
```

`dist/index.html`은 Streamlit 양방향 컴포넌트로 제공됩니다. 이미지를 선택하고
**분석 시작**을 누르면 Python이 모든 이미지에 대해 TruFor → FUSED → Recapture를
순차 실행합니다. 화면에서 파일별 모델 점수, 실행 시간, FUSED 히트맵을
확인하고 실제 결과 JSON을 내려받을 수 있습니다. HTML만 단독으로 열면 모델은
실행되지 않습니다.

- 최대 20장, 이미지당 20MB, 전체 100MB까지 지원합니다.
- 업로드와 추론 JSON/산출물은 `data/runtime/`에 저장됩니다.
- 청구건은 해당 브라우저의 IndexedDB에 원본 표시 이미지·히트맵·점수·분석 당시 임계값을 함께 저장합니다.
- 청구건 사진을 누르면 원본/의심 영역, 세 모델 점수, 통합 점수와 임계값 비교를 다시 확인할 수 있습니다.
- 검토대상/정상추정 분류는 저장된 임계값을 기준으로 하며, 미완료 분석은 별도로 표시합니다.
- 청구 목록 또는 상세 화면에서 저장된 청구건과 사진을 삭제할 수 있습니다.
- 이전 localStorage의 점수 기록은 자동으로 옮깁니다. 이전 기록에 사진이 없으면 재분석·저장이 필요합니다.
- 화면 연결은 `CG_STRICT_MODE=1`로 실행합니다. 모델 하나라도 실패하면 해당
  점수와 융합 점수는 비워 두고 오류를 표시하며, 정상 판정하지 않습니다.
- FUSED는 이미지별 별도 프로세스에서 실행하고 종료하여 다음 모델 실행 전에
  GPU 메모리를 반환합니다. 최초 로딩 시간도 진행 화면에 포함됩니다.
- 첫 화면으로 돌아가면 남은 모델 실행을 취소합니다. 이미 실행 중인 모델은
  종료 또는 제한 시간 도달 후 해제되며 결과가 화면을 덮어쓰지 않습니다.
- 실제 GPU 추론에는 CUDA에 접근 가능한 실행 환경이 필요합니다.

## ROI 임계값 설정

첫 화면의 **현재 ROI 임계값** 버튼에서 설정 화면으로 이동합니다. 저장된 청구건 중
분석이 완료되고 청구액이 있는 건을 학습 표본으로 사용합니다.

- 청구건에 사진이 여러 장이면 가장 높은 통합 점수를 대표 점수로 사용합니다.
- 예상손실 계산에는 표본 청구건들의 평균 청구액을 사용하며, 설정 화면에서 평균 청구액을 직접 입력할 수도 있습니다.
- 검토비용과 미탐지 예상손실 비율은 사용자가 입력합니다.
- 실제 조작 여부는 사후 조사로 확정되는 값이므로 각 청구건에 0 또는 1로 입력합니다.
- 후보 임계값 0%~100%의 예상비용을 비교해 비용이 가장 낮은 값을 권장합니다.
- 적용한 임계값은 브라우저 저장소에 보관되고 이후 새 분석에 사용됩니다. 기존 분석은
  감사 추적을 위해 당시 임계값을 유지합니다.

실제 모델 연결 검증(실패 시 폴백 없이 종료):

```bash
.venv/bin/python scripts/smoke_test_ui.py
# 실제 사진으로 확인하려면 --image /path/to/photo.jpg 추가
```

## 기존 스크립트의 데모 폴백 (Streamlit 실제 분석에서는 사용하지 않음)

- TruFor 공식 실행 실패: ELA + 노이즈 잔차 폴백
- FUSED 공식 실행 실패: 게이트웨이의 결정적 데모 점수
- Recapture 공식 실행 실패: FFT 주기성 폴백

폴백은 시스템 배선과 UI 시연을 위한 것이며 모델 성능 평가에 사용하면 안 됩니다.

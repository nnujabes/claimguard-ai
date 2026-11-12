# ClaimGuard Vision

**자동차보험 사고 사진의 위변조 부위를 표시해 심사를 돕는 이미지 포렌식 시스템**

> 조작 사진을 한눈에, 더 빠르고 정밀한 보험 심사의 시작

동국대 AI캠퍼스 · DAIVE 블루팀 (4인) · 2026년 10월

`Python` `PyTorch` `TruFor` `FUSED` `DINOv2 (Recapture)` `OpenCV` `Streamlit`

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [시스템 구성](#2-시스템-구성)
3. [모델과 파인튜닝](#3-모델과-파인튜닝)
4. [판정 로직](#4-판정-로직)
5. [빠른 실행](#5-빠른-실행)
6. [가중치 배치](#6-가중치-배치-저장소-미포함)
7. [실제 모델 연결](#7-실제-모델-연결)
8. [FUSED 파인튜닝 재현](#8-fused-파인튜닝-재현-wsl)
9. [검증](#9-검증)
10. [저장소 구성](#10-저장소-구성)
11. [팀원별 작업](#11-팀원별-작업)
12. [한계와 개선 계획](#12-한계와-개선-계획)

---

## 1. 프로젝트 개요

### 배경

- **생성형 AI로 만든 손상 조작은 육안으로 판별하기 어렵습니다.** AI가 픽셀·질감·반사까지 새로 그려 넣기 때문에 기존 편집 흔적만으로는 구분이 안 됩니다.
- **보험 업무의 비대면화.** 모바일·웹으로 사고 접수와 사진 제출이 이뤄지면서 조작 사진이 그대로 심사에 들어올 수 있습니다.
- **보험사기 규모.** 2025년 보험사기 적발금액은 **1조 1,571억 원**이며 그중 자동차보험이 **22.4%** 입니다(미적발 포함 추산 약 9조 원).
- 금융당국은 2026.6.4 「AI 기반 보험사기 방지체계 구축 TF」를 출범했지만, 패턴 분석·위험지수 중심으로 **이미지 포렌식 적용은 아직 없습니다.**

### 목표

AI가 사기를 **확정**하는 것이 아니라, 위변조 의심 부위를 표시해 **심사 담당자의 승인·반송 판단을 돕는** 웹 서비스를 만드는 것입니다. 의심 건은 원본 사진 요청·현장 사진 대조 같은 추가 조사로 이어지고, 최종 판단과 보험금 결정은 심사 담당자가 합니다.

### 사용 흐름

1. **사진 업로드** — 고객이 보험 청구 사진을 업로드
2. **3가지 위변조 유형 분석** — 서로 다른 흔적을 보는 3개 모델이 각각 분석
   - 포토샵 짜깁기·복제·객체 삭제 등 전통적 편집 → **TruFor**
   - 생성형 AI가 새로 그려 넣은 부분(Inpainting) → **FUSED**
   - 컴퓨터 화면을 다시 촬영한 사진(모아레 패턴 등) → **Recapture**
3. **심사 담당자 조회 및 판정** — 모델별 점수와 의심 영역 히트맵을 확인하고 승인 또는 반송

---

## 2. 시스템 구성

```text
 ┌─ 고객 화면 ─────────┐   ┌─ 내부 분석 ──────────────────────────────┐   ┌─ 심사 담당자 화면 ─┐
 │ 청구 신청·이미지 등록 │ → │ model_gateway.py                          │ → │ 청구 목록·필터      │
 └─────────────────────┘   │   TruFor → FUSED → Recapture (GPU 순차)   │   │ 이미지별 상세·히트맵 │
                           │ fusion.py / decision.py (점수·라우팅)      │   │ ROI 임계값 설정     │
                           │ repository.py (data/results 저장)         │   └────────────────────┘
                           └──────────────────────────────────────────┘
```

- **오케스트레이터** (`backend/`): Streamlit 앱 프로세스는 모델을 직접 import하지 않습니다. `model_gateway.py`가 모델별 독립 Python 환경을 subprocess로 호출하고, 각 모델은 `contracts/model_io_schema.json` 형식의 JSON을 반환합니다.
- **GPU 순차 점유**: 8GB VRAM 제약 때문에 한 번에 한 모델만 GPU를 쓰고, 끝나면 메모리를 반납합니다.
- **추론 속도 최적화**: 3개 모델을 매번 새로 띄우면 1장당 약 18초가 걸렸습니다. FUSED를 첫 요청 후 전용 프로세스에 상주시키고 비동기 처리를 적용해 **1장당 5초 이내**로 줄였습니다(목표 10초 이내).
- **모델별 독립 환경**: 세 모델은 Python·PyTorch 버전이 서로 달라 각각 독립 venv/프로세스로 분리했습니다.
- **화면**: `dist/`의 웹 UI를 Streamlit 양방향 컴포넌트로 제공합니다. 고객 접수 → 내부 분석 → 담당자 심사로 화면을 나눴습니다.

---

## 3. 모델과 파인튜닝

| 모델 | 역할 | 출력 | 학습 | 출처 |
|---|---|---|---|---|
| **TruFor** | 전통적 편집 탐지 (스플라이싱·리터치·객체 삭제) | Localization map · Heatmap | 파인튜닝 · best epoch 4 | [grip-unina/TruFor](https://github.com/grip-unina/TruFor) |
| **FUSED** | 생성형 AI 인페인팅 탐지 (2026년 최신 모델) | Probability map · Binary mask | 파인튜닝 · best epoch 24 | [AntonNuzhdin/FUSED](https://github.com/AntonNuzhdin/FUSED) |
| **Recapture** | 화면 재촬영 탐지 (모아레 패턴) | 재촬영 확률 | 사전학습 그대로 | [UserPollo/moire-pattern-detector](https://huggingface.co/UserPollo/moire-pattern-detector), [facebook/dinov2-with-registers-base](https://huggingface.co/facebook/dinov2-with-registers-base) |

한 모델이 놓칠 수 있는 조작 유형을 다른 모델이 보완하는 구조입니다.

### 학습 데이터 (1,503장)

- **수집**: AI Hub 차량 파손 데이터 중 범퍼·펜더·헤드램프·후드 등 차량 부품 10종
  (앞범퍼, 앞펜더, 전조등, 뒷범퍼, 뒷펜더, 앞도어, 뒷도어, 사이드미러, 본넷, 트렁크)
- **구성**: 원본(org) 501 · 편집본(edi) 501 · 마스크(msk) 501 = **501세트, 총 1,503장**
- **분할**: 원본을 perceptual hash(dHash)로 묶어 Train/Val/Test **80/10/10** — 한 원본의 파생본이 다른 split으로 새는 것을 방지
- **공유**: FUSED·TruFor가 같은 데이터셋을 쓰고, 모델별 매니페스트로만 변환

### 시행착오: 조작 이미지 제작

AI 편집 이미지로 학습을 **두 번 실패**했습니다. 편집본 전체가 원본과 재질·조명·화질이 달라져서, 모델이 조작 흔적이 아니라 촬영 조건 차이를 배웠기 때문입니다.

1. 조작할 부위를 **이진 마스크로 먼저 정의**하고 그 안에서만 생성 → 정확한 위치 정답(ground truth) 확보
2. **마스크 바깥 픽셀은 원본으로 강제 복원** — `edited = (마스크 내부) 생성 결과 + (마스크 외부) 원본 픽셀`
3. 원본·편집본·마스크를 **한 세트로 유지**해 분류(real/fake)와 위치 예측(localization)을 같이 학습

### 파인튜닝 설정

| | TruFor | FUSED |
|---|---|---|
| 데이터 | 1,503장 (org·edi·msk 트리플릿) | 1,503장 (원본·편집 이미지 쌍) |
| 분할 | 8:1:1 · seed 42 | 8:1:1 |
| 입력 | 원본 크기 (crop) | 512 × 512 |
| Epoch | 30 | 30 |
| Optimizer | SGD · lr 0.001 (점진 감소) | AdamW · lr 3e-5 |
| Batch | 2 | 1 |
| 동결층 | NP++ · MiT-B2 backbone | ConvNeXt XXL backbone |
| 학습층 | Localization · Confidence · Detection head | SparseViT, MoE (분류·영역 분할) |
| 선택 체크포인트 | Epoch 4 · `best.pth.tar` | Epoch 24 · `latest.pth` |

### 파인튜닝 전후 성능 (검증셋)

| 모델 · 지표 | 파인튜닝 전 (Epoch 0) | 파인튜닝 후 (best) | 변화 |
|---|---:|---:|---:|
| TruFor · Localization F1 | 0.28 | **0.99** | +0.71 |
| TruFor · Detection AUC | 0.76 | **0.99** | +0.23 |
| FUSED · Localization F1 | 0.31 | **0.82** | +0.51 |
| FUSED · Detection AUC | 0.81 | **0.99** | +0.18 |

> 검증셋은 같은 방식으로 만든 조작 이미지라 실제 청구 사진에서의 성능과는 다를 수 있습니다.

---

## 4. 판정 로직

### 점수 결합: 가중평균이 아닌 최대 점수(Max)

1. 모델별 추론: TruFor(파인튜닝) · FUSED(파인튜닝) · Recapture
2. **이미지 점수** = `max(세 모델 점수)`
3. **청구건 대표 점수** = `max(모든 이미지 점수)`
4. 대표 점수 ≥ 임계값 → **검토대상** (`HUMAN_REVIEW`), 미만 → **정상추정** (`AOS`)

한 모델의 강한 위험 신호가 평균 과정에서 묻히지 않도록 보수적으로 결합합니다. 미탐은 줄지만, 특정 모델의 오탐이 전체 판정에 영향을 줄 수 있다는 trade-off가 있습니다.

- 기본 임계값은 `config/runtime.json`의 `roi_threshold`(0.63)이고, 화면에서 바꿀 수 있습니다.
- 참고용 가중 결합 점수(TruFor 0.35 + FUSED 0.50 + Recapture 0.15)도 `backend/fusion.py`에서 계산해 결과 JSON에 저장합니다.
- 모델이 하나라도 실패하면(`CG_STRICT_MODE=1`) 점수를 비워 두고 **분석 미완료**로 표시하며, 정상으로 판정하지 않습니다.

### ROI 기반 임계값 최적화

첫 화면의 **현재 ROI 임계값** 버튼에서 설정합니다.

```text
Total Cost(t) = 검토 건수(t) × 검토비용 + 조작 미탐 건수(t) × 미탐 예상손실
```

- 입력: 검토비용 · 평균 청구액 · 미탐 예상손실 비율, 청구건별 실제 조작 여부(사후 조사 결과, 0/1)
- 탐색: 임계값 0.00 ~ 1.00을 0.01 간격으로 계산해 **총비용이 최소인 값**을 권장
- 적용한 임계값은 이후 새 분석부터 쓰이고, 기존 분석은 감사 추적을 위해 당시 임계값을 유지

---

## 5. 빠른 실행

Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
.\scripts\run_windows.ps1
```

설치 없이 화면만 보려면 `dist/index.html`을 브라우저로 엽니다. 이 정적 데모는 업로드·결과·심사 콘솔 흐름 확인용이고, 점수는 브라우저 안의 결정적 시뮬레이션입니다(모델은 실행되지 않음).

실제 모델로 분석하려면 [6. 가중치 배치](#6-가중치-배치-저장소-미포함)와 [7. 실제 모델 연결](#7-실제-모델-연결)을 먼저 진행하세요.

- 최대 20장, 이미지당 20MB, 전체 100MB (JPEG/PNG)
- 업로드와 추론 결과는 `data/runtime/`, 결과 JSON은 `data/results/`에 저장
- 청구건(원본 이미지·히트맵·점수·당시 임계값)은 브라우저 IndexedDB에 저장

---

## 6. 가중치 배치 (저장소 미포함)

> **GitHub 저장소에는 모델 가중치가 없습니다.** GitHub의 100MB 파일 제한 때문에
> `*.pth`, `*.pth.tar`, `*.pt`, `*.safetensors`, `*.ckpt`, `*.bin`, `*.onnx` 파일과 `downloads/` 폴더를
> `.gitignore`로 제외했습니다. 클론 후 팀 공유 `models.zip`(또는 별도 전달된 파일)에서 아래 경로에 직접 넣어 주세요.

| 배치 경로 | 용도 | 크기 |
|---|---|---|
| `weights/trufor/finetuned/best.pth.tar` | TruFor 파인튜닝 체크포인트 | 약 275MB |
| `weights/trufor/trufor.pth.tar` | TruFor 원본 체크포인트 | 약 269MB |
| `weights/fused/finetuned/latest.pth` | FUSED 파인튜닝 체크포인트 | 약 305MB |
| `weights/fused/fused_opensdi.pth` | FUSED 기본(OpenSDI) 체크포인트 | 약 305MB |
| `weights/recapture/best_screen_detector_backbone.pt` | Recapture backbone 마지막 2개 layer | 약 54MB |
| `weights/recapture/best_screen_detector_mlp.pt` | Recapture MLP head | 약 3MB |
| `downloads/recapture_dinov2/dinov2-with-registers-base/` | DINOv2-with-registers-base (`config.json`, `model.safetensors`) | 약 330MB |
| `vendor/trufor/TruFor_train_test/pretrained_models/segformers/mit_b2.pth` | TruFor 학습용 SegFormer 사전학습 가중치 (학습 시에만 필요) | 약 94MB |

파일별 SHA-256은 `weights/manifest.json`에 있습니다. 배치 후 `scripts/check_weights.py`로 **파일 존재**와 **공식 추론 실행 가능 상태**를 확인하세요.

| 모델 | 공식 추론이 안 될 때 (기존 스크립트 데모용) |
|---|---|
| TruFor | ELA + 노이즈 잔차 폴백 |
| FUSED | 게이트웨이의 결정적 데모 점수 |
| Recapture | FFT 주기성 폴백 |

폴백 결과는 JSON의 `meta.backend`와 `meta.is_official_model=false`로 표시되며, Streamlit 실제 분석 화면에서는 쓰지 않습니다. 모델 성능 평가에 사용하면 안 됩니다.

---

## 7. 실제 모델 연결

세 모델은 라이브러리 버전이 달라 독립 Python 환경에서 실행합니다. 실제 추론은 NVIDIA CUDA가 보이는 환경(WSL 권장)이 필요합니다.

| 모델 | Python | 주요 의존성 |
|---|---|---|
| TruFor | 3.8 | torch 1.11.0+cu113, timm 0.5.4, yacs |
| FUSED | 3.11 ~ 3.12 | torch 2.9.1+cu126, timm 1.0.24, opencv, scikit-learn |
| Recapture | 3.11 | torch 2.9.1+cu126, transformers ≥ 4.48, huggingface-hub ≥ 0.24 |

각 모델은 stdout 마지막 줄에 `contracts/model_io_schema.json` 형식의 JSON을 출력합니다. 인터프리터와 가중치는 환경 변수로 지정합니다(`.env.example` 참고).

```text
CG_TRUFOR_PYTHON          CG_TRUFOR_CHECKPOINT
CG_FUSED_PYTHON           CG_FUSED_CHECKPOINT
CG_RECAPTURE_PYTHON       CG_RECAPTURE_CHECKPOINT_DIR
```

FUSED는 첫 요청에서만 체크포인트를 로드한 뒤 전용 프로세스에 상주합니다. 기본/파인튜닝 비교처럼 체크포인트가 바뀌면 자동으로 재시작합니다. 요청별 실행 방식이 필요하면 `CG_FUSED_PERSISTENT=0`을 지정하세요.

환경별 설치 명령은 [`MODEL_SETUP.md`](MODEL_SETUP.md)를 참고하세요.

---

## 8. FUSED 파인튜닝 재현 (WSL)

학습 데이터는 아래 파일명 규칙을 따르며, 모든 이미지는 학습 시 512×512로 변환합니다.

```text
data/fused_source/
  f_bmp01_1_org.jpg    # 원본
  f_bmp01_2_edi.png    # 편집본
  f_bmp01_3_msk.png    # 마스크
```

프로젝트를 WSL에 복사하고 학습 이미지를 `data/fused_source/`에 넣은 뒤 실행합니다.

```bash
bash scripts/fused_wsl_all_in_one.sh
```

설정은 30 epochs, 전체 FUSED 구조, 8GB VRAM 안전 프로필로 고정되어 있습니다. 데이터 누수·손상 파일·빈 마스크·OOM 사전 조건·과열·디스크 부족·중단 후 재개를 자동으로 점검합니다.

---

## 9. 검증

```powershell
.\.venv\Scripts\python.exe scripts\check_weights.py
.\.venv\Scripts\python.exe scripts\smoke_test.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

실제 모델 연결 검증(실패 시 폴백 없이 종료):

```bash
.venv/bin/python scripts/smoke_test_ui.py
# 실제 사진으로 확인하려면 --image /path/to/photo.jpg 추가
```

---

## 10. 저장소 구성

```text
app/          Streamlit 앱 (분석 + 심사 담당자 콘솔)
backend/      순차 모델 실행(model_gateway), 점수 결합(fusion), 판정·ROI(decision), 결과 저장(repository)
models/       TruFor · FUSED · Recapture 실행 진입점과 어댑터
vendor/       TruFor · FUSED 공식 저장소 (FUSED는 로컬 수정본 포함)
training/     FUSED 학습 파이프라인
contracts/    모델 공통 JSON 입출력 스키마
config/       런타임 설정 (모델 순서, 가중치, 임계값, 제한 시간)
scripts/      설치 · 실행 · 점검 · 학습 스크립트
tests/        핵심 로직 · UI 테스트
dist/         웹 UI (Streamlit 컴포넌트 / 정적 데모)
weights/      가중치 배치 위치와 manifest (가중치 파일은 미포함)
downloads/    Recapture DINOv2 backbone (미포함, 별도 배치)
```

---

## 11. 팀원별 작업

전통적인 역할 분담 대신, 단위 주제별 **스프린트(애자일)** 방식으로 10일간 진행했습니다. 시장조사·모델 조사/검증·시스템 설계는 4명이 함께 했습니다.

| 팀원 | 담당 | 주요 결과물 | 서비스에 쓰인 곳 |
|---|---|---|---|
| 한병준 (팀장) | AI 생성·리캡처 탐지, TruFor 파인튜닝·웹 화면(공동) | Recapture(모아레) 모델 선정·연동, TruFor 파인튜닝 체크포인트, 웹 화면 | `models/recapture/`, `models/trufor/`, `dist/` |
| 박하영 | 조작 이미지 생성, FUSED 모델, 웹 화면(공동) | AI 편집 이미지·마스크 생성, FUSED 파인튜닝, 웹 화면 | `models/fused/`, `training/`, `weights/fused/`, `dist/` |
| **조혁준** | **추론 경량화, 학습 데이터 구축, TruFor** | 상주 프로세스 추론 게이트웨이, 데이터셋 구조·분할, TruFor 파인튜닝 체크포인트(한병준과 공동), 모델별 독립 환경 | `backend/model_gateway.py`, `models/*/adapter/`, `scripts/prepare_fused_dataset.py`, `models/trufor/`, `MODEL_SETUP.md` |
| 김병수 | 웹 화면(한병준·박하영과 공동), 모델 연동 검증 | 고객·내부 분석·심사 담당자 화면, 모델 결과 연동 검증 | `dist/`, `app/` |

### 조혁준 — 추론 경량화 · 데이터 구축 · TruFor

**1. 추론 속도 개선: 1장당 약 18초 → 5초 이내** (`backend/model_gateway.py`, `models/*/adapter/infer.py`)

- 문제: 처음에는 사진 1장마다 세 모델을 각각 새 프로세스로 띄웠습니다. 매번 체크포인트를 다시 읽어서 1장당 약 18초가 걸렸고, 여러 장을 올리면 실제 심사에 쓸 수 없는 속도였습니다.
- 해결
  - 모델별로 **메모리 상주 서버 프로세스**를 두었습니다. 첫 요청에서만 체크포인트를 로드하고, 이후 요청은 같은 프로세스를 재사용합니다(`_run_fused_persistent`, `_run_trufor_persistent`, `_run_recapture_persistent`).
  - 체크포인트가 바뀌면(기본/파인튜닝 비교 등) 기존 서버를 종료하고 새 체크포인트로 자동 재시작합니다.
  - 8GB VRAM 안에서 돌도록 **GPU 순차 점유**(TruFor → FUSED → Recapture)를 지키고, 화면 쪽은 비동기로 진행 상황을 받도록 했습니다.
  - 문제 진단용으로 예전 방식(요청별 실행)도 `CG_<MODEL>_PERSISTENT=0`으로 남겨 두었습니다.
- 결과: 3개 모델 기준 **1장당 5초 이내** (목표 10초 이내)

**2. 파인튜닝 데이터 수집 · 폴더 구조 설계** (`scripts/prepare_fused_dataset.py`, `data/`)

- AI Hub 차량 파손 검증 데이터에서 부품 10종을 골라 원본을 수집했습니다.
- 원본(org) · 편집본(edi) · 마스크(msk)를 한 세트로 묶는 파일명 규칙(`f_bmp01_1_org.jpg` / `_2_edi.png` / `_3_msk.png`)과 메타데이터 스키마를 설계했습니다. 501세트, 총 1,503장입니다.
- 한 원본에서 여러 파생본이 나와 데이터 누수 위험이 있었습니다. 원본을 **dHash로 그룹화**해 같은 그룹이 같은 split에 들어가도록 Train/Val/Test를 80/10/10으로 나눴습니다.
- FUSED와 TruFor가 같은 데이터셋을 쓰고 모델별 매니페스트로만 변환하도록 해, 두 모델의 학습 조건을 맞췄습니다.

**3. TruFor 파인튜닝 · 검증 (한병준과 공동)** (`models/trufor/`, `weights/trufor/finetuned/`)

- 한병준과 함께 1,503장으로 30 epoch 파인튜닝했습니다. backbone(NP++ · MiT-B2)은 동결하고 Localization · Confidence · Detection head만 학습했습니다(SGD, lr 0.001, batch 2).
- 검증셋 기준 Epoch 4를 best checkpoint로 골랐습니다.
  - Localization F1 **0.28 → 0.99** (+0.71)
  - Detection AUC **0.76 → 0.99** (+0.23)
- 파인튜닝 체크포인트를 서비스 추론 어댑터(`models/trufor/adapter/infer.py`)에 연결했습니다.

**4. 모델별 독립 venv · 프로세스 분리** (`MODEL_SETUP.md`, `scripts/setup_*.ps1`, `scripts/setup_fused_wsl.sh`)

- 세 모델의 의존성이 서로 충돌합니다(TruFor: Python 3.8 · torch 1.11 / FUSED·Recapture: Python 3.11 · torch 2.9).
- 모델마다 독립 venv를 만들고, 앱은 모델을 import하지 않고 subprocess로 호출하도록 분리했습니다. 모델 간 입출력은 공통 JSON 계약(`contracts/model_io_schema.json`)으로 맞췄습니다.


### 진행 일정 (10일)

| 단계 | 주요 작업 | 일정 |
|---|---|---|
| 사전기획 | 아이디어 결정 및 목표 정의 | 1일차 |
| 데이터 수집 | AI Hub 차량 파손 공공데이터 수집 | 2일차 |
| 데이터 전처리 | AI 편집 이미지·마스크 구성, 학습 데이터 분할 | 3~4일차 |
| 모델 선정 | AI 편집·포토샵·리캡처 역할별 모델 3개 선정 | 5~6일차 |
| 모델 파인튜닝 | FUSED · TruFor 파인튜닝 | 7~8일차 |
| UI/UX · 구현 · 결합 검증 | Streamlit UI/UX, 백엔드 모델 연동 | 8~9일차 |
| Merge | 팀원별 작업 병합 및 마무리 | 9~10일차 |

---

## 12. 한계와 개선 계획

**한계**

- **학습 데이터 규모**: 30 epoch까지 실험해 보니 반복 학습보다 데이터의 양과 다양성이 성능의 핵심이었습니다. 약 500세트로는 다양한 AI 조작 패턴과 사고 유형을 모두 다루기 어렵습니다.
- **부위 범위**: 기간 내 구현을 위해 주요 부위 10종 위주로 학습해, 전 부위의 미세 손상 판독에는 한계가 있습니다.
- **사진 간 비교**: 한 청구건의 여러 사진끼리 손상 부위·각도가 맞는지 대조하는 기능은 아직 없습니다.

**개선 계획**

- 수만~수십만 장 규모의 손상 사진과 신종 생성형 AI 조작 데이터를 확보하고, 탐지 영역을 차량 전 부위로 확대
- 같은 청구건 안의 여러 사진에 대해 손상 위치·빛·각도의 정합성을 검증하는 교차 분석
- 보험사 손상 판독/수리비 산정 시스템(AOS) 및 사기 탐지 시스템(FDS)과 REST API 기반 실시간 연동

---

> 이 시스템의 결과는 보험사기 확정 판정이 아니라 심사 담당자의 추가 검토를 돕는 참고 정보입니다.
> 발표 자료의 사례 이미지는 AI로 생성한 예시이며 실제 청구 내용이 아닙니다.

# ClaimGuard Vision — 팀 공유 패키지

자동차보험 청구 사고사진의 AI 인페인팅, 일반 편집, 화면 재촬영 흔적을 분석하고 심사 우선순위를 제안하는 실행형 프로토타입입니다.

## 1. 5분 실행

Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
.\scripts\run_windows.ps1
```

설치 없이 화면만 확인하려면 `dist/index.html`을 브라우저로 엽니다. 이 정적 데모는 업로드·결과·심사 콘솔 흐름을 확인하기 위한 것으로, 점수는 브라우저 내부의 결정적 시뮬레이션입니다.

## 2. 패키지 구성

```text
app/                 Streamlit 과제용 분석 + 업무담당자 콘솔
backend/             순차 모델 실행, score fusion, ROI 판정, 결과 저장
models/              TruFor/FUSED/Recapture 실행 진입점과 폴백
vendor/              TruFor·FUSED 공식 저장소 (FUSED는 로컬 수정본 포함)
downloads/           Recapture DINOv2 base backbone (저장소 미포함, 별도 배치)
training/            501세트 규칙 기반 FUSED 학습 파이프라인
contracts/           모델 공통 JSON 입출력 계약
config/              런타임 설정
scripts/             설치·실행·점검·패키징 스크립트
tests/               핵심 로직 테스트
dist/                팀 공유용 브라우저 데모
weights/             가중치 배치 위치와 상태 명세 (가중치 파일은 저장소 미포함)
```

## 3. 가중치에 관한 중요한 사실

> **GitHub 저장소에는 모델 가중치가 포함되어 있지 않습니다.** GitHub의 100MB 파일 제한 때문에
> `*.pth`, `*.pth.tar`, `*.pt`, `*.safetensors`, `*.ckpt`, `*.bin`, `*.onnx` 파일과 `downloads/` 폴더는
> `.gitignore`로 제외했습니다. 클론 후 팀 공유 `models.zip`(또는 별도 전달된 파일)에서 아래 경로에 직접 배치하세요.

| 배치 경로 | 용도 | 크기 |
|---|---|---|
| `weights/trufor/finetuned/best.pth.tar` | TruFor 파인튜닝 체크포인트 | 약 275MB |
| `weights/trufor/trufor.pth.tar` | TruFor 원본 체크포인트 | 약 269MB |
| `weights/fused/finetuned/latest.pth` | FUSED 파인튜닝 체크포인트 | 약 305MB |
| `weights/fused/fused_opensdi.pth` | FUSED 기본(OpenSDI) 체크포인트 | 약 305MB |
| `weights/recapture/best_screen_detector_backbone.pt` | Recapture backbone 마지막 2개 layer | 약 54MB |
| `weights/recapture/best_screen_detector_mlp.pt` | Recapture MLP head | 약 3MB |
| `downloads/recapture_dinov2/dinov2-with-registers-base/` | DINOv2-with-registers-base (`config.json`, `model.safetensors`) — Hugging Face `facebook/dinov2-with-registers-base` | 약 330MB |
| `vendor/trufor/TruFor_train_test/pretrained_models/segformers/mit_b2.pth` | TruFor 학습용 SegFormer 사전학습 가중치 (학습 시에만 필요) | 약 94MB |

파일별 SHA-256은 `weights/manifest.json`에 있습니다. 배치 후 `scripts/check_weights.py`로 확인하세요.

`scripts/check_weights.py`는 **파일 존재**와 **공식 추론 실행 가능 상태**를 따로 검사합니다.

| 모델 | 기본 경로 | 가중치가 없을 때 |
|---|---|---|
| TruFor | `weights/trufor/finetuned/best.pth.tar` | 공식 저장소·CUDA가 없으면 ELA + 노이즈 잔차 폴백 |
| FUSED | `weights/fused/finetuned/latest.pth` | 공식 `eval.opensdi` 구현·CUDA가 필요 |
| Recapture | `weights/recapture/best_screen_detector_*.pt` | DINOv2 base backbone·CUDA가 없으면 FFT 폴백 |

공식 TruFor 코드와 공식 FUSED 코드(로컬 수정 포함)는 `vendor/`에 일반 파일로 포함되어 있습니다. 위 가중치를 배치하면 소스·체크포인트 자산이 모두 준비됩니다. 실제 공식 추론에는 모델별 Python 의존성과 CUDA 환경이 추가로 필요합니다. 폴백 결과는 JSON의 `meta.backend`와 `meta.is_official_model=false`에 표시되어 실제 모델 결과와 혼동되지 않습니다.

## 4. FUSED 30-epoch 파인튜닝 (WSL 권장)

요구사항 시트의 규칙을 그대로 지원합니다. 모든 이미지는 학습 시 512×512로 변환합니다.

```text
data/
  f_bmp01_1_org.jpg
  f_bmp01_2_edi.png
  f_bmp01_3_msk.png
```

지원 부위: 앞범퍼, 앞펜더, 전조등, 뒷범퍼, 뒷펜더, 앞도어, 뒷도어, 사이드미러, 본넷, 트렁크. 명세 수량은 총 501세트입니다.

프로젝트를 WSL의 `/home/ai/새폴더`에 복사하고, 압축을 푼 학습 이미지를 프로젝트 내부 `data/fused_source/`에 넣습니다. 그다음 인자 없이 실행합니다.

```bash
cd /home/ai/새폴더
bash scripts/fused_wsl_all_in_one.sh
```

설정은 30 epochs, 전체 FUSED 구조, 8GB VRAM 안전 프로필로 고정되어 있습니다. 데이터 누수·손상 파일·빈 마스크·OOM 사전 조건·과열·디스크 부족·중단 후 재개까지 자동 점검합니다.

## 5. 실제 모델 연결

각 모델은 독립 Python 환경에서 실행되고 stdout 마지막 줄에 `contracts/model_io_schema.json` 형식의 JSON을 출력해야 합니다. 다음 환경 변수로 인터프리터와 가중치를 지정합니다.

```text
CG_TRUFOR_PYTHON
CG_FUSED_PYTHON
CG_RECAPTURE_PYTHON
CG_TRUFOR_CHECKPOINT
CG_FUSED_CHECKPOINT
CG_RECAPTURE_CHECKPOINT_DIR
```

앱 프로세스는 모델을 직접 import하지 않습니다. `backend/model_gateway.py`가 TruFor → FUSED → Recapture를 순차 실행하며, GPU 연산은 동시에 수행하지 않습니다.

FUSED는 첫 요청에서만 체크포인트를 로드한 뒤 전용 프로세스에 상주하며, 이후 요청은 같은 프로세스를 재사용합니다. 기본/파인튜닝 비교처럼 체크포인트가 바뀌면 기존 서버를 종료하고 선택한 체크포인트로 자동 재시작합니다. 문제 진단을 위해 예전의 요청별 실행 방식이 필요하면 앱 실행 전에 `CG_FUSED_PERSISTENT=0`을 지정할 수 있습니다.

모델별 WSL 환경 구성은 `MODEL_SETUP.md`를 참고하세요.

## 6. 검증

```powershell
.\.venv\Scripts\python.exe scripts\check_weights.py
.\.venv\Scripts\python.exe scripts\smoke_test.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 7. 판정 원칙

- 기본 score fusion: TruFor 0.35 + FUSED 0.50 + Recapture 0.15
- 기본 ROI 임계값: 0.63
- 임계값 이상: `HUMAN_REVIEW`
- 임계값 미만: `AOS`
- 결과는 보험사기 확정 판정이 아니라 추가 검토 지원 정보입니다.

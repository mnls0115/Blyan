학습을 가지는 블록체인 AI: DAG + MoE 구조 기본 백서

1. 포스켜인 (리턴)
Web3 모델은 매우 방향성이 가능한 것처럼 개념적으로 들리지만, 모델 테이블의 모델 구조와 블록체인의 관계를 정리하면서, 고용되고 현실적인 "명목적 프로세스" 구조를 제안합니다.

2. 개인 모델의 무건적 (MoE)과 블록체인의 충돌
전통 블록체인 / MoE에 맞지 않는 이유
선형 구조 (1 → 2 → 3...)
모든 블록을 순차로 읽어야함

모든 weight를 하나의 호움에 저장
특정 Expert만 필요해도 모든 블록 읽기 필요

하나의 방향, 전체 합치
선택적 실행 불가

▶️ MoE의 특징 (선택적/부모적 모델 실행)은 DAG 구조가 필요
3. 기술적 구조: Parameter DAG
하나의 Expert만을 포함한 block을 구성
관계가 보유되는 MetaBlock을 가지고, Expert block들이 각각 규칙처럼 다음과 같이 가지고 여부 관계을 선정:
{
  "index": 123,
  "type": "expert",
  "expert_name": "layer4.expert7",
  "depends_on": ["meta_hash_abc123"],
  "payload_hash": "...",
  "payload": "..." // Tensor
}

4. 구조 구성
구성 요소 / 역할
- MetaChain
Router 및 Expert 정의

- Parameter DAG
Expert block 매 개별 포함

- ParamIndex
Expert 이름 → Block hash/index 검색

- ModelManager
Router의 선택 기준에 따라 Expert 바로 로드

- Miner
Partial Mining (특정 Expert만 학습 및 업로드)

- Rewarder
Expert 품질 기반 보상 (QoS + usage 기반)

5. 작업 지칭
1) 바이딩: chain_id = "B-MOE" 로 새 채인 분기
2) Expert 단위 저장/불러오기 logic 구현 (upload_parameters 수정)
3) ModelManager.forward()에서 선택적 expert 조합 구성
4) reward_expert() 함수 구현 - 호출수, 정확도 기반 보상 로직
5) ParamIndex 관리 DB (복잡된 Layer 구조 포함)

6. 현실적 실행 로직 (컴퓨팅 파워 배분)
- 컴퓨팅 자원 배분 규칙:
Inference 요청이 들어오면 → 최우선
요청이 없는 경우 → Idle 자원은 학습에 배정 (background fine-tune)

- 러닝 우선순위 (러닝 라운더):
모든 Expert를 round-robin 방식으로 순차적으로 학습
또는 최근 호출/활용량이 많은 Expert 우선 (hot path 기반 학습)

- 체인 분기 조건:
특정 Expert block의 성능이 일정 임계치를 초과해 MetaBlock 규칙 위반 시 → 새로운 MetaBlock 체인 분기
예: 기존 MetaBlock에서 정의한 Router 규칙과 호환되지 않으면 새로운 “버전”으로 전환

- 점수 측정 및 채택 로직:
public validation dataset을 통해 baseline 대비 Δ 점수 평가
개선량이 δ 이상일 경우 block 채택

7. 최종 결론
DAG 구조가 MoE와 최고의 형식으로 최계화
Partial Inference 및 Partial Mining 가능
Expert 단위의 리소스 배포, 구매, 결정 가능
DAG가 모델 차이의 모델의 경우에도 개발자가 당장 투입 가능
블록체인은 더 이상 정적인 저장소가 아닌, 스스로 살아 움직이는 학습 시스템의 유전체처럼 진화함

## 8. Reference Implementation: AI-Block
DAG+MoE 구조를 실제 구현한 AI-Block 플랫폼은 다음과 같은 시스템 구성요소를 가집니다.

### 8.1 Dual Chain 설계
| 체인 ID | 역할 |
|---------|------|
| A       | Meta-chain (Router 규칙, 모델 구조 정의) |
| B       | Parameter-chain (Expert 단위 weight 블록) |

### 8.2 블록 헤더 필드 (공통)
| 필드 | 설명 |
|------|------|
| index | 블록 높이 |
| chain_id | A or B |
| points_to | 반대 체인 참조 블록 해시 |
| payload_hash | tensor 또는 메타코드의 SHA-256 |
| nonce | PoW를 위한 난수 |
| miner_pub | 서명 키 |
| payload_sig | payload에 대한 ECDSA 서명 |

### 8.3 추론 플로우
1. `/chat` API로 prompt 전송
2. `ModelManager`가 MetaBlock 기준 Router 로직 수립
3. ParamIndex에서 필요한 Expert weight block만 selective load
4. `state_dict` 생성 후 HuggingFace로 추론 수행

### 8.4 마이닝 플로우
1. miner가 fine-tuning → `state_dict` 생성
2. 블록 서명 및 업로드 (chunked or split 방식)
3. PoL, PoW, Signature 검증 후 채굴 성공 시 보상

## 9. Developer Onboarding Guide
### 9.1 기술 스택
- Language: Python 3.9+
- ML: PyTorch, HuggingFace
- Server: FastAPI + Uvicorn
- Chain: Custom Python modules (`core/`)
- Crypto: `ecdsa`, `hashlib`, secp256k1
- Storage: JSON → LevelDB (계획 중)

### 9.2 폴더 구조 예시
backend/
core/ # 블록체인, 보상, PoW, PoL
model/ # 모델 로딩/조합/인덱싱
api/ # 서버 API
miner/ # 커맨드라인 채굴 도구
frontend/ # 단순 웹 UI (채팅, 블록 확인)
scripts/

### 9.3 작업 체크리스트
- [x] Dual-chain core 구축
- [x] Parameter index → selective tensor load
- [x] Token ledger 및 지갑 기능
- [x] CLI 기반 채굴 도구 (`miner/`)
- [ ] Proof-of-Learning 자동화
- [ ] P2P + Block Gossip
- [ ] LevelDB 적용
- [ ] Docker/K8s 배포 스크립트

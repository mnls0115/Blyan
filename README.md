# AI-Block: 분산 AI 블록체인 플랫폼

AI 모델을 블록체인 기반으로 호스팅하고 진화시키는 혁신적인 시스템입니다. Mixture-of-Experts(MoE) 아키텍처와 DAG 구조를 활용하여 선택적 추론, 부분 마이닝, 분산 컴퓨팅을 가능하게 합니다.

## 🎯 주요 기능

### 1. AI 사용자 인터페이스
- **웹 채팅**: 브라우저에서 `frontend/index.html`을 열어 AI와 대화
- **API 호출**: REST API를 통해 프롬프트 입력 및 응답 받기
- **블록체인 정보**: 체인 상태 및 블록 정보 조회

### 2. AI 학습 및 블록 관리
- **전문가 블록 추가**: AI 모델의 개별 전문가를 블록체인에 업로드
- **모델 업데이트**: 기존 전문가의 성능을 개선하여 새 블록 생성
- **선택적 추론**: 필요한 전문가만 로드하여 효율적인 추론 수행

### 3. 프로젝트 정보
- **목적**: 투명하고 검증 가능한 AI 모델 호스팅
- **특징**: 분산화된 AI 진화 시스템
- **혁신**: 전통적인 중앙화된 AI 서비스의 대안 제공

## 🚀 빠른 시작

### 환경 설정
```bash
# Python 가상환경 생성
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 필요한 패키지 설치
pip install -r requirements.txt
```

### 초기 설정
```bash
# 메타체인 초기화 (최초 1회만)
python -c "
import json
from pathlib import Path
from backend.core.chain import Chain

root_dir = Path('./data')
meta_chain = Chain(root_dir, 'A')
spec = {
    'model_name': 'distilbert-base-uncased',
    'architecture': 'mixture-of-experts',
    'num_layers': 4,
    'num_experts': 8,
    'routing_strategy': 'top2'
}
meta_chain.add_block(json.dumps(spec).encode(), block_type='meta')
print('✅ 메타체인이 초기화되었습니다.')
"
```

### 웹 인터페이스 사용
1. **백엔드 서버 실행**:
   ```bash
   python -m api.server
   ```

2. **프론트엔드 열기**:
   - `frontend/index.html` 파일을 브라우저에서 열기
   - 채팅 인터페이스를 통해 AI와 대화

## 💡 사용 예시

### 기본 AI 대화
웹 인터페이스에서 프롬프트를 입력하면 AI가 응답합니다.

### API 사용 (선택사항)
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "안녕하세요", "use_moe": true}'
```

### 블록체인 정보 조회
```bash
curl -X GET "http://localhost:8000/chain/A/blocks"
```

## 🏗️ 시스템 아키텍처

- **메타체인 (A)**: AI 모델 아키텍처 및 라우팅 규칙 저장
- **파라미터체인 (B)**: 개별 전문가 가중치를 DAG 블록으로 저장
- **DAG 구조**: 병렬 전문가 진화를 가능하게 하는 의존성 그래프
- **선택적 로딩**: 필요한 전문가만 메모리에 로드하여 효율성 향상

## 📁 주요 파일

- `frontend/index.html` - 웹 사용자 인터페이스
- `api/server.py` - REST API 서버
- `backend/core/` - 블록체인 핵심 로직
- `backend/model/` - AI 모델 관리
- `miner/` - 블록 생성 도구

## 🎯 시스템 특징

- **🔄 자율 진화**: 전문가 레벨의 독립적 성능 향상
- **🤝 분산 협력**: P2P 전문가 공유 및 로드 밸런싱
- **📈 지속적 학습**: 실시간 성능 모니터링 및 적응적 라우팅
- **🧬 유기적 성장**: DAG 구조를 통한 병렬 전문가 개발
- **💰 경제적 인센티브**: 사용량 기반 자동 보상 분배

## 📚 추가 문서

프로젝트에 대한 더 자세한 정보는 다음 파일들을 참조하세요:
- `CLAUDE.md` - 개발자 가이드
- `TESTING_GUIDE.md` - 테스트 방법
- `POL_SYSTEM_GUIDE.md` - Proof-of-Learning 시스템

---

AI-Block은 투명하고 분산화된 AI의 미래를 위한 혁신적인 플랫폼입니다. 🤖✨
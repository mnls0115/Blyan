# 🚀 Blyan Network Quick Reference Guide

## 🤖 AI 개발자용 체크리스트

### 모델 작업할 때
```bash
# Expert 추가/수정
vim backend/model/moe_infer.py

# Router 로직 변경
vim backend/model/dynamic_router.py

# 새 아키텍처 추가
vim backend/model/arch.py
```

### 추론 최적화
```bash
# 캐싱 조정
vim backend/p2p/hot_expert_cache.py

# 배치 처리
vim backend/inference/batch_manager.py

# 분산 추론
vim backend/p2p/distributed_inference.py
```

### 학습 구현
```bash
# Micro-step 학습
vim backend/learning/micro_step_trainer.py

# Tile 기반 학습
vim backend/learning/tile_coordinator.py
```

### 데이터 품질
```bash
# L0 필터 조정
vim backend/data/l0_prefilter.py

# L1 AI 게이트
vim backend/data/l1_ai_quality_gate.py

# Hidden QA 추가
vim data/hidden_qa_v1.jsonl
vim backend/data/hidden_qa_loader.py
```

---

## ⛓️ 블록체인 개발자용 체크리스트

### 블록 작업
```bash
# 블록 구조 수정
vim backend/core/block.py

# 체인 로직
vim backend/core/chain.py

# DAG 검증
grep "has_cycle\|topological_sort" backend/core/chain.py
```

### 합의 메커니즘
```bash
# PoL 검증
vim backend/core/pol.py
vim backend/core/pol_validator.py

# 합의 API
vim backend/api/consensus.py
```

### 트랜잭션/원장
```bash
# 원장 스키마
vim migrations/001_create_ledger.sql

# 원장 로직
vim backend/accounting/ledger.py

# 트랜잭션 확인
docker-compose exec postgres psql -U blyan_user -d blyan_db -c "SELECT * FROM ledger.transactions;"
```

### 지갑/인증
```bash
# MetaMask 통합
vim frontend/wallet.js

# SIWE 인증
vim backend/api/siwe_auth.py

# 지갑 API
vim backend/api/wallet_auth.py
```

### 토큰 경제
```bash
# 토큰 파라미터
vim config/tokenomics.yaml

# 보상 엔진
vim backend/core/reward_engine_secure.py

# 경제 API
vim backend/api/economy.py
```

---

## 🔗 통합 작업 체크리스트

### Expert 업로드 (AI → Blockchain)
```bash
# 1. Expert 추출
python scripts/extract_individual_experts.py

# 2. 블록 생성 & 업로드
python miner/upload_moe_parameters.py \
    --address alice \
    --model-file ./models/tiny_mistral_moe \
    --meta-hash $(curl -s http://localhost:8000/chain/A/blocks | jq -r '.[0].hash')
```

### 블록체인 기반 추론 (Blockchain → AI)
```bash
# 1. 필요한 Expert 확인
curl http://localhost:8000/experts/list

# 2. 추론 실행
curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Hello", "use_moe": true}'

# 3. 사용량 확인
curl http://localhost:8000/experts/stats/layer0.expert0
```

### 분산 추론 설정
```bash
# 1. Expert 노드 시작
python -m backend.p2p.distributed_inference server node1 8001

# 2. 노드 등록
curl -X POST http://localhost:8000/p2p/register \
    -d '{"node_id": "node1", "experts": ["layer0.expert0"]}'

# 3. 분산 추론 테스트
curl -X POST http://localhost:8000/chat/distributed \
    -d '{"prompt": "test", "required_experts": ["layer0.expert0"]}'
```

### PoL 검증
```bash
# 1. 학습 전 성능 측정
curl http://localhost:8000/model/performance

# 2. 학습 실행 (미구현)
# python scripts/train_expert.py

# 3. PoL 증명 생성
curl -X POST http://localhost:8000/pol/submit \
    -d '{"improvement": 0.02, "dataset_hash": "..."}'
```

---

## 🛠️ 일반 운영 명령어

### 서비스 관리
```bash
# 시작
docker-compose up -d

# 상태 확인
docker-compose ps
./server.sh status

# 로그 보기
docker-compose logs -f api
tail -f logs/api.log

# 재시작
docker-compose restart api
./server.sh restart
```

### 헬스체크
```bash
# 간단 체크
curl http://localhost:8000/health

# 종합 체크
curl http://localhost:8000/health/comprehensive

# Teacher 모델
curl http://localhost:8000/health/teacher

# Redis
curl http://localhost:8000/health/redis

# PostgreSQL
curl http://localhost:8000/health/postgresql
```

### 테스트
```bash
# 프로덕션 테스트
python scripts/test_production.py

# L1 품질 게이트 테스트
python -m backend.data.l1_ai_quality_gate

# Hidden QA 테스트
python -m backend.data.hidden_qa_loader

# Teacher 모델 테스트
python scripts/prepare_teacher_model.py --verify-only
```

### 데이터베이스
```bash
# PostgreSQL 접속
docker-compose exec postgres psql -U blyan_user -d blyan_db

# 원장 잔액 확인
SELECT account_code, balance FROM ledger.accounts;

# 트랜잭션 확인
SELECT * FROM ledger.transactions ORDER BY created_at DESC LIMIT 10;

# Redis 접속
docker-compose exec redis redis-cli
AUTH ${REDIS_PASSWORD}
```

---

## 📍 주요 환경 변수

```bash
# .env.production 필수 설정

# 보안 (반드시 변경!)
REDIS_PASSWORD=복잡한비밀번호
DB_PASSWORD=다른복잡한비밀번호

# External APIs
OPENAI_API_KEY=sk-실제키
PERSPECTIVE_API_KEY=실제키
STRIPE_SECRET_KEY=sk_live_실제키

# Teacher 모델
BLYAN_TEACHER_CKPT=/models/teacher_v17-int8.safetensors

# 도메인
DOMAIN=blyan.network
SSL_EMAIL=admin@blyan.network
```

---

## 🔥 자주 발생하는 문제 해결

### "Teacher model not found"
```bash
python scripts/prepare_teacher_model.py
```

### "Redis connection refused"
```bash
docker-compose up -d redis
# 비밀번호 확인
grep REDIS_PASSWORD .env
```

### "Ledger schema not found"
```bash
docker-compose exec postgres psql -U blyan_user -d blyan_db < migrations/001_create_ledger.sql
```

### "No Expert blocks found"
```bash
# Genesis 블록 생성
python scripts/init_genesis.py

# Expert 업로드
python scripts/demo_upload.py
```

---

## 📚 추가 문서

- **아키텍처**: `PROJECT_ARCHITECTURE_REVIEW.md`
- **개발 로드맵**: `DEVELOPMENT_ROADMAP.md`
- **백서**: `moe_dag_whitepaper.md`
- **배포 가이드**: `DEPLOY_GUIDE.md`
- **프로젝트 지침**: `CLAUDE.md`

---

*빠른 시작: `./scripts/production_setup.sh` 실행*
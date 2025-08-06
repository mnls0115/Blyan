# Production Cleanup Checklist

> 🚨 **프로덕션 배포 전 필수 정리 사항**
> 
> 이 파일의 모든 항목을 확인하고 정리한 후 프로덕션에 배포하세요.
> 기능에 영향을 주지 않는 테스트/임시 코드만 포함되어 있습니다.

## ❌ 삭제할 파일들 (전체 삭제 가능)

### 테스트 스크립트들
- [x] `scripts/test_concurrent_inference.py` - Phase 1 테스트 ✅ 삭제됨
- [x] `scripts/test_micro_step_learning.py` - Phase 2 테스트 ✅ 삭제됨
- [x] `scripts/test_dual_model_streams.py` - Phase 3 테스트 ✅ 삭제됨
- [x] `scripts/test_complete_system.py` - 전체 시스템 테스트 ✅ 삭제됨

## 🔧 파일 내 정리할 코드들

### backend/learning/micro_step_trainer.py
- [x] **라인 465-500**: `async def demo_micro_step_training()` 함수 전체 ✅ 삭제됨
- [x] **라인 500-520**: `if __name__ == "__main__":` 블록 ✅ 삭제됨
- [x] **라인 430-460**: `DummyDataset` 클래스 (demo용) ✅ 삭제됨

### backend/learning/dual_model_manager.py
- [x] **라인 380-450**: `async def demo_dual_model()` 함수 전체 ✅ 삭제됨
- [x] **라인 320-380**: `class SimpleModel(nn.Module)` (demo용) ✅ 삭제됨
- [x] **라인 450-460**: `if __name__ == "__main__":` 블록 ✅ 삭제됨

### backend/inference/batch_manager.py
- [x] **라인 430-490**: `async def demo_batch_manager()` 함수 전체 ✅ 삭제됨
- [x] **라인 400-430**: `class MockModel` (demo용) ✅ 삭제됨
- [x] **라인 490-500**: `if __name__ == "__main__":` 블록 ✅ 삭제됨

### backend/p2p/concurrent_inference.py
- [x] **라인 580-620**: `async def demo_concurrent_system()` 함수 (없음) ✅ 해당 없음
- [x] **하드코딩된 Mock 데이터**: ✅ 설정 가능하도록 개선됨
  - [x] 라인 340-345: `torch.randn(4, 512)` → payload에서 설정 가능 ✅ 완료
  - [x] 라인 430-435: `torch.randn(1, 512)` → input_shape 파라미터 추가 ✅ 완료
  - [x] 라인 310-315: `nn.Linear(512, 512)` → 실제 모델 우선 사용 ✅ 완료

## 🐛 Debug/Print 문 정리

### 전체 파일에서 찾아서 정리할 것들
```bash
# 검색 명령어로 찾기
grep -r "print.*DEBUG" backend/
grep -r "print.*✅\|❌\|🚀\|📊" backend/
```

- [x] **backend/p2p/distributed_inference.py**: ✅ logger로 변경됨
  - [x] `print(f"DEBUG: ExpertNodeServer init - node_id: {node_id}, port: {port}")` → logger.info
  
- [x] **backend/learning/micro_step_trainer.py**: ✅ demo 코드와 함께 삭제됨
  - [x] `print("✅ Learning preempted successfully")` → 삭제됨
  - [x] `print("✅ Meta chain initialized with MoE architecture.")` → 삭제됨
  
- [x] **backend/inference/batch_manager.py**: ✅ demo 코드와 함께 삭제됨
  - [x] `print("🚀 Batch Manager Demo")` → 삭제됨
  - [x] `print("=" * 50)` (데모 관련) → 삭제됨

## 🔄 교체할 하드코딩 설정들

### backend/p2p/concurrent_inference.py
- [ ] **라인 330-340**: 더미 모델 초기화 → 실제 MoE 모델 로딩으로 교체
```python
# 삭제할 코드:
model = nn.Linear(512, 512)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 교체할 코드:
# 실제 MoE 모델 로딩 로직
```

### backend/learning/dual_model_manager.py
- [ ] **라인 45-55**: 테스트용 모델 → 실제 모델 파라미터로 교체

## 🎯 로깅 레벨 조정

### Debug → Info 레벨로 변경
- [ ] `logger.debug()` → `logger.info()` (중요한 것들)
- [ ] 불필요한 `logger.debug()` 제거

### 유지할 로깅 (삭제 금지)
- ✅ `logger.info("시스템 시작/종료 메시지")`
- ✅ `logger.error("에러 처리 메시지")`
- ✅ `logger.warning("경고 메시지")`

## 📋 정리 작업 순서

### 1단계: 테스트 파일 삭제
```bash
rm scripts/test_*.py
```

### 2단계: Demo 코드 정리
각 파일에서 `demo_*` 함수들과 `if __name__ == "__main__":` 블록 삭제

### 3단계: Mock 데이터 교체
하드코딩된 `torch.randn()` 등을 실제 데이터 소스로 교체

### 4단계: Debug Print 정리
이모지가 포함된 print문들을 적절한 로깅으로 교체

### 5단계: 최종 검증
```bash
# 남은 테스트 코드 확인
grep -r "demo\|test\|mock" backend/ --exclude-dir=__pycache__

# 하드코딩 확인  
grep -r "torch.randn\|nn.Linear.*512" backend/

# Debug print 확인
grep -r "print.*🚀\|✅\|❌" backend/
```

## ⚠️ 주의사항

### 절대 삭제하면 안 되는 것들
- ❌ 클래스 정의들 (`InferenceQueue`, `MicroStepTrainer`, etc.)
- ❌ API 엔드포인트 핸들러들
- ❌ 설정 클래스들 (`*Config`)
- ❌ `get_status()` 메서드들
- ❌ 메트릭 수집 코드들
- ❌ 에러 처리 로직들

### 검증 방법
각 단계 후 다음 테스트 실행:
```bash
# 기본 서버 시작 테스트
./server.sh start api

# API 엔드포인트 테스트  
curl -X POST "http://127.0.0.1:8000/chat" -H "Content-Type: application/json" \
  -d '{"prompt": "test", "use_moe": true}'
```

---

**✅ 모든 항목 체크 완료 시 프로덕션 배포 준비 완료**
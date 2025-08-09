# 견적-결제-사용 상태머신 문서

## 🔄 상태 플로우 다이어그램

```
[사용자 요청] 
    │
    ▼
[견적 생성] ──── quote_id 생성
    │             ├── TTL: 5분
    │             ├── Redis 저장
    │             └── 토큰수/가격 계산
    ▼
[견적 제공] ──── 클라이언트에 전달
    │             ├── 총 비용 표시
    │             ├── 무료티어 체크
    │             └── 만료시간 안내
    ▼
[사용자 승인] ─── 사용자가 "Proceed" 클릭
    │
    ▼
[서버 검증] ───── verify_chat_request_cost()
    │              ├── 견적 유효성 (TTL)
    │              ├── 토큰수 일치 (±10%)  
    │              ├── 무료티어 한도
    │              └── 잔액 확인 (유료시)
    ▼
[추론 실행] ───── MoE 추론 또는 분산 추론
    │              ├── 전문가 선택
    │              ├── 응답 생성
    │              └── 결과 반환
    ▼
[완료 처리] ───── complete_chat_request()
    │              ├── 무료티어 차감
    │              ├── 유료 결제 처리
    │              └── 사용량 추적
    ▼
[견적 무효화] ─── Redis에서 quote_id 삭제
```

## 📊 상태별 세부 정보

### 1. 견적 생성 단계
```python
# /economy/quote_chat 엔드포인트
quote_data = {
    "quote_id": f"quote_{secrets.token_hex(8)}_{int(time.time())}",
    "input_tokens": estimated_input_tokens,
    "output_tokens_est": output_tokens_est,
    "total_cost": calculated_cost,
    "created_at": time.time(),
    "expires_at": time.time() + 300,  # 5분 TTL
    "user_address": user_address
}
redis_client.setex(f"quote:{quote_id}", 300, json.dumps(quote_data))
```

### 2. 서버 검증 단계
```python
# verify_chat_request_cost() 핵심 로직
quote_data = quote_redis.get(quote_key)

# TTL 만료 체크
if time.time() > quote["expires_at"]:
    quote_redis.delete(quote_key)  # 즉시 삭제
    return {"allowed": False, "message": "Quote has expired"}

# 토큰수 검증 (10% 허용 오차)
if abs(actual_input_tokens - quote_input_tokens) > quote_input_tokens * 0.1:
    return {"allowed": False, "message": "Token count mismatch"}
```

### 3. 완료 처리 단계
```python
# 성공/실패 모두 처리
def complete_chat_request(user_address: str, success: bool = True):
    free_tier_manager.consume_request(user_address, success)
    # success=True: 정상 차감
    # success=False: 실패 기록만, 차감 없음
```

## 🚨 에지케이스 처리

### Case 1: 만료된 견적 사용 시도
```
상황: 사용자가 5분 후 견적으로 요청
처리: 
  - 서버에서 TTL 체크
  - Redis에서 즉시 삭제  
  - "Quote has expired" 오류 반환
  - 사용자에게 새 견적 요청 안내
```

### Case 2: 토큰수 불일치
```  
상황: 사용자가 프롬프트 수정 후 견적 재사용
처리:
  - 실제 토큰수 vs 견적 토큰수 비교
  - 10% 이상 차이시 거부
  - "Token count mismatch" 오류
  - 정확한 토큰수 함께 반환
```

### Case 3: 무료티어 한도 초과
```
상황: 견적 생성 후 다른 요청들로 한도 소진
처리:
  - 서버 검증 시점에 실시간 한도 확인
  - Redis SSOT에서 현재 상태 조회  
  - "Free tier limit exceeded" 안내
  - 업그레이드 옵션 제공
```

### Case 4: 잔액 부족 (유료 사용자)
```
상황: 견적 생성 후 다른 거래로 잔액 감소
처리:
  - 서버 검증 시점에 잔액 재확인
  - 부족시 "Insufficient balance" 오류
  - 필요 금액과 현재 잔액 표시
```

## 🔒 보안 강화 요소

### 클라이언트 우회 방지
- **서버 권위**: 모든 검증을 서버에서 수행
- **실시간 검증**: 견적 생성 시점이 아닌 사용 시점 검증
- **토큰수 재계산**: 서버에서 tiktoken으로 정확한 계산

### 남용 방지
- **견적 재사용 방지**: 사용 후 즉시 무효화
- **시간 제한**: 5분 TTL로 견적 캐싱 공격 방지  
- **사용자별 제한**: Redis SSOT로 동시 요청 제어

### 데이터 일관성
- **원자적 처리**: 성공시에만 차감, 실패시 원복
- **이중 지출 방지**: quote_id 단일 사용 보장
- **감사 추적**: 모든 거래 PostgreSQL 기록

## ✅ 상태머신 검증 체크리스트

- [x] 견적 생성 시 TTL 설정 (5분)
- [x] 서버 검증에서 TTL 만료 체크  
- [x] 만료된 견적 자동 삭제
- [x] 토큰수 불일치 감지 (±10%)
- [x] 무료티어 실시간 한도 확인
- [x] 유료 사용자 잔액 실시간 확인
- [x] 성공/실패 구분 처리
- [x] 클라이언트 우회 불가능한 서버 권위
- [x] 견적 재사용 방지 (단일 사용)
- [x] 전체 플로우 예외 처리

## 📈 성능 최적화

### Redis 최적화
- **TTL 자동 만료**: 수동 삭제 불필요
- **키 네이밍**: `quote:{quote_id}` 패턴
- **직렬화**: JSON 사용 (orjson 고려)

### 데이터베이스 최적화  
- **비동기 로깅**: 응답 속도 영향 없음
- **배치 처리**: 소액 거래 배치 집계
- **인덱싱**: user_address, timestamp 인덱스

이 상태머신은 **보안**, **정확성**, **성능**을 모두 보장하며 운영 안정성을 위한 모든 에지케이스를 처리합니다.
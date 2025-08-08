# 🚀 Digital Ocean 배포 가이드

## 준비사항

### 1. Digital Ocean Droplet 생성
- **Size**: 최소 2GB RAM, 2 vCPU ($18/month)
- **추천**: 4GB RAM, 2 vCPU ($36/month)
- **Region**: Singapore (아시아) 또는 San Francisco (미국)
- **OS**: Ubuntu 22.04 LTS

### 2. 도메인 설정
- Digital Ocean DNS 또는 Cloudflare 사용
- A 레코드: `@` → Droplet IP
- A 레코드: `www` → Droplet IP

### 3. 필수 계정
- Stripe 계정 (결제 처리)
- Redis Cloud (선택사항, 자체 호스팅 가능)

## 📦 배포 단계

### Step 1: Droplet 접속
```bash
ssh root@your_droplet_ip
```

### Step 2: 파일 업로드
```bash
# 로컬에서
scp -r /Users/mnls/projects/aiblock root@your_droplet_ip:/root/blyan

# 또는 GitHub에서
git clone https://github.com/yourusername/blyan.git /opt/blyan
```

### Step 3: 배포 스크립트 실행
```bash
cd /root/blyan
chmod +x deploy_digitalocean.sh
./deploy_digitalocean.sh
```

### Step 4: 환경변수 설정
```bash
nano /opt/blyan/.env
```

필수 설정:
```env
# 꼭 변경해야 할 값들
REDIS_PASSWORD=매우복잡한비밀번호123!@#
DB_PASSWORD=다른복잡한비밀번호456$%^

# Stripe (dashboard.stripe.com에서 복사)
STRIPE_SECRET_KEY=sk_live_실제키
STRIPE_WEBHOOK_SECRET=whsec_실제키
STRIPE_PUBLISHABLE_KEY=pk_live_실제키

# 도메인
DOMAIN=your-domain.com
SSL_EMAIL=your-email@gmail.com
```

### Step 5: 서비스 시작
```bash
cd /opt/blyan
docker-compose up -d
```

### Step 6: SSL 인증서 설정
```bash
certbot --nginx -d your-domain.com -d www.your-domain.com
```

## 🔍 상태 확인

### 서비스 상태
```bash
# Docker 컨테이너 확인
docker-compose ps

# API 로그 보기
docker-compose logs -f api

# 전체 시스템 상태
systemctl status blyan
```

### 헬스체크
```bash
# API 헬스체크
curl https://your-domain.com/health

# Redis 연결 테스트
docker-compose exec redis redis-cli ping

# PostgreSQL 연결 테스트
docker-compose exec postgres psql -U blyan_user -d blyan_db -c "SELECT 1"
```

## 💰 비용 예상

| 항목 | 월 비용 | 설명 |
|-----|--------|------|
| Droplet (2GB) | $18 | 최소 사양 |
| Droplet (4GB) | $36 | 추천 사양 |
| 백업 | $3.60 | 20% of droplet |
| 도메인 | $1 | .com 도메인 |
| **총계** | **$40-50** | 월 예상 비용 |

## 🛠️ 문제 해결

### Redis 연결 실패
```bash
# Redis 비밀번호 확인
docker-compose exec redis redis-cli
> AUTH your_password
> PING
```

### Stripe Webhook 실패
1. Stripe Dashboard → Webhooks
2. Endpoint URL: `https://your-domain.com/payment/webhook`
3. Events: `payment_intent.succeeded`, `charge.refunded`

### 메모리 부족
```bash
# Swap 추가 (4GB)
fallocate -l 4G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
```

## 📊 모니터링

### Grafana 접속
- URL: `https://your-domain.com:3000`
- 초기 로그인: admin/admin
- 비밀번호 즉시 변경!

### Prometheus 메트릭
- URL: `https://your-domain.com:9090`
- API 메트릭: `/metrics`

### 로그 확인
```bash
# 실시간 로그
tail -f /opt/blyan/logs/api.log

# Docker 로그
docker-compose logs --tail=100 -f api
```

## 🔒 보안 체크리스트

- [ ] 모든 기본 비밀번호 변경
- [ ] SSH 키 인증만 허용
- [ ] Fail2ban 설정 확인
- [ ] 방화벽 규칙 확인 (`ufw status`)
- [ ] 정기 백업 설정
- [ ] 모니터링 알림 설정

## 🔄 업데이트 방법

```bash
cd /opt/blyan

# 백업 먼저!
./backup.sh

# 코드 업데이트
git pull  # 또는 새 파일 업로드

# 서비스 재시작
docker-compose down
docker-compose up -d --build
```

## 📞 지원

문제 발생시:
1. 로그 확인: `docker-compose logs api`
2. 시스템 리소스 확인: `htop`
3. 디스크 공간 확인: `df -h`
4. 네트워크 확인: `netstat -tlnp`

---

**🎉 축하합니다! Blyan Network가 배포되었습니다!**

접속 URL: `https://your-domain.com`
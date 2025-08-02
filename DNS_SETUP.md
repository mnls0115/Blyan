# blyan.com DNS 설정 가이드

## 🌐 DNS 설정

PorkBun 도메인 관리 페이지에서 다음 DNS 레코드를 설정하세요:

### A 레코드 설정
```
레코드 타입: A
이름: @ (또는 blyan.com)
값: [서버 IP 주소]
TTL: 300 (5분)

레코드 타입: A  
이름: www
값: [서버 IP 주소]
TTL: 300 (5분)
```

### 선택사항: CNAME 레코드
```
레코드 타입: CNAME
이름: www
값: blyan.com
TTL: 300 (5분)
```

## 🚀 배포 방법

1. **서버에 코드 업로드**
   ```bash
   git clone https://github.com/your-repo/aiblock.git
   cd aiblock
   ```

2. **DNS 전파 확인** (변경 후 24시간 기다림)
   ```bash
   nslookup blyan.com
   ping blyan.com
   ```

3. **배포 실행**
   ```bash
   chmod +x deploy_blyan.sh
   ./deploy_blyan.sh
   ```

## 🔍 확인 사항

### DNS 전파 확인
```bash
# DNS lookup
nslookup blyan.com
nslookup www.blyan.com

# Ping 테스트
ping blyan.com
ping www.blyan.com

# 온라인 DNS 체크
https://www.whatsmydns.net/#A/blyan.com
```

### 배포 후 확인
```bash
# SSL 인증서 확인
curl -I https://blyan.com

# API 상태 확인
curl https://blyan.com/api/pol/status

# 웹사이트 접속
open https://blyan.com
```

## 🛠️ 트러블슈팅

### DNS 전파가 안 될 때
- 24-48시간 기다림
- TTL 값을 300 (5분)으로 설정
- DNS 캐시 플러시: `sudo dscacheutil -flushcache`

### SSL 인증서 오류
```bash
# Let's Encrypt 수동 갱신
sudo certbot --nginx -d blyan.com -d www.blyan.com

# 인증서 상태 확인
sudo certbot certificates
```

### 서비스 재시작
```bash
# Nginx 재시작
sudo systemctl restart nginx

# AI-Block API 재시작
sudo systemctl restart aiblock

# 로그 확인
sudo tail -f /var/log/nginx/error.log
```

## 📊 최종 확인 URL

- 🌍 메인 웹사이트: https://blyan.com
- 📊 API 상태: https://blyan.com/api/pol/status
- 💬 AI 채팅: https://blyan.com
- 🔍 블록체인 탐색기: https://blyan.com/explorer.html
- 🧮 PoL 검증기: https://blyan.com/pol_validator.html
- 🛡️ 보안 상태: https://blyan.com/api/security/network_health
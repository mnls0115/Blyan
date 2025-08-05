#!/bin/bash

# Blyan 웹사이트 배포 스크립트
# 사용법: ./deploy.sh your-domain.com

DOMAIN=$1
if [ -z "$DOMAIN" ]; then
    echo "❌ 도메인을 입력해주세요: ./deploy.sh your-domain.com"
    exit 1
fi

echo "🚀 Blyan 웹사이트 배포 시작..."
echo "📡 도메인: $DOMAIN"

# 1. 시스템 패키지 업데이트
echo "📦 시스템 업데이트..."
sudo apt update && sudo apt upgrade -y

# 2. 필요한 패키지 설치
echo "🔧 필요한 패키지 설치..."
sudo apt install -y nginx python3 python3-pip python3-venv certbot python3-certbot-nginx

# 3. Python 환경 설정
echo "🐍 Python 환경 설정..."
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 4. 웹 디렉토리 생성 및 파일 복사
echo "📁 웹 디렉토리 설정..."
sudo mkdir -p /var/www/aiblock
sudo cp -r frontend/* /var/www/aiblock/frontend/
sudo chown -R www-data:www-data /var/www/aiblock

# 5. Nginx 설정
echo "⚙️ Nginx 설정..."
sudo cp nginx.conf /etc/nginx/sites-available/aiblock
sudo sed -i "s/your-domain.com/$DOMAIN/g" /etc/nginx/sites-available/aiblock
sudo ln -sf /etc/nginx/sites-available/aiblock /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# 6. Nginx 테스트 및 재시작
echo "🔄 Nginx 재시작..."
sudo nginx -t
sudo systemctl restart nginx
sudo systemctl enable nginx

# 7. SSL 인증서 발급 (Let's Encrypt)
echo "🔒 SSL 인증서 발급..."
sudo certbot --nginx -d $DOMAIN -d www.$DOMAIN --non-interactive --agree-tos --email admin@$DOMAIN

# 8. 방화벽 설정
echo "🛡️ 방화벽 설정..."
sudo ufw allow 'Nginx Full'
sudo ufw allow 22
sudo ufw --force enable

# 9. 백엔드 서비스 시작
echo "🖥️ 백엔드 서비스 시작..."
./server.sh start api

# 10. 시스템 서비스 등록 (선택사항)
echo "🔄 시스템 서비스 등록..."
sudo tee /etc/systemd/system/aiblock.service > /dev/null <<EOF
[Unit]
Description=Blyan API Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/.venv/bin
ExecStart=$(pwd)/.venv/bin/uvicorn api.server:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable aiblock
sudo systemctl start aiblock

echo "✅ 배포 완료!"
echo "🌍 웹사이트: https://$DOMAIN"
echo "📊 API 상태: https://$DOMAIN/api/pol/status"
echo ""
echo "📋 다음 단계:"
echo "1. DNS A 레코드를 서버 IP로 설정"
echo "2. 24시간 후 SSL 자동 갱신 테스트: sudo certbot renew --dry-run"
echo "3. 로그 확인: sudo tail -f /var/log/nginx/access.log"
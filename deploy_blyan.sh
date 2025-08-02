#!/bin/bash

echo "🚀 blyan.com 도메인 배포 시작..."

# blyan.com으로 deploy.sh 실행
./deploy.sh blyan.com

echo "✅ blyan.com 배포 완료!"
echo ""
echo "🌍 웹사이트: https://blyan.com"
echo "📊 API: https://blyan.com/api/pol/status" 
echo "💬 채팅: https://blyan.com"
echo ""
echo "📋 DNS 설정이 필요합니다:"
echo "   A 레코드: blyan.com → [서버 IP]"
echo "   A 레코드: www.blyan.com → [서버 IP]"
echo ""
echo "🔍 DNS 설정 확인:"
echo "   nslookup blyan.com"
echo "   ping blyan.com"
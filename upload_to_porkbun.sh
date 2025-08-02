#!/bin/bash

echo "📁 PorkBun Static Hosting 업로드 준비..."

# 업로드용 디렉토리 생성
mkdir -p porkbun_upload

# 프론트엔드 파일들 복사
cp frontend/index.html porkbun_upload/
cp frontend/main.js porkbun_upload/
cp frontend/common-header.js porkbun_upload/
cp frontend/explorer.html porkbun_upload/
cp frontend/explorer.js porkbun_upload/
cp frontend/pol_validator.html porkbun_upload/
cp frontend/pol_validator.js porkbun_upload/

# API 없이도 작동하도록 config.js 수정
cat > porkbun_upload/config.js << 'EOF'
// API Configuration for Static Hosting
const API_CONFIG = {
    // Demo mode - API 서버 없이 프론트엔드만 작동
    baseURL: '', // 일단 비워둠
    
    // Individual endpoints
    chat: '/chat',
    balance: '/balance/',
    chain: '/chain/B/blocks?limit=10',
    polStatus: '/pol/status',
    
    // Demo mode flag
    demoMode: true
};

// Export for use in other files
window.API_CONFIG = API_CONFIG;
EOF

echo "✅ 업로드 파일 준비 완료!"
echo ""
echo "📋 FTP 업로드 방법:"
echo "1. FileZilla 또는 FTP 클라이언트 실행"
echo "2. 호스트: pixie-ftp.porkbun.com"
echo "3. 사용자: blyan.com"
echo "4. 패스워드: [PorkBun에서 확인]"
echo "5. porkbun_upload/ 폴더의 모든 파일을 업로드"
echo ""
echo "🌐 업로드 후 https://blyan.com 에서 확인!"
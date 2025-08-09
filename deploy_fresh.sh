#!/bin/bash
# 🚀 Blyan Network 새로운 배포 스크립트

echo "🚀 Blyan Network 새로운 배포를 시작합니다..."

# Step 1: 기존 코드 완전 삭제
echo "📂 기존 코드 삭제 중..."
cd /root
sudo fuser -k 8000/tcp 2>/dev/null || echo "포트 8000에 실행 중인 프로세스 없음"
rm -rf dnai/
echo "✅ 기존 코드 삭제 완료"

# Step 2: 새 코드 클론
echo "📥 최신 코드 다운로드 중..."
git clone https://github.com/mnls0115/Blyan.git dnai
cd dnai
echo "✅ 코드 다운로드 완료"

# Step 3: 환경 설정
echo "🔧 환경 설정 중..."
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 성능 최적화 패키지 (Linux에서만)
pip install accelerate bitsandbytes --quiet || echo "양자화 패키지 설치 실패 (GPU 없음)"
echo "✅ 환경 설정 완료"

# Step 4: 디렉토리 생성
echo "📁 필요한 디렉토리 생성 중..."
mkdir -p data logs
echo "✅ 디렉토리 생성 완료"

# Step 5: 블록체인 초기화
echo "⛓️ 블록체인 초기화 중..."
source .venv/bin/activate
python -c "
import json
from pathlib import Path
from backend.core.chain import Chain

root_dir = Path('./data')
meta_chain = Chain(root_dir, 'A')
spec = {
    'model_name': 'gpt_oss_20b',
    'architecture': 'mixture-of-experts', 
    'num_layers': 24,
    'num_experts': 16,
    'routing_strategy': 'top2'
}
meta_chain.add_block(json.dumps(spec).encode(), block_type='meta')
print('✅ 블록체인 초기화 완료')
"

# Step 6: 서버 시작
echo "🚀 서버 시작 중..."
chmod +x manage.sh 2>/dev/null || echo "manage.sh 없음"

if [ -f "manage.sh" ]; then
    ./manage.sh start
else
    # 직접 서버 시작
    source .venv/bin/activate
    nohup python3 -m uvicorn api.server:app --host 0.0.0.0 --port 8000 > logs/api.log 2>&1 &
fi

# Step 7: 배포 확인
echo "✅ 배포 완료! 서버 상태 확인 중..."
sleep 5

# 서버 응답 확인
if curl -f http://localhost:8000/ > /dev/null 2>&1; then
    echo "🎉 배포 성공! 서버가 정상 작동 중입니다."
    echo ""
    echo "📍 접속 정보:"
    echo "   - API: http://your-server-ip:8000"
    echo "   - Frontend: http://your-server-ip:8000/frontend/index.html"
    echo ""
    echo "🔧 관리 명령어:"
    echo "   - 서버 상태: curl http://localhost:8000/"
    echo "   - 로그 확인: tail -f /root/dnai/logs/api.log"
    echo "   - 서버 중지: sudo fuser -k 8000/tcp"
    echo ""
else
    echo "❌ 서버 시작 실패. 로그를 확인하세요:"
    echo "   tail -f /root/dnai/logs/api.log"
fi

echo "🚀 Blyan Network 배포 스크립트 완료!"
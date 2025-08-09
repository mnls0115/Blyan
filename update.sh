#!/bin/bash
# 블록체인 데이터 보존하면서 업데이트하는 간단한 스크립트

echo "🔄 Blyan 업데이트 (데이터 보존)"

# 서버 중지
sudo fuser -k 8000/tcp 2>/dev/null || true

# 데이터 백업
if [ -d "data" ]; then
    cp -r data ../data_backup_$(date +%m%d_%H%M)
    echo "✅ 블록체인 데이터 백업됨"
fi

# 코드 업데이트
git pull

# 데이터 복원 (혹시 덮어씌워졌다면)
if [ -d "../data_backup_$(date +%m%d_%H%M)" ]; then
    cp -r ../data_backup_$(date +%m%d_%H%M)/* data/ 2>/dev/null || true
fi

# Python 업데이트
source .venv/bin/activate || (python3 -m venv .venv && source .venv/bin/activate)
pip install -r requirements.txt --quiet

# 서버 재시작
mkdir -p logs
nohup python3 -m uvicorn api.server:app --host 0.0.0.0 --port 8000 > logs/api.log 2>&1 &

echo "🎉 업데이트 완료! 로그: tail -f logs/api.log"
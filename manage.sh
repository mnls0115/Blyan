#!/bin/bash
# Blyan Network 통합 관리 스크립트

case "$1" in
    # ====== 서버 관리 ======
    "start"|"stop"|"restart"|"status"|"logs")
        echo "🚀 서버 관리"
        ./server.sh "$@"
        ;;
    
    # ====== 업데이트 ======
    "update")
        echo "🔄 시스템 업데이트"
        ./update.sh
        ;;
    
    # ====== 테스트 ======
    "test")
        echo "🧪 테스트 실행"
        if [ ! -f ".venv/bin/activate" ]; then
            python3 -m venv .venv
        fi
        source .venv/bin/activate
        
        case "$2" in
            "full")
                python scripts/demo_full_moe_flow.py
                ;;
            "inference")
                python scripts/test_inference_only.py
                ;;
            "gpu")
                python test_gpu_node.py
                ;;
            *)
                echo "테스트 옵션: full, inference, gpu"
                ;;
        esac
        ;;
    
    # ====== MoE 업로드 ======
    "upload")
        echo "📦 MoE 모델 업로드"
        META_HASH=$(curl -s http://127.0.0.1:8000/chain/A/blocks | grep -o '"hash":"[^"]*"' | head -1 | cut -d'"' -f4)
        python miner/upload_moe_parameters.py \
            --address alice \
            --model-file ./models/Qwen/Qwen3-8B-FP8 \
            --meta-hash $META_HASH \
            --candidate-loss 0.8 \
            --skip-pow
        ;;
    
    # ====== 배포 (처음 한 번만) ======
    "deploy")
        echo "🚀 간단한 배포"
        if [ "$2" = "init" ]; then
            # 최소한의 설정만
            sudo apt update
            sudo apt install -y python3 python3-pip python3-venv nginx
            python3 -m venv .venv
            source .venv/bin/activate
            pip install -r requirements.txt
            
            echo "✅ 설정 완료!"
            echo "이제 ./manage.sh start 로 서버 시작하세요."
        else
            echo "사용법: $0 deploy init"
        fi
        ;;
    
    # ====== 도움말 ======
    *)
        echo "🤖 Blyan Network 관리 도구"
        echo "========================="
        echo ""
        echo "서버 관리:"
        echo "  $0 start          # 모든 서버 시작"
        echo "  $0 stop           # 모든 서버 중지"
        echo "  $0 restart        # 모든 서버 재시작"
        echo "  $0 status         # 서버 상태 확인"
        echo "  $0 logs [server]  # 로그 확인"
        echo ""
        echo "업데이트:"
        echo "  $0 update         # 코드 업데이트 (데이터 보존)"
        echo ""
        echo "테스트:"
        echo "  $0 test full      # 전체 테스트"
        echo "  $0 test inference # 추론 테스트"
        echo "  $0 test gpu       # GPU 노드 테스트"
        echo ""
        echo "개발:"
        echo "  $0 upload         # MoE 모델 업로드"
        echo ""
        echo "배포:"
        echo "  $0 deploy digitalocean  # 처음 배포시만"
        ;;
esac
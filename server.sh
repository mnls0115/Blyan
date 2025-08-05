#!/bin/bash

# Blyan Multi-Server Management Script

# 가상환경 활성화 (있는 경우)
if [ -d "myenv" ]; then
    source myenv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# 서버 구성 (포트:모듈 형식)
SERVERS="api:8000:api.server:app p2p-node1:8001:backend.p2p.distributed_inference p2p-node2:8002:backend.p2p.distributed_inference"

get_server_config() {
    local name=$1
    for server in $SERVERS; do
        if [[ $server == $name:* ]]; then
            echo $server
            return
        fi
    done
}

start_server() {
    local name=$1
    local config=$(get_server_config $name)
    local port=$(echo $config | cut -d: -f2)
    local module=$(echo $config | cut -d: -f3-)
    
    echo "🚀 Starting $name server on port $port..."
    
    if [ "$name" = "api" ]; then
        # 가상환경 경로 직접 사용
        if [ -f "myenv/bin/python3" ]; then
            myenv/bin/python3 -m uvicorn $module --reload --host 0.0.0.0 --port $port > logs/${name}.log 2>&1 &
        else
            python3 -m uvicorn $module --reload --host 0.0.0.0 --port $port > logs/${name}.log 2>&1 &
        fi
    else
        echo "Debug: Starting P2P node with command: python3 -m $module server $name $port"
        if [ -f "myenv/bin/python3" ]; then
            myenv/bin/python3 -m $module server $name $port > logs/${name}.log 2>&1 &
        else
            python3 -m $module server $name $port > logs/${name}.log 2>&1 &
        fi
    fi
    
    local pid=$!
    echo "   ✓ $name started (PID: $pid) - http://127.0.0.1:$port"
    sleep 1
}

stop_server() {
    local name=$1
    local config=$(get_server_config $name)
    local port=$(echo $config | cut -d: -f2)
    
    echo "🛑 Stopping $name server..."
    
    # 포트로 프로세스 찾아서 종료
    local pids=$(lsof -ti:$port 2>/dev/null)
    if [ ! -z "$pids" ]; then
        for pid in $pids; do
            kill $pid 2>/dev/null
        done
        sleep 2
        # 강제 종료로 확인
        local remaining=$(lsof -ti:$port 2>/dev/null)
        if [ ! -z "$remaining" ]; then
            for pid in $remaining; do
                kill -9 $pid 2>/dev/null
            done
        fi
        echo "   ✓ $name stopped (PIDs: $pids)"
    else
        echo "   ⚠️  $name was not running"
    fi
}

status_server() {
    local name=$1
    local config=$(get_server_config $name)
    local port=$(echo $config | cut -d: -f2)
    
    local pids=$(lsof -ti:$port 2>/dev/null)
    if [ ! -z "$pids" ]; then
        echo "✅ $name: Running (PIDs: $(echo $pids | tr '\n' ' '), Port: $port)"
        # API 응답 확인
        if [ "$name" = "api" ]; then
            curl -s http://127.0.0.1:$port/pol/status > /dev/null 2>&1 && echo "   🌐 API responding" || echo "   ❌ API not responding"
        fi
    else
        echo "❌ $name: Stopped (Port: $port)"
    fi
}

case "$1" in
    start)
        mkdir -p logs
        if [ -z "$2" ]; then
            echo "🔄 Starting all servers..."
            
            # 먼저 모든 서버 중지
            echo "Cleaning up existing processes..."
            for server in $SERVERS; do
                name=$(echo $server | cut -d: -f1)
                stop_server $name >/dev/null 2>&1
            done
            
            sleep 3
            
            # 순차적으로 시작 및 확인
            for server in $SERVERS; do
                name=$(echo $server | cut -d: -f1)
                port=$(echo $server | cut -d: -f2)
                
                start_server $name
                sleep 2
                
                # 시작 확인
                attempts=0
                while [ $attempts -lt 5 ]; do
                    if lsof -ti:$port >/dev/null 2>&1; then
                        echo "   ✓ $name confirmed running on port $port"
                        break
                    fi
                    sleep 1
                    attempts=$((attempts + 1))
                done
                
                if [ $attempts -eq 5 ]; then
                    echo "   ⚠️  Warning: $name may not have started properly"
                fi
            done
        else
            start_server $2
        fi
        ;;
    stop)
        if [ -z "$2" ]; then
            echo "🔄 Stopping all servers..."
            for server in $SERVERS; do
                name=$(echo $server | cut -d: -f1)
                stop_server $name
            done
        else
            stop_server $2
        fi
        ;;
    restart)
        if [ -z "$2" ]; then
            echo "🔄 Restarting all servers..."
            for server in $SERVERS; do
                name=$(echo $server | cut -d: -f1)
                stop_server $name
            done
            sleep 2
            for server in $SERVERS; do
                name=$(echo $server | cut -d: -f1)
                start_server $name
            done
        else
            stop_server $2
            sleep 1
            start_server $2
        fi
        ;;
    status)
        echo "📊 Server Status:"
        echo "=================="
        for server in $SERVERS; do
            name=$(echo $server | cut -d: -f1)
            status_server $name
        done
        ;;
    logs)
        if [ -z "$2" ]; then
            echo "📋 Available logs:"
            for server in $SERVERS; do
                name=$(echo $server | cut -d: -f1)
                echo "  - $name (logs/${name}.log)"
            done
        else
            echo "📋 Showing logs for $2..."
            tail -f logs/$2.log
        fi
        ;;
    list)
        echo "🔧 Available servers:"
        for server in $SERVERS; do
            name=$(echo $server | cut -d: -f1)
            port=$(echo $server | cut -d: -f2)
            echo "  - $name (port $port)"
        done
        ;;
    *)
        echo "🤖 Blyan Multi-Server Manager"
        echo "==============================="
        echo "Usage: $0 {start|stop|restart|status|logs|list} [server-name]"
        echo ""
        echo "Commands:"
        echo "  start [server]   - Start all servers or specific server"
        echo "  stop [server]    - Stop all servers or specific server"
        echo "  restart [server] - Restart all servers or specific server"
        echo "  status           - Show status of all servers"
        echo "  logs [server]    - Show logs (specify server name)"
        echo "  list             - List available servers"
        echo ""
        echo "Examples:"
        echo "  $0 start         # Start all servers"
        echo "  $0 start api     # Start only API server"
        echo "  $0 logs api      # Show API server logs"
        echo "  $0 stop p2p-node1 # Stop specific P2P node"
        exit 1
        ;;
esac
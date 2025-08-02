#!/usr/bin/env python3
"""
Simple HTTP server for serving AI-Block frontend
사용법: python3 simple-server.py [port]
"""

import http.server
import socketserver
import os
import sys
from pathlib import Path

# 포트 설정
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 3000

# frontend 디렉토리로 이동
frontend_dir = Path(__file__).parent / "frontend"
os.chdir(frontend_dir)

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # CORS 헤더 추가
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

# 서버 시작
with socketserver.TCPServer(("0.0.0.0", PORT), CustomHTTPRequestHandler) as httpd:
    print(f"🌍 AI-Block Frontend 서버 시작")
    print(f"📡 http://localhost:{PORT}")
    print(f"🔗 http://0.0.0.0:{PORT} (외부 접근 가능)")
    print("⏹️  Ctrl+C로 종료")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 서버 종료")
        sys.exit(0)
#!/usr/bin/env python3
"""
SSL 인증서 자동 갱신 및 관리 스크립트
SSL certificate auto-renewal and management script
"""

import os
import subprocess
import json
from datetime import datetime, timedelta
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SSLManager:
    def __init__(self, domain="blyan.com", email="admin@blyan.com"):
        self.domain = domain
        self.email = email
        self.cert_dir = Path("/etc/letsencrypt/live") / domain
        self.config_dir = Path("./config")
        self.config_dir.mkdir(exist_ok=True)
        self.log_file = Path("./logs/ssl_renewal.json")
        self.log_file.parent.mkdir(exist_ok=True)
    
    def check_certificate_status(self):
        """인증서 상태 확인"""
        try:
            if not self.cert_dir.exists():
                return {
                    "exists": False,
                    "message": "인증서가 존재하지 않음"
                }
            
            cert_file = self.cert_dir / "cert.pem"
            if not cert_file.exists():
                return {
                    "exists": False,
                    "message": "인증서 파일이 존재하지 않음"
                }
            
            # OpenSSL로 인증서 만료일 확인
            result = subprocess.run([
                "openssl", "x509", "-in", str(cert_file), "-noout", "-dates"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                return {
                    "exists": False,
                    "message": "인증서 읽기 실패"
                }
            
            # 만료일 파싱
            lines = result.stdout.split('\n')
            not_after = None
            for line in lines:
                if line.startswith("notAfter="):
                    not_after = line.split("=", 1)[1]
                    break
            
            if not not_after:
                return {
                    "exists": True,
                    "message": "만료일을 찾을 수 없음"
                }
            
            # 만료일까지 남은 일수 계산
            from datetime import datetime
            expires = datetime.strptime(not_after, "%b %d %H:%M:%S %Y %Z")
            days_left = (expires - datetime.now()).days
            
            return {
                "exists": True,
                "expires": expires.isoformat(),
                "days_left": days_left,
                "needs_renewal": days_left < 30,
                "message": f"인증서는 {days_left}일 후 만료됩니다"
            }
            
        except Exception as e:
            return {
                "exists": False,
                "error": str(e),
                "message": f"인증서 확인 중 오류: {e}"
            }
    
    def install_certbot(self):
        """Certbot 설치"""
        try:
            # certbot이 이미 설치되어 있는지 확인
            result = subprocess.run(["which", "certbot"], capture_output=True)
            if result.returncode == 0:
                logger.info("✅ Certbot이 이미 설치되어 있습니다")
                return True
            
            # macOS에서 brew로 설치
            if subprocess.run(["which", "brew"], capture_output=True).returncode == 0:
                subprocess.run(["brew", "install", "certbot"], check=True)
                logger.info("✅ Certbot 설치 완료 (Homebrew)")
                return True
            else:
                logger.error("❌ Homebrew가 설치되어 있지 않습니다")
                logger.info("💡 수동 설치: brew install certbot")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Certbot 설치 실패: {e}")
            return False
    
    def obtain_certificate(self):
        """새 인증서 발급"""
        try:
            # 웹서버 임시 중지 (필요시)
            self.stop_webserver()
            
            # Certbot으로 인증서 발급
            cmd = [
                "sudo", "certbot", "certonly",
                "--standalone",
                "--email", self.email,
                "--agree-tos",
                "--no-eff-email",
                "-d", self.domain
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✅ {self.domain} 인증서 발급 성공")
                self.start_webserver()
                return True
            else:
                logger.error(f"❌ 인증서 발급 실패: {result.stderr}")
                self.start_webserver()
                return False
                
        except Exception as e:
            logger.error(f"❌ 인증서 발급 중 오류: {e}")
            self.start_webserver()
            return False
    
    def renew_certificate(self):
        """인증서 갱신"""
        try:
            result = subprocess.run([
                "sudo", "certbot", "renew", "--quiet"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ 인증서 갱신 성공")
                self.reload_webserver()
                return True
            else:
                logger.error(f"❌ 인증서 갱신 실패: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ 인증서 갱신 중 오류: {e}")
            return False
    
    def setup_auto_renewal(self):
        """자동 갱신 설정 (crontab)"""
        try:
            # certbot 실제 경로 찾기
            certbot_path = subprocess.run(["which", "certbot"], capture_output=True, text=True).stdout.strip()
            if not certbot_path:
                # macOS Homebrew 경로 체크
                if os.path.exists("/opt/homebrew/bin/certbot"):
                    certbot_path = "/opt/homebrew/bin/certbot"
                elif os.path.exists("/usr/local/bin/certbot"):
                    certbot_path = "/usr/local/bin/certbot"
                else:
                    logger.error("❌ certbot을 찾을 수 없습니다")
                    return False
            
            logger.info(f"📍 certbot 경로: {certbot_path}")
            
            # crontab 항목 생성
            cron_entry = f"0 2 * * * {certbot_path} renew --quiet --post-hook 'sudo nginx -s reload'\n"
            
            # 현재 crontab 확인
            result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
            current_cron = result.stdout if result.returncode == 0 else ""
            
            # certbot 항목이 이미 있는지 확인
            if "certbot renew" in current_cron:
                logger.info("✅ 자동 갱신이 이미 설정되어 있음")
                return True
            
            # 새 cron 항목 추가
            new_cron = current_cron + cron_entry
            
            # crontab 업데이트
            process = subprocess.Popen(["crontab", "-"], stdin=subprocess.PIPE, text=True)
            process.communicate(input=new_cron)
            
            if process.returncode == 0:
                logger.info("✅ 자동 갱신 cron 설정 완료")
                logger.info("   - 매일 오전 2시에 인증서 갱신 시도")
                return True
            else:
                logger.error("❌ cron 설정 실패")
                return False
                
        except Exception as e:
            logger.error(f"❌ 자동 갱신 설정 중 오류: {e}")
            return False
    
    def stop_webserver(self):
        """웹서버 중지"""
        try:
            # Nginx 중지
            subprocess.run(["sudo", "nginx", "-s", "stop"], capture_output=True)
            logger.info("🛑 Nginx 중지됨")
        except:
            pass
        
        try:
            # API 서버 중지 (blyan 서버)
            subprocess.run(["./server.sh", "stop", "api"], capture_output=True)
            logger.info("🛑 API 서버 중지됨")
        except:
            pass
    
    def start_webserver(self):
        """웹서버 시작"""
        try:
            # API 서버 시작
            subprocess.run(["./server.sh", "start", "api"], capture_output=True)
            logger.info("🚀 API 서버 시작됨")
        except:
            pass
        
        try:
            # Nginx 시작
            subprocess.run(["sudo", "nginx"], capture_output=True)
            logger.info("🚀 Nginx 시작됨")
        except:
            pass
    
    def reload_webserver(self):
        """웹서버 리로드 (인증서 갱신 후)"""
        try:
            subprocess.run(["sudo", "nginx", "-s", "reload"], capture_output=True)
            logger.info("🔄 Nginx 리로드됨")
        except Exception as e:
            logger.error(f"❌ Nginx 리로드 실패: {e}")
    
    def create_nginx_config(self):
        """Nginx SSL 설정 생성"""
        nginx_config = f'''
server {{
    listen 80;
    server_name {self.domain} www.{self.domain};
    
    # HTTP to HTTPS redirect
    return 301 https://$server_name$request_uri;
}}

server {{
    listen 443 ssl http2;
    server_name {self.domain} www.{self.domain};
    
    # SSL 설정
    ssl_certificate /etc/letsencrypt/live/{self.domain}/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/{self.domain}/privkey.pem;
    
    # SSL 보안 설정
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # 보안 헤더
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    
    # API 프록시
    location / {{
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }}
}}
'''
        
        config_file = self.config_dir / "nginx_ssl.conf"
        config_file.write_text(nginx_config.strip())
        
        logger.info(f"📝 Nginx SSL 설정 생성: {config_file}")
        logger.info("💡 이 파일을 /etc/nginx/sites-available/로 복사하세요:")
        logger.info(f"   sudo cp {config_file} /etc/nginx/sites-available/{self.domain}")
        logger.info(f"   sudo ln -s /etc/nginx/sites-available/{self.domain} /etc/nginx/sites-enabled/")
    
    def log_renewal_attempt(self, success, message):
        """갱신 시도 로그 저장"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "message": message
        }
        
        logs = []
        if self.log_file.exists():
            try:
                logs = json.loads(self.log_file.read_text())
            except:
                logs = []
        
        logs.append(log_entry)
        
        # 최근 50개 로그만 유지
        if len(logs) > 50:
            logs = logs[-50:]
        
        self.log_file.write_text(json.dumps(logs, indent=2))
    
    def run_ssl_management(self):
        """전체 SSL 관리 프로세스 실행"""
        logger.info("🔐 SSL 인증서 관리 시작...")
        
        # 현재 상태 확인
        status = self.check_certificate_status()
        logger.info(f"📋 인증서 상태: {status['message']}")
        
        if not status.get("exists"):
            # 인증서가 없으면 새로 발급
            logger.info("🆕 새 인증서 발급 시도...")
            
            if not self.install_certbot():
                return False
            
            if self.obtain_certificate():
                self.log_renewal_attempt(True, "새 인증서 발급 성공")
            else:
                self.log_renewal_attempt(False, "새 인증서 발급 실패")
                return False
        
        elif status.get("needs_renewal"):
            # 갱신 필요
            logger.info("🔄 인증서 갱신 시도...")
            
            if self.renew_certificate():
                self.log_renewal_attempt(True, f"인증서 갱신 성공 ({status['days_left']}일 남음)")
            else:
                self.log_renewal_attempt(False, f"인증서 갱신 실패 ({status['days_left']}일 남음)")
                return False
        
        # 자동 갱신 설정
        self.setup_auto_renewal()
        
        # Nginx 설정 생성
        self.create_nginx_config()
        
        logger.info("✅ SSL 관리 완료!")
        return True

def main():
    # 도메인과 이메일 설정 (실제 값으로 변경 필요)
    ssl_manager = SSLManager(
        domain="blyan.com", 
        email="admin@blyan.com"
    )
    
    ssl_manager.run_ssl_management()

if __name__ == "__main__":
    main()
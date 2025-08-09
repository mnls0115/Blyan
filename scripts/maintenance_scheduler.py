#!/usr/bin/env python3
"""
시스템 유지보수 스케줄러
System maintenance scheduler - 모든 유지보수 작업을 자동화
"""

import schedule
import time
import subprocess
import logging
from pathlib import Path
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/maintenance.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MaintenanceScheduler:
    def __init__(self):
        self.scripts_dir = Path("./scripts")
        
    def run_script(self, script_name, description):
        """스크립트 실행"""
        try:
            logger.info(f"🔧 {description} 시작...")
            script_path = self.scripts_dir / script_name
            
            if not script_path.exists():
                logger.error(f"❌ 스크립트를 찾을 수 없음: {script_path}")
                return False
            
            result = subprocess.run([
                "python3", str(script_path)
            ], capture_output=True, text=True, cwd=".")
            
            if result.returncode == 0:
                logger.info(f"✅ {description} 완료")
                return True
            else:
                logger.error(f"❌ {description} 실패: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ {description} 중 오류: {e}")
            return False
    
    def memory_optimization(self):
        """메모리 최적화 작업"""
        self.run_script("optimize_memory.py", "메모리 최적화")
    
    def dependency_check(self):
        """의존성 검사 및 보안 업데이트"""
        # 비대화형 모드로 실행
        try:
            logger.info(f"🔧 의존성 관리 시작...")
            script_path = self.scripts_dir / "dependency_manager.py"
            
            if not script_path.exists():
                logger.error(f"❌ 스크립트를 찾을 수 없음: {script_path}")
                return False
            
            # --auto-approve 플래그와 함께 실행
            result = subprocess.run([
                "python3", str(script_path), "--auto-approve"
            ], capture_output=True, text=True, cwd=".")
            
            if result.returncode == 0:
                logger.info(f"✅ 의존성 관리 완료")
                return True
            else:
                logger.error(f"❌ 의존성 관리 실패: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ 의존성 관리 중 오류: {e}")
            return False
    
    def ssl_renewal_check(self):
        """SSL 인증서 갱신 확인"""
        self.run_script("ssl_manager.py", "SSL 인증서 관리")
    
    def system_cleanup(self):
        """시스템 정리 작업"""
        try:
            logger.info("🧹 시스템 정리 중...")
            
            # 로그 파일 정리 (30일 이상 된 파일 삭제)
            log_dir = Path("./logs")
            if log_dir.exists():
                import os
                import time
                
                now = time.time()
                for file_path in log_dir.glob("*.log*"):
                    if file_path.is_file():
                        file_age = now - file_path.stat().st_mtime
                        if file_age > 30 * 24 * 3600:  # 30일
                            file_path.unlink()
                            logger.info(f"🗑️ 오래된 로그 파일 삭제: {file_path}")
            
            # 임시 파일 정리
            temp_files = [
                "./requirements_backup_*.txt",
                "./*.tmp",
                "./.DS_Store"
            ]
            
            for pattern in temp_files:
                for file_path in Path(".").glob(pattern):
                    if file_path.is_file():
                        file_path.unlink()
                        logger.info(f"🗑️ 임시 파일 삭제: {file_path}")
            
            logger.info("✅ 시스템 정리 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ 시스템 정리 중 오류: {e}")
            return False
    
    def health_check(self):
        """시스템 헬스 체크"""
        try:
            logger.info("🏥 시스템 헬스 체크 중...")
            
            # API 서버 상태 확인
            import requests
            try:
                response = requests.get("http://127.0.0.1:8000/health", timeout=10)
                if response.status_code == 200:
                    logger.info("✅ API 서버 정상")
                else:
                    logger.warning(f"⚠️ API 서버 응답 이상: {response.status_code}")
            except requests.RequestException as e:
                logger.error(f"❌ API 서버 연결 실패: {e}")
            
            # Redis 상태 확인
            try:
                result = subprocess.run(["redis-cli", "ping"], 
                                      capture_output=True, text=True, timeout=10)
                if result.stdout.strip() == "PONG":
                    logger.info("✅ Redis 정상")
                else:
                    logger.warning("⚠️ Redis 응답 이상")
            except Exception as e:
                logger.error(f"❌ Redis 연결 실패: {e}")
            
            # 디스크 사용량 확인
            import psutil
            disk_usage = psutil.disk_usage('.')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            
            if disk_percent > 90:
                logger.error(f"❌ 디스크 사용량 위험: {disk_percent:.1f}%")
            elif disk_percent > 80:
                logger.warning(f"⚠️ 디스크 사용량 높음: {disk_percent:.1f}%")
            else:
                logger.info(f"✅ 디스크 사용량 정상: {disk_percent:.1f}%")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 헬스 체크 중 오류: {e}")
            return False
    
    def setup_schedule(self):
        """유지보수 스케줄 설정"""
        logger.info("📅 유지보수 스케줄 설정 중...")
        
        # 매 시간마다 - 메모리 최적화
        schedule.every().hour.do(self.memory_optimization)
        
        # 매일 오전 2시 - 의존성 검사
        schedule.every().day.at("02:00").do(self.dependency_check)
        
        # 매일 오전 3시 - SSL 인증서 확인
        schedule.every().day.at("03:00").do(self.ssl_renewal_check)
        
        # 매일 오전 4시 - 시스템 정리
        schedule.every().day.at("04:00").do(self.system_cleanup)
        
        # 매 30분마다 - 헬스 체크
        schedule.every(30).minutes.do(self.health_check)
        
        logger.info("✅ 스케줄 설정 완료")
        logger.info("   - 메모리 최적화: 매 시간")
        logger.info("   - 의존성 검사: 매일 02:00")
        logger.info("   - SSL 관리: 매일 03:00")
        logger.info("   - 시스템 정리: 매일 04:00")
        logger.info("   - 헬스 체크: 매 30분")
    
    def run_scheduler(self):
        """스케줄러 실행"""
        self.setup_schedule()
        
        logger.info("🚀 유지보수 스케줄러 시작...")
        logger.info("   Ctrl+C로 중지")
        
        # 시작 시 한 번 헬스 체크 실행
        self.health_check()
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # 1분마다 체크
                
        except KeyboardInterrupt:
            logger.info("🛑 스케줄러 중지됨")

def main():
    # 스케줄러 생성 및 실행
    scheduler = MaintenanceScheduler()
    
    # 로그 디렉토리 생성
    Path("./logs").mkdir(exist_ok=True)
    
    # 스케줄러 실행
    scheduler.run_scheduler()

if __name__ == "__main__":
    main()
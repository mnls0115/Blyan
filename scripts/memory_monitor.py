#!/usr/bin/env python3
"""
메모리 사용량 모니터링 및 알림 시스템
Memory usage monitoring and alert system
"""

import psutil
import time
import json
import logging
from datetime import datetime
from pathlib import Path
import subprocess

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryMonitor:
    def __init__(self, alert_threshold=85.0, critical_threshold=95.0):
        """
        메모리 모니터 초기화
        Args:
            alert_threshold: 경고 임계값 (%)
            critical_threshold: 위험 임계값 (%)
        """
        self.alert_threshold = alert_threshold
        self.critical_threshold = critical_threshold
        self.log_file = Path("./logs/memory_usage.json")
        self.log_file.parent.mkdir(exist_ok=True)
        
    def get_memory_info(self):
        """현재 메모리 정보 가져오기"""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_gb": round(memory.total / 1024**3, 2),
            "available_gb": round(memory.available / 1024**3, 2),
            "used_gb": round(memory.used / 1024**3, 2),
            "used_percent": memory.percent,
            "swap_used_percent": swap.percent,
            "processes": self.get_top_memory_processes()
        }
    
    def get_top_memory_processes(self, count=5):
        """메모리 사용량 상위 프로세스 가져오기"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'memory_percent']):
            try:
                if proc.info['memory_percent'] > 0.1:  # 0.1% 이상만 수집
                    processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'memory_percent': round(proc.info['memory_percent'], 2)
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        # 메모리 사용량 기준 정렬
        processes.sort(key=lambda x: x['memory_percent'], reverse=True)
        return processes[:count]
    
    def log_memory_usage(self, memory_info):
        """메모리 사용량 로그 저장"""
        logs = []
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r') as f:
                    logs = json.load(f)
            except:
                logs = []
        
        logs.append(memory_info)
        
        # 최근 24시간 데이터만 보관 (1분마다 저장시 1440개)
        if len(logs) > 1440:
            logs = logs[-1440:]
            
        with open(self.log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    
    def check_and_alert(self, memory_info):
        """메모리 사용량 체크 및 알림"""
        used_percent = memory_info['used_percent']
        
        if used_percent >= self.critical_threshold:
            logger.error(f"🚨 위험: 메모리 사용량 {used_percent}% (임계값: {self.critical_threshold}%)")
            self.cleanup_memory()
        elif used_percent >= self.alert_threshold:
            logger.warning(f"⚠️ 경고: 메모리 사용량 {used_percent}% (임계값: {self.alert_threshold}%)")
            
        # 상위 프로세스 출력
        if used_percent >= self.alert_threshold:
            logger.info("상위 메모리 사용 프로세스:")
            for proc in memory_info['processes'][:3]:
                logger.info(f"  - {proc['name']} (PID: {proc['pid']}): {proc['memory_percent']}%")
    
    def cleanup_memory(self):
        """메모리 정리 작업"""
        try:
            # Python 가비지 컬렉션
            import gc
            gc.collect()
            
            # 시스템 캐시 정리 (macOS)
            subprocess.run(['sudo', 'purge'], check=False, capture_output=True)
            logger.info("메모리 정리 작업 완료")
            
        except Exception as e:
            logger.error(f"메모리 정리 중 오류: {e}")
    
    def run_monitor(self, interval=60):
        """지속적 모니터링 실행"""
        logger.info(f"메모리 모니터링 시작 (간격: {interval}초)")
        logger.info(f"경고 임계값: {self.alert_threshold}%, 위험 임계값: {self.critical_threshold}%")
        
        try:
            while True:
                memory_info = self.get_memory_info()
                self.log_memory_usage(memory_info)
                self.check_and_alert(memory_info)
                
                logger.info(f"메모리 사용량: {memory_info['used_percent']}% "
                          f"({memory_info['used_gb']:.1f}GB / {memory_info['total_gb']:.1f}GB)")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("모니터링 종료")

def main():
    # 모니터 생성 및 실행
    monitor = MemoryMonitor(alert_threshold=85.0, critical_threshold=95.0)
    
    # 현재 상태 출력
    current_info = monitor.get_memory_info()
    print(f"📊 현재 메모리 상태:")
    print(f"   전체: {current_info['total_gb']}GB")
    print(f"   사용: {current_info['used_gb']}GB ({current_info['used_percent']}%)")
    print(f"   가용: {current_info['available_gb']}GB")
    
    # 지속 모니터링 시작 (테스트용으로 10초 간격)
    monitor.run_monitor(interval=10)

if __name__ == "__main__":
    main()
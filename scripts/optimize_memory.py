#!/usr/bin/env python3
"""
메모리 최적화 스크립트
Memory optimization script for AI model loading
"""

import torch
import gc
import psutil
import json
import sys
from pathlib import Path

class MemoryOptimizer:
    def __init__(self):
        self.config_file = Path("./config/memory_config.json")
        self.load_config()
    
    def load_config(self):
        """메모리 최적화 설정 로드"""
        if self.config_file.exists():
            with open(self.config_file) as f:
                self.config = json.load(f)
        else:
            # 기본 설정
            self.config = {
                "max_model_memory_gb": 4.0,
                "expert_cache_limit_gb": 2.0,
                "inference_batch_size": 1,
                "enable_gradient_checkpointing": True,
                "use_cpu_offload": True,
                "torch_cache_limit_gb": 1.0
            }
            self.save_config()
    
    def save_config(self):
        """설정 저장"""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get_available_memory_gb(self):
        """사용 가능한 메모리 (GB) 계산"""
        memory = psutil.virtual_memory()
        return memory.available / (1024**3)
    
    def optimize_torch_settings(self):
        """PyTorch 메모리 최적화"""
        print("🔧 PyTorch 메모리 최적화 중...")
        
        # CUDA 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # CUDA 메모리 할당 방식 최적화
            torch.backends.cudnn.benchmark = False  # 메모리 사용량 감소
            torch.backends.cudnn.deterministic = True
        
        # CPU 메모리 최적화
        torch.set_num_threads(2)  # CPU 스레드 제한으로 메모리 절약
        
        # 가비지 컬렉션
        gc.collect()
        
        print(f"✅ PyTorch 설정 완료")
        print(f"   - CUDA 사용 가능: {torch.cuda.is_available()}")
        print(f"   - CPU 스레드: {torch.get_num_threads()}")
    
    def update_model_config(self):
        """모델 설정 업데이트"""
        available_memory = self.get_available_memory_gb()
        
        # 메모리에 따른 동적 설정
        if available_memory < 8:
            self.config.update({
                "max_model_memory_gb": min(2.0, available_memory * 0.25),
                "expert_cache_limit_gb": min(1.0, available_memory * 0.125),
                "inference_batch_size": 1
            })
        elif available_memory < 16:
            self.config.update({
                "max_model_memory_gb": min(4.0, available_memory * 0.25),
                "expert_cache_limit_gb": min(2.0, available_memory * 0.125),
                "inference_batch_size": 2
            })
        else:
            self.config.update({
                "max_model_memory_gb": min(8.0, available_memory * 0.3),
                "expert_cache_limit_gb": min(4.0, available_memory * 0.2),
                "inference_batch_size": 4
            })
        
        self.save_config()
        print(f"📝 메모리 설정 업데이트 완료 (사용 가능: {available_memory:.1f}GB)")
        print(f"   - 최대 모델 메모리: {self.config['max_model_memory_gb']}GB")
        print(f"   - 전문가 캐시 한도: {self.config['expert_cache_limit_gb']}GB")
        print(f"   - 배치 크기: {self.config['inference_batch_size']}")
    
    def setup_monitoring_script(self):
        """모니터링 스크립트를 백그라운드에서 실행"""
        import subprocess
        import os
        
        script_path = Path(__file__).parent / "memory_monitor.py"
        
        # Python 경로 찾기
        python_paths = [
            f"{os.getcwd()}/myenv/bin/python",     # 현재 디렉토리 가상환경
            f"{os.getcwd()}/.venv/bin/python",     # .venv 가상환경
            sys.executable,                         # 현재 실행 중인 Python
            "/usr/bin/python3",                     # 시스템 Python3
            "python3"                               # PATH의 Python3
        ]
        
        python_exe = None
        for path in python_paths:
            try:
                if Path(path).exists() or subprocess.run(["which", path], capture_output=True).returncode == 0:
                    python_exe = path
                    break
            except:
                continue
        
        if not python_exe:
            print(f"⚠️ Python 실행 파일을 찾을 수 없습니다")
            return False
        
        try:
            # 백그라운드 실행
            process = subprocess.Popen(
                [python_exe, str(script_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # PID 저장
            pid_file = Path("./logs/memory_monitor.pid")
            pid_file.parent.mkdir(exist_ok=True)
            with open(pid_file, 'w') as f:
                f.write(str(process.pid))
            
            print(f"🔍 메모리 모니터링 시작됨 (PID: {process.pid})")
            return True
            
        except Exception as e:
            print(f"⚠️ 모니터링 스크립트 실행 실패: {e}")
            return False

def main():
    optimizer = MemoryOptimizer()
    
    print("🚀 메모리 최적화 시작...")
    print(f"💾 현재 사용 가능한 메모리: {optimizer.get_available_memory_gb():.1f}GB")
    
    # 최적화 실행
    optimizer.optimize_torch_settings()
    optimizer.update_model_config()
    
    # 모니터링 시작
    optimizer.setup_monitoring_script()
    
    print("✅ 메모리 최적화 완료!")

if __name__ == "__main__":
    main()
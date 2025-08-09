#!/usr/bin/env python3
"""
Python 의존성 관리 및 업데이트 스크립트
Python dependency management and update script
"""

import subprocess
import json
import requests
from pathlib import Path
from datetime import datetime
import sys
import os
import argparse

class DependencyManager:
    def __init__(self, auto_approve=False):
        self.requirements_file = Path("requirements.txt")
        self.frozen_file = Path("requirements_frozen.txt")
        self.security_file = Path("requirements_security.txt")
        self.log_file = Path("logs/dependency_updates.json")
        self.log_file.parent.mkdir(exist_ok=True)
        # 비대화형 모드 설정
        self.auto_approve = auto_approve or os.getenv("DEPENDENCY_AUTO_APPROVE", "").lower() in ("1", "true", "yes")
        
    def get_installed_packages(self):
        """설치된 패키지 목록 가져오기"""
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "list", "--format=json"], 
                                  capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"패키지 목록 조회 실패: {e}")
            return []
    
    def check_security_vulnerabilities(self):
        """보안 취약점 검사"""
        print("🔍 보안 취약점 검사 중...")
        
        try:
            # safety 패키지 설치 확인
            subprocess.run([sys.executable, "-m", "pip", "show", "safety"], 
                          capture_output=True, check=True)
        except subprocess.CalledProcessError:
            print("safety 패키지 설치 중...")
            subprocess.run([sys.executable, "-m", "pip", "install", "safety"], 
                          check=True)
        
        try:
            # safety check 실행
            result = subprocess.run([sys.executable, "-m", "safety", "check", "--json"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ 보안 취약점 없음")
                return []
            else:
                vulnerabilities = json.loads(result.stdout) if result.stdout else []
                print(f"⚠️ {len(vulnerabilities)}개의 보안 취약점 발견")
                for vuln in vulnerabilities:
                    print(f"   - {vuln.get('package_name')}: {vuln.get('vulnerability_id')}")
                return vulnerabilities
                
        except Exception as e:
            print(f"보안 검사 실패: {e}")
            return []
    
    def check_outdated_packages(self):
        """업데이트 가능한 패키지 확인"""
        print("📦 업데이트 가능한 패키지 확인 중...")
        
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "list", "--outdated", "--format=json"], 
                                  capture_output=True, text=True, check=True)
            outdated = json.loads(result.stdout)
            
            if outdated:
                print(f"📋 {len(outdated)}개의 업데이트 가능한 패키지:")
                for pkg in outdated[:5]:  # 상위 5개만 표시
                    print(f"   - {pkg['name']}: {pkg['version']} → {pkg['latest_version']}")
                if len(outdated) > 5:
                    print(f"   ... 및 {len(outdated) - 5}개 더")
            else:
                print("✅ 모든 패키지가 최신 버전입니다")
            
            return outdated
            
        except subprocess.CalledProcessError as e:
            print(f"패키지 확인 실패: {e}")
            return []
    
    def create_backup(self):
        """현재 환경 백업"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = Path(f"requirements_backup_{timestamp}.txt")
        
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "freeze"], 
                                  capture_output=True, text=True, check=True)
            backup_file.write_text(result.stdout)
            print(f"💾 백업 생성: {backup_file}")
            return backup_file
        except Exception as e:
            print(f"백업 실패: {e}")
            return None
    
    def update_requirements_files(self):
        """requirements 파일들 업데이트"""
        print("📝 requirements 파일 업데이트 중...")
        
        try:
            # 현재 설치된 패키지로 frozen 파일 생성
            result = subprocess.run([sys.executable, "-m", "pip", "freeze"], 
                                  capture_output=True, text=True, check=True)
            self.frozen_file.write_text(result.stdout)
            print(f"✅ {self.frozen_file} 업데이트 완료")
            
            # 핵심 의존성만 requirements.txt에 유지
            core_packages = [
                "fastapi", "uvicorn", "torch", "transformers", "numpy", 
                "redis", "psutil", "tiktoken", "requests", "aiofiles"
            ]
            
            installed = self.get_installed_packages()
            core_requirements = []
            
            for pkg in installed:
                if pkg['name'].lower() in [p.lower() for p in core_packages]:
                    core_requirements.append(f"{pkg['name']}=={pkg['version']}")
            
            if core_requirements:
                self.requirements_file.write_text('\n'.join(core_requirements) + '\n')
                print(f"✅ {self.requirements_file} 업데이트 완료")
            
        except Exception as e:
            print(f"파일 업데이트 실패: {e}")
    
    def log_update(self, vulnerabilities, outdated, backup_file):
        """업데이트 로그 저장"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "vulnerabilities_count": len(vulnerabilities),
            "vulnerabilities": vulnerabilities,
            "outdated_count": len(outdated),
            "outdated_packages": [pkg['name'] for pkg in outdated],
            "backup_file": str(backup_file) if backup_file else None
        }
        
        logs = []
        if self.log_file.exists():
            try:
                logs = json.loads(self.log_file.read_text())
            except:
                logs = []
        
        logs.append(log_entry)
        
        # 최근 30개 로그만 유지
        if len(logs) > 30:
            logs = logs[-30:]
        
        self.log_file.write_text(json.dumps(logs, indent=2))
    
    def install_security_updates(self, vulnerabilities):
        """보안 업데이트 설치"""
        if not vulnerabilities:
            return
        
        print("🛡️ 보안 업데이트 설치 중...")
        
        for vuln in vulnerabilities:
            package_name = vuln.get('package_name')
            if package_name:
                try:
                    subprocess.run([sys.executable, "-m", "pip", "install", 
                                  "--upgrade", package_name], check=True)
                    print(f"✅ {package_name} 보안 업데이트 완료")
                except subprocess.CalledProcessError as e:
                    print(f"❌ {package_name} 업데이트 실패: {e}")
    
    def run_maintenance(self):
        """전체 유지보수 실행"""
        print("🔧 의존성 유지보수 시작...")
        
        # 백업 생성
        backup_file = self.create_backup()
        
        # 보안 취약점 검사
        vulnerabilities = self.check_security_vulnerabilities()
        
        # 업데이트 가능한 패키지 확인
        outdated = self.check_outdated_packages()
        
        # 보안 업데이트 설치
        if vulnerabilities:
            if self.auto_approve:
                print("🔧 자동 승인 모드: 보안 업데이트 설치 중...")
                self.install_security_updates(vulnerabilities)
            else:
                confirm = input("보안 취약점이 발견되었습니다. 업데이트하시겠습니까? (y/N): ")
                if confirm.lower() == 'y':
                    self.install_security_updates(vulnerabilities)
        
        # requirements 파일 업데이트
        self.update_requirements_files()
        
        # 로그 저장
        self.log_update(vulnerabilities, outdated, backup_file)
        
        print("✅ 의존성 유지보수 완료!")
        
        # 요약 출력
        print(f"\n📊 요약:")
        print(f"   - 보안 취약점: {len(vulnerabilities)}개")
        print(f"   - 업데이트 가능: {len(outdated)}개")
        print(f"   - 백업 파일: {backup_file}")

def main():
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description="Python 의존성 관리 도구")
    parser.add_argument(
        "--auto-approve", "-y",
        action="store_true",
        help="자동으로 모든 질문에 yes로 응답 (비대화형 모드)"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="검사만 수행하고 변경사항 적용 안함"
    )
    
    args = parser.parse_args()
    
    # 매니저 생성
    manager = DependencyManager(auto_approve=args.auto_approve)
    
    # 검사만 수행
    if args.check_only:
        print("🔍 검사 모드 (변경사항 적용 안함)")
        vulnerabilities = manager.check_security_vulnerabilities()
        outdated = manager.check_outdated_packages()
        print(f"\n📊 결과:")
        print(f"   - 보안 취약점: {len(vulnerabilities)}개")
        print(f"   - 업데이트 가능: {len(outdated)}개")
        sys.exit(0 if not vulnerabilities else 1)
    
    # 전체 유지보수 실행
    manager.run_maintenance()

if __name__ == "__main__":
    main()
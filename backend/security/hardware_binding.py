#!/usr/bin/env python3
"""
GPU UUID Hardware Binding System
Binds nodes to specific GPU hardware to prevent hardware spoofing and ensure authentic compute resources.
"""

import os
import json
import hashlib
import subprocess
import time
import platform
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timedelta
import threading
from enum import Enum

# Optional GPU libraries
try:
    import pynvml
    HAS_NVIDIA = True
except ImportError:
    HAS_NVIDIA = False

try:
    import py3nvml.py3nvml as nvml
    HAS_PY3NVML = True
except ImportError:
    HAS_PY3NVML = False


class HardwareType(Enum):
    """Types of hardware that can be bound."""
    NVIDIA_GPU = "nvidia_gpu"
    AMD_GPU = "amd_gpu"
    INTEL_GPU = "intel_gpu"
    CPU = "cpu"
    SYSTEM = "system"


class BindingStatus(Enum):
    """Hardware binding status."""
    BOUND = "bound"
    UNBOUND = "unbound"
    VERIFICATION_FAILED = "verification_failed"
    HARDWARE_CHANGED = "hardware_changed"
    MISSING = "missing"


@dataclass
class HardwareInfo:
    """Hardware component information."""
    hardware_id: str
    hardware_type: HardwareType
    name: str
    uuid: Optional[str]
    vendor: str
    memory_mb: int
    compute_capability: Optional[str]
    driver_version: Optional[str]
    firmware_version: Optional[str]
    pci_bus_id: Optional[str]
    temperature_c: Optional[float]
    power_draw_w: Optional[float]
    utilization_percent: Optional[float]
    last_seen: float
    metadata: Dict[str, Any]


@dataclass
class HardwareBinding:
    """Hardware binding record."""
    binding_id: str
    node_id: str
    hardware_id: str
    hardware_fingerprint: str
    binding_timestamp: float
    last_verified: float
    verification_count: int
    binding_status: BindingStatus
    expert_assignments: List[str]
    performance_metrics: Dict[str, float]
    trust_score: float  # 0.0 to 1.0
    metadata: Dict[str, Any]


@dataclass
class VerificationResult:
    """Hardware verification result."""
    hardware_id: str
    verification_timestamp: float
    verification_success: bool
    expected_fingerprint: str
    actual_fingerprint: Optional[str]
    hardware_present: bool
    performance_delta: float
    trust_score_change: float
    anomalies_detected: List[str]


class GPUDetector:
    """Detect and identify GPU hardware."""
    
    def __init__(self):
        self.nvidia_available = self._init_nvidia()
        self.detected_hardware: List[HardwareInfo] = []
    
    def _init_nvidia(self) -> bool:
        """Initialize NVIDIA GPU detection."""
        if HAS_NVIDIA:
            try:
                pynvml.nvmlInit()
                return True
            except Exception as e:
                print(f"Warning: NVIDIA GPU detection failed: {e}")
        
        if HAS_PY3NVML:
            try:
                nvml.nvmlInit()
                return True
            except Exception as e:
                print(f"Warning: py3nvml GPU detection failed: {e}")
        
        return False
    
    def detect_nvidia_gpus(self) -> List[HardwareInfo]:
        """Detect NVIDIA GPUs."""
        gpus = []
        
        if not self.nvidia_available:
            return gpus
        
        try:
            if HAS_NVIDIA:
                device_count = pynvml.nvmlDeviceGetCount()
                
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # Get basic info
                    name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    uuid = pynvml.nvmlDeviceGetUUID(handle).decode('utf-8')
                    
                    # Get memory info
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    memory_mb = mem_info.total // (1024 * 1024)
                    
                    # Get additional info
                    try:
                        driver_version = pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
                    except:
                        driver_version = None
                    
                    try:
                        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                        compute_capability = f"{major}.{minor}"
                    except:
                        compute_capability = None
                    
                    try:
                        pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
                        pci_bus_id = pci_info.busId.decode('utf-8')
                    except:
                        pci_bus_id = None
                    
                    # Get real-time metrics
                    try:
                        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    except:
                        temperature = None
                    
                    try:
                        power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                    except:
                        power_draw = None
                    
                    try:
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        util_percent = utilization.gpu
                    except:
                        util_percent = None
                    
                    hardware_info = HardwareInfo(
                        hardware_id=f"nvidia_gpu_{i}",
                        hardware_type=HardwareType.NVIDIA_GPU,
                        name=name,
                        uuid=uuid,
                        vendor="NVIDIA",
                        memory_mb=memory_mb,
                        compute_capability=compute_capability,
                        driver_version=driver_version,
                        firmware_version=None,
                        pci_bus_id=pci_bus_id,
                        temperature_c=temperature,
                        power_draw_w=power_draw,
                        utilization_percent=util_percent,
                        last_seen=time.time(),
                        metadata={
                            "device_index": i,
                            "detection_method": "pynvml"
                        }
                    )
                    
                    gpus.append(hardware_info)
                    
        except Exception as e:
            print(f"Warning: NVIDIA GPU detection error: {e}")
        
        return gpus
    
    def detect_system_hardware(self) -> List[HardwareInfo]:
        """Detect system hardware (CPU, motherboard, etc.)."""
        hardware = []
        
        try:
            # CPU information
            if platform.system() == "Linux":
                try:
                    with open("/proc/cpuinfo", "r") as f:
                        cpuinfo = f.read()
                    
                    # Extract CPU model
                    cpu_lines = [line for line in cpuinfo.split('\n') if 'model name' in line]
                    if cpu_lines:
                        cpu_name = cpu_lines[0].split(':')[1].strip()
                    else:
                        cpu_name = "Unknown CPU"
                    
                    # Create CPU hardware info
                    cpu_info = HardwareInfo(
                        hardware_id="system_cpu",
                        hardware_type=HardwareType.CPU,
                        name=cpu_name,
                        uuid=None,
                        vendor=platform.processor() or "Unknown",
                        memory_mb=0,
                        compute_capability=None,
                        driver_version=None,
                        firmware_version=None,
                        pci_bus_id=None,
                        temperature_c=None,
                        power_draw_w=None,
                        utilization_percent=None,
                        last_seen=time.time(),
                        metadata={
                            "architecture": platform.architecture()[0],
                            "cores": os.cpu_count(),
                            "system": platform.system(),
                            "release": platform.release()
                        }
                    )
                    
                    hardware.append(cpu_info)
                    
                except Exception as e:
                    print(f"Warning: CPU detection failed: {e}")
            
            # System motherboard/machine ID
            try:
                machine_id = None
                
                # Try different methods to get machine ID
                machine_id_files = [
                    "/etc/machine-id",
                    "/var/lib/dbus/machine-id",
                    "/sys/class/dmi/id/product_uuid"
                ]
                
                for id_file in machine_id_files:
                    try:
                        with open(id_file, "r") as f:
                            machine_id = f.read().strip()
                            break
                    except:
                        continue
                
                if not machine_id:
                    # Generate a pseudo-unique ID based on system info
                    system_info = f"{platform.node()}{platform.system()}{platform.release()}"
                    machine_id = hashlib.sha256(system_info.encode()).hexdigest()[:32]
                
                system_info = HardwareInfo(
                    hardware_id="system_board",
                    hardware_type=HardwareType.SYSTEM,
                    name=f"{platform.system()} System",
                    uuid=machine_id,
                    vendor=platform.system(),
                    memory_mb=0,
                    compute_capability=None,
                    driver_version=None,
                    firmware_version=platform.release(),
                    pci_bus_id=None,
                    temperature_c=None,
                    power_draw_w=None,
                    utilization_percent=None,
                    last_seen=time.time(),
                    metadata={
                        "hostname": platform.node(),
                        "platform": platform.platform(),
                        "python_version": platform.python_version()
                    }
                )
                
                hardware.append(system_info)
                
            except Exception as e:
                print(f"Warning: System hardware detection failed: {e}")
                
        except Exception as e:
            print(f"Warning: System hardware detection error: {e}")
        
        return hardware
    
    def detect_all_hardware(self) -> List[HardwareInfo]:
        """Detect all available hardware."""
        all_hardware = []
        
        # Detect NVIDIA GPUs
        nvidia_gpus = self.detect_nvidia_gpus()
        all_hardware.extend(nvidia_gpus)
        
        # Detect system hardware
        system_hardware = self.detect_system_hardware()
        all_hardware.extend(system_hardware)
        
        self.detected_hardware = all_hardware
        return all_hardware


class HardwareBindingManager:
    """Manage hardware bindings for node authentication."""
    
    def __init__(self, storage_dir: Path = None):
        self.storage_dir = storage_dir or Path("./data/hardware_binding")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.gpu_detector = GPUDetector()
        self.bindings: Dict[str, HardwareBinding] = {}
        self.verification_history: List[VerificationResult] = []
        
        # Configuration
        self.trust_score_threshold = 0.7
        self.verification_interval_minutes = 30
        self.max_performance_delta = 0.2  # 20% performance variation allowed
        
        # Monitoring
        self.monitoring_enabled = True
        self.monitor_thread = None
        
        self._load_bindings_db()
        self._start_hardware_monitoring()
        
        print(f"🔗 Hardware Binding Manager initialized")
        print(f"   Storage: {self.storage_dir}")
        print(f"   GPU Detection: {'✅' if self.gpu_detector.nvidia_available else '❌'}")
        print(f"   Bindings loaded: {len(self.bindings)}")
    
    def _load_bindings_db(self):
        """Load hardware bindings from storage."""
        bindings_file = self.storage_dir / "hardware_bindings.json"
        if bindings_file.exists():
            try:
                with open(bindings_file) as f:
                    data = json.load(f)
                    for binding_id, binding_data in data.items():
                        binding = HardwareBinding(**binding_data)
                        binding.binding_status = BindingStatus(binding_data["binding_status"])
                        self.bindings[binding_id] = binding
            except Exception as e:
                print(f"Warning: Failed to load hardware bindings: {e}")
    
    def _save_bindings_db(self):
        """Save hardware bindings to storage."""
        bindings_file = self.storage_dir / "hardware_bindings.json"
        try:
            data = {}
            for binding_id, binding in self.bindings.items():
                binding_data = asdict(binding)
                binding_data["binding_status"] = binding.binding_status.value
                data[binding_id] = binding_data
            
            # Atomic write
            temp_file = bindings_file.with_suffix(".tmp")
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            temp_file.replace(bindings_file)
            
        except Exception as e:
            print(f"Warning: Failed to save hardware bindings: {e}")
    
    def create_hardware_fingerprint(self, hardware_list: List[HardwareInfo]) -> str:
        """Create a unique fingerprint from hardware configuration."""
        fingerprint_data = []
        
        for hardware in sorted(hardware_list, key=lambda x: x.hardware_id):
            # Include stable hardware characteristics
            hw_data = {
                "type": hardware.hardware_type.value,
                "name": hardware.name,
                "uuid": hardware.uuid,
                "vendor": hardware.vendor,
                "memory_mb": hardware.memory_mb,
                "compute_capability": hardware.compute_capability,
                "pci_bus_id": hardware.pci_bus_id
            }
            
            # Create hash of hardware data
            hw_json = json.dumps(hw_data, sort_keys=True)
            hw_hash = hashlib.sha256(hw_json.encode()).hexdigest()
            fingerprint_data.append(hw_hash)
        
        # Combine all hardware hashes
        combined_fingerprint = hashlib.sha256(
            "".join(sorted(fingerprint_data)).encode()
        ).hexdigest()
        
        return combined_fingerprint
    
    def bind_node_to_hardware(self, node_id: str, expert_assignments: List[str] = None) -> Optional[str]:
        """Bind a node to its current hardware configuration."""
        try:
            # Detect current hardware
            current_hardware = self.gpu_detector.detect_all_hardware()
            
            if not current_hardware:
                print(f"❌ No hardware detected for node {node_id}")
                return None
            
            # Create hardware fingerprint
            fingerprint = self.create_hardware_fingerprint(current_hardware)
            
            # Create binding
            binding_id = f"binding_{node_id}_{int(time.time())}"
            
            binding = HardwareBinding(
                binding_id=binding_id,
                node_id=node_id,
                hardware_id=",".join([hw.hardware_id for hw in current_hardware]),
                hardware_fingerprint=fingerprint,
                binding_timestamp=time.time(),
                last_verified=time.time(),
                verification_count=1,
                binding_status=BindingStatus.BOUND,
                expert_assignments=expert_assignments or [],
                performance_metrics={},
                trust_score=1.0,
                metadata={
                    "hardware_count": len(current_hardware),
                    "gpu_count": len([hw for hw in current_hardware if hw.hardware_type == HardwareType.NVIDIA_GPU]),
                    "creation_method": "initial_binding"
                }
            )
            
            self.bindings[binding_id] = binding
            self._save_bindings_db()
            
            print(f"🔗 Hardware binding created: {binding_id}")
            print(f"   Node: {node_id}")
            print(f"   Hardware: {len(current_hardware)} components")
            print(f"   GPUs: {binding.metadata['gpu_count']}")
            print(f"   Fingerprint: {fingerprint[:16]}...")
            
            # Record security event
            from .monitoring import record_security_event
            record_security_event(
                "hardware_binding_created",
                "hardware_binding_manager",
                {
                    "binding_id": binding_id,
                    "node_id": node_id,
                    "hardware_count": len(current_hardware),
                    "gpu_count": binding.metadata["gpu_count"]
                }
            )
            
            return binding_id
            
        except Exception as e:
            print(f"❌ Failed to bind hardware for node {node_id}: {e}")
            return None
    
    def verify_hardware_binding(self, binding_id: str) -> VerificationResult:
        """Verify that hardware binding is still valid."""
        binding = self.bindings.get(binding_id)
        if not binding:
            return VerificationResult(
                hardware_id="unknown",
                verification_timestamp=time.time(),
                verification_success=False,
                expected_fingerprint="",
                actual_fingerprint=None,
                hardware_present=False,
                performance_delta=0.0,
                trust_score_change=-1.0,
                anomalies_detected=["binding_not_found"]
            )
        
        try:
            # Detect current hardware
            current_hardware = self.gpu_detector.detect_all_hardware()
            
            if not current_hardware:
                return VerificationResult(
                    hardware_id=binding.hardware_id,
                    verification_timestamp=time.time(),
                    verification_success=False,
                    expected_fingerprint=binding.hardware_fingerprint,
                    actual_fingerprint=None,
                    hardware_present=False,
                    performance_delta=0.0,
                    trust_score_change=-0.3,
                    anomalies_detected=["no_hardware_detected"]
                )
            
            # Create current fingerprint
            current_fingerprint = self.create_hardware_fingerprint(current_hardware)
            
            # Check if fingerprint matches
            fingerprint_match = current_fingerprint == binding.hardware_fingerprint
            
            # Calculate performance delta (placeholder)
            performance_delta = 0.0  # TODO: Implement actual performance comparison
            
            # Detect anomalies
            anomalies = []
            if not fingerprint_match:
                anomalies.append("hardware_fingerprint_mismatch")
            
            if performance_delta > self.max_performance_delta:
                anomalies.append("performance_degradation")
            
            # Calculate trust score change
            trust_score_change = 0.0
            if fingerprint_match:
                trust_score_change = 0.1  # Increase trust on successful verification
            else:
                trust_score_change = -0.5  # Significant decrease on mismatch
            
            verification_success = fingerprint_match and performance_delta <= self.max_performance_delta
            
            # Update binding
            binding.last_verified = time.time()
            binding.verification_count += 1
            binding.trust_score = max(0.0, min(1.0, binding.trust_score + trust_score_change))
            
            if verification_success:
                binding.binding_status = BindingStatus.BOUND
            else:
                binding.binding_status = BindingStatus.VERIFICATION_FAILED
                if not fingerprint_match:
                    binding.binding_status = BindingStatus.HARDWARE_CHANGED
            
            self._save_bindings_db()
            
            result = VerificationResult(
                hardware_id=binding.hardware_id,
                verification_timestamp=time.time(),
                verification_success=verification_success,
                expected_fingerprint=binding.hardware_fingerprint,
                actual_fingerprint=current_fingerprint,
                hardware_present=True,
                performance_delta=performance_delta,
                trust_score_change=trust_score_change,
                anomalies_detected=anomalies
            )
            
            self.verification_history.append(result)
            
            if verification_success:
                print(f"✅ Hardware verification passed: {binding_id}")
            else:
                print(f"❌ Hardware verification failed: {binding_id}")
                print(f"   Anomalies: {', '.join(anomalies)}")
                
                # Record security event
                from .monitoring import record_security_event
                record_security_event(
                    "hardware_verification_failed",
                    "hardware_binding_manager",
                    {
                        "binding_id": binding_id,
                        "node_id": binding.node_id,
                        "anomalies": anomalies,
                        "trust_score": binding.trust_score
                    }
                )
            
            return result
            
        except Exception as e:
            print(f"❌ Hardware verification error for {binding_id}: {e}")
            
            return VerificationResult(
                hardware_id=binding.hardware_id,
                verification_timestamp=time.time(),
                verification_success=False,
                expected_fingerprint=binding.hardware_fingerprint,
                actual_fingerprint=None,
                hardware_present=False,
                performance_delta=0.0,
                trust_score_change=-0.2,
                anomalies_detected=["verification_error"]
            )
    
    def should_trust_node(self, node_id: str) -> Tuple[bool, float, List[str]]:
        """Determine if a node should be trusted based on hardware binding."""
        node_bindings = [b for b in self.bindings.values() if b.node_id == node_id]
        
        if not node_bindings:
            return False, 0.0, ["no_hardware_binding"]
        
        # Get the most recent binding
        latest_binding = max(node_bindings, key=lambda x: x.binding_timestamp)
        
        trust_issues = []
        
        # Check binding status
        if latest_binding.binding_status != BindingStatus.BOUND:
            trust_issues.append(f"binding_status_{latest_binding.binding_status.value}")
        
        # Check trust score
        if latest_binding.trust_score < self.trust_score_threshold:
            trust_issues.append("low_trust_score")
        
        # Check last verification time
        time_since_verification = time.time() - latest_binding.last_verified
        if time_since_verification > (self.verification_interval_minutes * 60 * 2):  # 2x interval
            trust_issues.append("verification_overdue")
        
        should_trust = len(trust_issues) == 0
        
        return should_trust, latest_binding.trust_score, trust_issues
    
    def _start_hardware_monitoring(self):
        """Start periodic hardware verification monitoring."""
        if not self.monitor_thread or not self.monitor_thread.is_alive():
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            print("🔍 Hardware binding monitoring started")
    
    def _monitoring_loop(self):
        """Periodic hardware verification loop."""
        while self.monitoring_enabled:
            try:
                # Verify all active bindings
                for binding_id, binding in self.bindings.items():
                    if binding.binding_status == BindingStatus.BOUND:
                        time_since_verification = time.time() - binding.last_verified
                        
                        if time_since_verification > (self.verification_interval_minutes * 60):
                            print(f"🔍 Verifying hardware binding: {binding_id}")
                            self.verify_hardware_binding(binding_id)
                
                # Sleep until next check
                time.sleep(300)  # 5 minutes
                
            except Exception as e:
                print(f"Warning: Hardware monitoring error: {e}")
                time.sleep(600)  # 10 minutes on error
    
    def get_hardware_binding_status(self) -> Dict:
        """Get hardware binding system status."""
        total_bindings = len(self.bindings)
        active_bindings = len([b for b in self.bindings.values() if b.binding_status == BindingStatus.BOUND])
        failed_bindings = len([b for b in self.bindings.values() if b.binding_status == BindingStatus.VERIFICATION_FAILED])
        
        avg_trust_score = 0.0
        if self.bindings:
            avg_trust_score = sum(b.trust_score for b in self.bindings.values()) / len(self.bindings)
        
        recent_verifications = [v for v in self.verification_history if time.time() - v.verification_timestamp < 3600]
        
        return {
            "total_bindings": total_bindings,
            "active_bindings": active_bindings,
            "failed_bindings": failed_bindings,
            "average_trust_score": avg_trust_score,
            "gpu_detection_available": self.gpu_detector.nvidia_available,
            "monitoring_enabled": self.monitoring_enabled,
            "verification_interval_minutes": self.verification_interval_minutes,
            "recent_verifications": len(recent_verifications),
            "successful_recent_verifications": len([v for v in recent_verifications if v.verification_success]),
            "binding_status_distribution": {
                status.value: len([b for b in self.bindings.values() if b.binding_status == status])
                for status in BindingStatus
            }
        }


# Global instance
hardware_binding_manager = HardwareBindingManager()


# Convenience functions
def bind_node_hardware(node_id: str, expert_assignments: List[str] = None) -> Optional[str]:
    """Bind a node to its current hardware."""
    return hardware_binding_manager.bind_node_to_hardware(node_id, expert_assignments)


def verify_node_hardware(binding_id: str) -> VerificationResult:
    """Verify node hardware binding."""
    return hardware_binding_manager.verify_hardware_binding(binding_id)


def check_node_trust(node_id: str) -> Tuple[bool, float, List[str]]:
    """Check if a node should be trusted based on hardware binding."""
    return hardware_binding_manager.should_trust_node(node_id)


def get_hardware_status() -> Dict:
    """Get hardware binding system status."""
    return hardware_binding_manager.get_hardware_binding_status()


def detect_current_hardware() -> List[HardwareInfo]:
    """Detect current hardware configuration."""
    return hardware_binding_manager.gpu_detector.detect_all_hardware()
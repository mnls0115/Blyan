#!/usr/bin/env python3
"""
SBOM and License Validation System
Software Bill of Materials tracking and license compliance validation.
"""

import os
import json
import hashlib
import subprocess
import time
import re
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timedelta
import threading
from enum import Enum

# Optional SPDX integration
try:
    import spdx_tools
    HAS_SPDX = True
except ImportError:
    HAS_SPDX = False


class LicenseRisk(Enum):
    """License risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Types of software components."""
    PYTHON_PACKAGE = "python_package"
    SYSTEM_LIBRARY = "system_library"
    AI_MODEL = "ai_model"
    DATASET = "dataset"
    CONTAINER_IMAGE = "container_image"
    BINARY = "binary"


@dataclass
class LicenseInfo:
    """License information and classification."""
    license_id: str
    license_name: str
    license_text: Optional[str]
    risk_level: LicenseRisk
    commercial_use: bool
    copyleft: bool
    patent_grant: bool
    trademark_use: bool
    attribution_required: bool
    source_disclosure_required: bool
    compatible_licenses: Set[str]
    incompatible_licenses: Set[str]


@dataclass
class SoftwareComponent:
    """Software component information."""
    component_id: str
    name: str
    version: str
    component_type: ComponentType
    supplier: Optional[str]
    download_location: Optional[str]
    file_hash: Optional[str]
    license_info: List[LicenseInfo]
    dependencies: List[str]
    vulnerabilities: List[Dict]
    last_updated: float
    verification_status: str  # "verified", "unverified", "failed"
    metadata: Dict[str, Any]


@dataclass
class SBOMReport:
    """SBOM validation report."""
    report_id: str
    timestamp: float
    total_components: int
    verified_components: int
    high_risk_licenses: int
    license_conflicts: List[Dict]
    missing_licenses: List[str]
    security_vulnerabilities: int
    compliance_status: str  # "compliant", "non_compliant", "needs_review"
    recommendations: List[str]
    components: List[SoftwareComponent]


class LicenseDatabase:
    """Database of known licenses and their classifications."""
    
    def __init__(self):
        # Well-known license classifications
        self.licenses = {
            "MIT": LicenseInfo(
                license_id="MIT",
                license_name="MIT License",
                license_text=None,
                risk_level=LicenseRisk.LOW,
                commercial_use=True,
                copyleft=False,
                patent_grant=False,
                trademark_use=False,
                attribution_required=True,
                source_disclosure_required=False,
                compatible_licenses={"Apache-2.0", "BSD-3-Clause", "ISC"},
                incompatible_licenses={"GPL-3.0", "AGPL-3.0"}
            ),
            "Apache-2.0": LicenseInfo(
                license_id="Apache-2.0",
                license_name="Apache License 2.0",
                license_text=None,
                risk_level=LicenseRisk.LOW,
                commercial_use=True,
                copyleft=False,
                patent_grant=True,
                trademark_use=False,
                attribution_required=True,
                source_disclosure_required=False,
                compatible_licenses={"MIT", "BSD-3-Clause", "ISC"},
                incompatible_licenses={"GPL-2.0"}
            ),
            "GPL-3.0": LicenseInfo(
                license_id="GPL-3.0",
                license_name="GNU General Public License v3.0",
                license_text=None,
                risk_level=LicenseRisk.HIGH,
                commercial_use=True,
                copyleft=True,
                patent_grant=True,
                trademark_use=False,
                attribution_required=True,
                source_disclosure_required=True,
                compatible_licenses={"AGPL-3.0"},
                incompatible_licenses={"Apache-2.0", "MIT", "BSD-3-Clause"}
            ),
            "AGPL-3.0": LicenseInfo(
                license_id="AGPL-3.0",
                license_name="GNU Affero General Public License v3.0",
                license_text=None,
                risk_level=LicenseRisk.CRITICAL,
                commercial_use=True,
                copyleft=True,
                patent_grant=True,
                trademark_use=False,
                attribution_required=True,
                source_disclosure_required=True,
                compatible_licenses={"GPL-3.0"},
                incompatible_licenses={"Apache-2.0", "MIT", "BSD-3-Clause"}
            ),
            "BSD-3-Clause": LicenseInfo(
                license_id="BSD-3-Clause",
                license_name="BSD 3-Clause License",
                license_text=None,
                risk_level=LicenseRisk.LOW,
                commercial_use=True,
                copyleft=False,
                patent_grant=False,
                trademark_use=False,
                attribution_required=True,
                source_disclosure_required=False,
                compatible_licenses={"MIT", "Apache-2.0", "ISC"},
                incompatible_licenses={"GPL-3.0", "AGPL-3.0"}
            ),
            "ISC": LicenseInfo(
                license_id="ISC",
                license_name="ISC License",
                license_text=None,
                risk_level=LicenseRisk.LOW,
                commercial_use=True,
                copyleft=False,
                patent_grant=False,
                trademark_use=False,
                attribution_required=True,
                source_disclosure_required=False,
                compatible_licenses={"MIT", "Apache-2.0", "BSD-3-Clause"},
                incompatible_licenses={"GPL-3.0", "AGPL-3.0"}
            ),
            "Proprietary": LicenseInfo(
                license_id="Proprietary",
                license_name="Proprietary License",
                license_text=None,
                risk_level=LicenseRisk.CRITICAL,
                commercial_use=False,
                copyleft=False,
                patent_grant=False,
                trademark_use=False,
                attribution_required=False,
                source_disclosure_required=False,
                compatible_licenses=set(),
                incompatible_licenses={"GPL-3.0", "AGPL-3.0", "MIT", "Apache-2.0"}
            )
        }
    
    def get_license_info(self, license_id: str) -> Optional[LicenseInfo]:
        """Get license information by ID."""
        return self.licenses.get(license_id)
    
    def classify_license(self, license_text: str) -> Optional[LicenseInfo]:
        """Classify a license based on its text."""
        # Simple keyword-based classification
        license_text_lower = license_text.lower()
        
        if "mit license" in license_text_lower or "mit" in license_text_lower:
            return self.licenses.get("MIT")
        elif "apache license" in license_text_lower or "apache-2.0" in license_text_lower:
            return self.licenses.get("Apache-2.0")
        elif "gnu general public license" in license_text_lower and "version 3" in license_text_lower:
            return self.licenses.get("GPL-3.0")
        elif "gnu affero general public license" in license_text_lower:
            return self.licenses.get("AGPL-3.0")
        elif "bsd" in license_text_lower and "3-clause" in license_text_lower:
            return self.licenses.get("BSD-3-Clause")
        elif "isc license" in license_text_lower:
            return self.licenses.get("ISC")
        else:
            return LicenseInfo(
                license_id="UNKNOWN",
                license_name="Unknown License",
                license_text=license_text[:500],  # First 500 chars
                risk_level=LicenseRisk.UNKNOWN,
                commercial_use=False,
                copyleft=False,
                patent_grant=False,
                trademark_use=False,
                attribution_required=True,
                source_disclosure_required=False,
                compatible_licenses=set(),
                incompatible_licenses=set()
            )


class SBOMValidator:
    """Software Bill of Materials validator and license compliance checker."""
    
    def __init__(self, storage_dir: Path = None):
        self.storage_dir = storage_dir or Path("./data/sbom")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.license_db = LicenseDatabase()
        self.components: Dict[str, SoftwareComponent] = {}
        self.validation_reports: List[SBOMReport] = []
        
        # Configuration
        self.allowed_risk_levels = {LicenseRisk.LOW, LicenseRisk.MEDIUM}
        self.require_attribution = True
        self.allow_copyleft = False
        self.scan_interval_hours = 24
        
        # Auto-scanning
        self.scanning_enabled = True
        self.scan_thread = None
        
        self._load_components_db()
        self._start_periodic_scanning()
        
        print(f"📋 SBOM Validator initialized")
        print(f"   Storage: {self.storage_dir}")
        print(f"   Known licenses: {len(self.license_db.licenses)}")
        print(f"   Components tracked: {len(self.components)}")
    
    def _load_components_db(self):
        """Load component database from storage."""
        components_file = self.storage_dir / "components.json"
        if components_file.exists():
            try:
                with open(components_file) as f:
                    data = json.load(f)
                    for comp_id, comp_data in data.items():
                        component = SoftwareComponent(**comp_data)
                        component.component_type = ComponentType(comp_data["component_type"])
                        
                        # Convert license info
                        license_info = []
                        for lic_data in comp_data["license_info"]:
                            lic_info = LicenseInfo(**lic_data)
                            lic_info.risk_level = LicenseRisk(lic_data["risk_level"])
                            lic_info.compatible_licenses = set(lic_data["compatible_licenses"])
                            lic_info.incompatible_licenses = set(lic_data["incompatible_licenses"])
                            license_info.append(lic_info)
                        
                        component.license_info = license_info
                        self.components[comp_id] = component
                        
            except Exception as e:
                print(f"Warning: Failed to load components database: {e}")
    
    def _save_components_db(self):
        """Save component database to storage."""
        components_file = self.storage_dir / "components.json"
        try:
            data = {}
            for comp_id, component in self.components.items():
                comp_data = asdict(component)
                comp_data["component_type"] = component.component_type.value
                
                # Convert license info
                license_info_data = []
                for lic_info in component.license_info:
                    lic_data = asdict(lic_info)
                    lic_data["risk_level"] = lic_info.risk_level.value
                    lic_data["compatible_licenses"] = list(lic_info.compatible_licenses)
                    lic_data["incompatible_licenses"] = list(lic_info.incompatible_licenses)
                    license_info_data.append(lic_data)
                
                comp_data["license_info"] = license_info_data
                data[comp_id] = comp_data
            
            # Atomic write
            temp_file = components_file.with_suffix(".tmp")
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            temp_file.replace(components_file)
            
        except Exception as e:
            print(f"Warning: Failed to save components database: {e}")
    
    def scan_python_environment(self) -> List[SoftwareComponent]:
        """Scan current Python environment for packages."""
        components = []
        
        try:
            # Use pip list to get installed packages
            result = subprocess.run(
                ["pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                check=True
            )
            
            packages = json.loads(result.stdout)
            
            for package in packages:
                name = package["name"]
                version = package["version"]
                component_id = f"python:{name}:{version}"
                
                # Try to get license information
                license_info = self._get_package_license_info(name)
                
                component = SoftwareComponent(
                    component_id=component_id,
                    name=name,
                    version=version,
                    component_type=ComponentType.PYTHON_PACKAGE,
                    supplier=None,
                    download_location=f"https://pypi.org/project/{name}/{version}/",
                    file_hash=None,
                    license_info=license_info,
                    dependencies=[],
                    vulnerabilities=[],
                    last_updated=time.time(),
                    verification_status="unverified",
                    metadata={"source": "pip_list"}
                )
                
                components.append(component)
                self.components[component_id] = component
                
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to scan Python environment: {e}")
        except Exception as e:
            print(f"Warning: Error during Python environment scan: {e}")
        
        return components
    
    def _get_package_license_info(self, package_name: str) -> List[LicenseInfo]:
        """Get license information for a Python package."""
        license_info = []
        
        try:
            # Use pip show to get package metadata
            result = subprocess.run(
                ["pip", "show", package_name],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse license from output
            for line in result.stdout.split('\n'):
                if line.startswith("License:"):
                    license_text = line.split(":", 1)[1].strip()
                    if license_text and license_text != "UNKNOWN":
                        # Try to match known license
                        known_license = self.license_db.get_license_info(license_text)
                        if known_license:
                            license_info.append(known_license)
                        else:
                            # Classify unknown license
                            classified = self.license_db.classify_license(license_text)
                            if classified:
                                license_info.append(classified)
                    break
            
        except subprocess.CalledProcessError:
            # Package not found or pip show failed
            pass
        except Exception as e:
            print(f"Warning: Failed to get license info for {package_name}: {e}")
        
        # If no license found, mark as unknown
        if not license_info:
            license_info.append(LicenseInfo(
                license_id="UNKNOWN",
                license_name="Unknown License",
                license_text=None,
                risk_level=LicenseRisk.UNKNOWN,
                commercial_use=False,
                copyleft=False,
                patent_grant=False,
                trademark_use=False,
                attribution_required=True,
                source_disclosure_required=False,
                compatible_licenses=set(),
                incompatible_licenses=set()
            ))
        
        return license_info
    
    def scan_ai_models(self, models_dir: Path = None) -> List[SoftwareComponent]:
        """Scan AI models for license compliance."""
        models_dir = models_dir or Path("./models")
        components = []
        
        if not models_dir.exists():
            return components
        
        for model_path in models_dir.iterdir():
            if model_path.is_dir():
                component_id = f"ai_model:{model_path.name}"
                
                # Look for license files
                license_files = list(model_path.glob("LICENSE*")) + list(model_path.glob("license*"))
                license_info = []
                
                for license_file in license_files:
                    try:
                        license_text = license_file.read_text(encoding='utf-8')
                        classified = self.license_db.classify_license(license_text)
                        if classified:
                            license_info.append(classified)
                    except Exception as e:
                        print(f"Warning: Failed to read license file {license_file}: {e}")
                
                # Calculate hash of model directory
                model_hash = self._calculate_directory_hash(model_path)
                
                component = SoftwareComponent(
                    component_id=component_id,
                    name=model_path.name,
                    version="unknown",
                    component_type=ComponentType.AI_MODEL,
                    supplier=None,
                    download_location=None,
                    file_hash=model_hash,
                    license_info=license_info or [LicenseInfo(
                        license_id="UNKNOWN",
                        license_name="Unknown License",
                        license_text=None,
                        risk_level=LicenseRisk.UNKNOWN,
                        commercial_use=False,
                        copyleft=False,
                        patent_grant=False,
                        trademark_use=False,
                        attribution_required=True,
                        source_disclosure_required=False,
                        compatible_licenses=set(),
                        incompatible_licenses=set()
                    )],
                    dependencies=[],
                    vulnerabilities=[],
                    last_updated=time.time(),
                    verification_status="unverified",
                    metadata={"model_path": str(model_path)}
                )
                
                components.append(component)
                self.components[component_id] = component
        
        return components
    
    def _calculate_directory_hash(self, directory: Path) -> str:
        """Calculate hash of directory contents."""
        hash_md5 = hashlib.md5()
        
        for file_path in sorted(directory.rglob("*")):
            if file_path.is_file():
                relative_path = file_path.relative_to(directory)
                hash_md5.update(str(relative_path).encode())
                
                try:
                    with open(file_path, 'rb') as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hash_md5.update(chunk)
                except Exception:
                    # Skip files that can't be read
                    pass
        
        return hash_md5.hexdigest()
    
    def validate_license_compliance(self) -> SBOMReport:
        """Validate license compliance for all components."""
        report_id = f"sbom_report_{int(time.time())}"
        timestamp = time.time()
        
        high_risk_licenses = 0
        license_conflicts = []
        missing_licenses = []
        recommendations = []
        
        # Check each component
        for component in self.components.values():
            # Check for missing licenses
            if not component.license_info:
                missing_licenses.append(component.component_id)
                continue
            
            # Check license risk levels
            for license_info in component.license_info:
                if license_info.risk_level not in self.allowed_risk_levels:
                    high_risk_licenses += 1
                    
                    if license_info.risk_level == LicenseRisk.CRITICAL:
                        recommendations.append(
                            f"CRITICAL: Remove or replace component {component.name} "
                            f"with {license_info.license_name} license"
                        )
                    elif license_info.risk_level == LicenseRisk.HIGH:
                        recommendations.append(
                            f"HIGH RISK: Review component {component.name} "
                            f"with {license_info.license_name} license"
                        )
                
                # Check copyleft restrictions
                if not self.allow_copyleft and license_info.copyleft:
                    license_conflicts.append({
                        "component": component.name,
                        "license": license_info.license_name,
                        "conflict": "Copyleft license not allowed in this project",
                        "severity": "high"
                    })
        
        # Determine compliance status
        compliance_status = "compliant"
        if high_risk_licenses > 0 or license_conflicts or missing_licenses:
            if any(c["severity"] == "high" for c in license_conflicts) or high_risk_licenses > 0:
                compliance_status = "non_compliant"
            else:
                compliance_status = "needs_review"
        
        # Generate additional recommendations
        if missing_licenses:
            recommendations.append(
                f"Add license information for {len(missing_licenses)} components"
            )
        
        if license_conflicts:
            recommendations.append(
                f"Resolve {len(license_conflicts)} license conflicts"
            )
        
        report = SBOMReport(
            report_id=report_id,
            timestamp=timestamp,
            total_components=len(self.components),
            verified_components=len([c for c in self.components.values() if c.verification_status == "verified"]),
            high_risk_licenses=high_risk_licenses,
            license_conflicts=license_conflicts,
            missing_licenses=missing_licenses,
            security_vulnerabilities=0,  # TODO: Integrate with vulnerability scanning
            compliance_status=compliance_status,
            recommendations=recommendations,
            components=list(self.components.values())
        )
        
        self.validation_reports.append(report)
        self._save_validation_report(report)
        
        print(f"📋 SBOM validation completed")
        print(f"   Total components: {report.total_components}")
        print(f"   Compliance status: {report.compliance_status}")
        print(f"   High-risk licenses: {report.high_risk_licenses}")
        print(f"   License conflicts: {len(report.license_conflicts)}")
        
        # Record security event
        from .monitoring import record_security_event
        record_security_event(
            "sbom_validation_completed",
            "sbom_validator",
            {
                "report_id": report_id,
                "compliance_status": compliance_status,
                "total_components": report.total_components,
                "high_risk_licenses": high_risk_licenses
            }
        )
        
        return report
    
    def _save_validation_report(self, report: SBOMReport):
        """Save validation report to storage."""
        report_file = self.storage_dir / f"{report.report_id}.json"
        try:
            # Convert to JSON-serializable format
            report_data = asdict(report)
            
            # Convert components
            components_data = []
            for component in report.components:
                comp_data = asdict(component)
                comp_data["component_type"] = component.component_type.value
                
                # Convert license info
                license_info_data = []
                for lic_info in component.license_info:
                    lic_data = asdict(lic_info)
                    lic_data["risk_level"] = lic_info.risk_level.value
                    lic_data["compatible_licenses"] = list(lic_info.compatible_licenses)
                    lic_data["incompatible_licenses"] = list(lic_info.incompatible_licenses)
                    license_info_data.append(lic_data)
                
                comp_data["license_info"] = license_info_data
                components_data.append(comp_data)
            
            report_data["components"] = components_data
            
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Failed to save validation report: {e}")
    
    def _start_periodic_scanning(self):
        """Start periodic SBOM scanning."""
        if not self.scan_thread or not self.scan_thread.is_alive():
            self.scan_thread = threading.Thread(target=self._scanning_loop, daemon=True)
            self.scan_thread.start()
            print("🔍 Periodic SBOM scanning started")
    
    def _scanning_loop(self):
        """Periodic scanning loop."""
        while self.scanning_enabled:
            try:
                # Perform full scan
                print("🔍 Starting periodic SBOM scan...")
                
                # Scan Python environment
                python_components = self.scan_python_environment()
                print(f"   Found {len(python_components)} Python packages")
                
                # Scan AI models
                model_components = self.scan_ai_models()
                print(f"   Found {len(model_components)} AI models")
                
                # Save updated components database
                self._save_components_db()
                
                # Validate compliance
                self.validate_license_compliance()
                
                # Sleep until next scan
                time.sleep(self.scan_interval_hours * 3600)
                
            except Exception as e:
                print(f"Warning: SBOM scanning error: {e}")
                time.sleep(1800)  # 30 minutes on error
    
    def get_sbom_status(self) -> Dict:
        """Get SBOM validation system status."""
        latest_report = self.validation_reports[-1] if self.validation_reports else None
        
        license_risk_counts = {risk.value: 0 for risk in LicenseRisk}
        component_type_counts = {ctype.value: 0 for ctype in ComponentType}
        
        for component in self.components.values():
            component_type_counts[component.component_type.value] += 1
            for license_info in component.license_info:
                license_risk_counts[license_info.risk_level.value] += 1
        
        return {
            "total_components": len(self.components),
            "component_types": component_type_counts,
            "license_risk_distribution": license_risk_counts,
            "scanning_enabled": self.scanning_enabled,
            "scan_interval_hours": self.scan_interval_hours,
            "last_scan": max([c.last_updated for c in self.components.values()]) if self.components else None,
            "latest_report": {
                "report_id": latest_report.report_id if latest_report else None,
                "compliance_status": latest_report.compliance_status if latest_report else "no_reports",
                "timestamp": latest_report.timestamp if latest_report else None
            } if latest_report else None,
            "configuration": {
                "allowed_risk_levels": [risk.value for risk in self.allowed_risk_levels],
                "allow_copyleft": self.allow_copyleft,
                "require_attribution": self.require_attribution
            }
        }


# Global instance
sbom_validator = SBOMValidator()


# Convenience functions
def scan_software_components() -> Dict[str, int]:
    """Scan all software components."""
    python_components = sbom_validator.scan_python_environment()
    model_components = sbom_validator.scan_ai_models()
    
    return {
        "python_packages": len(python_components),
        "ai_models": len(model_components),
        "total_components": len(python_components) + len(model_components)
    }


def validate_license_compliance() -> SBOMReport:
    """Validate license compliance."""
    return sbom_validator.validate_license_compliance()


def get_sbom_status() -> Dict:
    """Get SBOM validation status."""
    return sbom_validator.get_sbom_status()


def get_latest_sbom_report() -> Optional[SBOMReport]:
    """Get the latest SBOM validation report."""
    return sbom_validator.validation_reports[-1] if sbom_validator.validation_reports else None
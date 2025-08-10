"""
Node Authentication System for Blyan
Prevents unauthorized nodes from taking over as main node
"""

import os
import json
import hashlib
import hmac
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import secrets

class NodeAuthenticator:
    """Manages node authentication and authorization"""
    
    def __init__(self, config_path: str = "./config/node_auth.json"):
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config = self._load_config()
        self.main_node_secret = os.environ.get('BLYAN_MAIN_NODE_SECRET')
        
    def _load_config(self) -> Dict:
        """Load node authentication configuration"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        default_config = {
            "main_node": {
                "id": None,
                "host": None,
                "registered_at": None,
                "public_key": None
            },
            "authorized_nodes": [],
            "node_roles": {
                "main": [],      # Only one main node allowed
                "validator": [], # Can validate but not lead
                "worker": []     # Can contribute compute only
            },
            "settings": {
                "require_auth": True,
                "allow_node_registration": False,  # Set to True only during setup
                "max_validators": 10,
                "main_node_election_interval_hours": 24
            }
        }
        
        self._save_config(default_config)
        return default_config
    
    def _save_config(self, config: Dict):
        """Save configuration to file"""
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def register_main_node(self, node_id: str, host: str, secret: str) -> Dict:
        """
        Register the main node (only once, requires secret)
        This should be done during initial setup on your Digital Ocean server
        """
        if not self.main_node_secret:
            return {"error": "BLYAN_MAIN_NODE_SECRET environment variable not set"}
        
        if secret != self.main_node_secret:
            return {"error": "Invalid main node secret"}
        
        if self.config["main_node"]["id"]:
            return {"error": "Main node already registered"}
        
        # Generate node authentication token
        auth_token = secrets.token_hex(32)
        
        self.config["main_node"] = {
            "id": node_id,
            "host": host,
            "registered_at": datetime.utcnow().isoformat(),
            "auth_token_hash": hashlib.sha256(auth_token.encode()).hexdigest()
        }
        
        self.config["node_roles"]["main"] = [node_id]
        self._save_config(self.config)
        
        return {
            "success": True,
            "auth_token": auth_token,  # Save this securely!
            "message": "Main node registered. Save the auth_token securely!"
        }
    
    def verify_main_node(self, node_id: str, auth_token: str) -> bool:
        """Verify if a node is the legitimate main node"""
        if not self.config["main_node"]["id"]:
            return False
        
        if self.config["main_node"]["id"] != node_id:
            return False
        
        # Verify auth token
        token_hash = hashlib.sha256(auth_token.encode()).hexdigest()
        return token_hash == self.config["main_node"].get("auth_token_hash")
    
    def authorize_validator(self, node_id: str, main_node_token: str) -> Dict:
        """Authorize a node as validator (requires main node token)"""
        # First verify this is coming from main node
        main_id = self.config["main_node"]["id"]
        if not main_id or not self.verify_main_node(main_id, main_node_token):
            return {"error": "Unauthorized: Invalid main node credentials"}
        
        validators = self.config["node_roles"]["validator"]
        max_validators = self.config["settings"]["max_validators"]
        
        if len(validators) >= max_validators:
            return {"error": f"Maximum validators ({max_validators}) reached"}
        
        if node_id not in validators:
            validators.append(node_id)
            self.config["node_roles"]["validator"] = validators
            self._save_config(self.config)
        
        return {"success": True, "role": "validator"}
    
    def get_node_role(self, node_id: str) -> Optional[str]:
        """Get the role of a node"""
        for role, nodes in self.config["node_roles"].items():
            if node_id in nodes:
                return role
        return "worker"  # Default role
    
    def can_write_blocks(self, node_id: str, auth_token: str = None) -> bool:
        """Check if a node can write blocks to the chain"""
        role = self.get_node_role(node_id)
        
        # Only main node can write blocks
        if role == "main":
            return self.verify_main_node(node_id, auth_token)
        
        # Validators can propose but not write directly
        return False
    
    def get_main_node_info(self) -> Dict:
        """Get public information about main node"""
        if not self.config["main_node"]["id"]:
            return {"error": "No main node registered"}
        
        return {
            "id": self.config["main_node"]["id"],
            "host": self.config["main_node"]["host"],
            "registered_at": self.config["main_node"]["registered_at"]
        }
    
    def prevent_hostile_takeover(self, requesting_node: str, claimed_role: str) -> bool:
        """
        Prevent hostile takeover attempts
        Returns True if takeover attempt detected
        """
        # Check if someone is trying to claim main node role
        if claimed_role == "main":
            current_main = self.config["main_node"]["id"]
            if current_main and requesting_node != current_main:
                # Log the attempt
                self._log_security_event({
                    "type": "hostile_takeover_attempt",
                    "node": requesting_node,
                    "claimed_role": claimed_role,
                    "timestamp": datetime.utcnow().isoformat()
                })
                return True
        
        return False
    
    def _log_security_event(self, event: Dict):
        """Log security events for monitoring"""
        log_path = Path("./logs/security_events.json")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        events = []
        if log_path.exists():
            with open(log_path, 'r') as f:
                events = json.load(f)
        
        events.append(event)
        
        # Keep only last 1000 events
        events = events[-1000:]
        
        with open(log_path, 'w') as f:
            json.dump(events, f, indent=2)


# API endpoint protection decorator
def require_main_node_auth(func):
    """Decorator to protect endpoints that only main node should access"""
    def wrapper(request, *args, **kwargs):
        auth_header = request.headers.get('X-Node-Auth-Token')
        node_id = request.headers.get('X-Node-ID')
        
        if not auth_header or not node_id:
            return {"error": "Missing authentication headers"}, 401
        
        authenticator = NodeAuthenticator()
        if not authenticator.verify_main_node(node_id, auth_header):
            # Log the attempt
            authenticator._log_security_event({
                "type": "unauthorized_api_access",
                "endpoint": func.__name__,
                "node_id": node_id,
                "timestamp": datetime.utcnow().isoformat()
            })
            return {"error": "Unauthorized"}, 403
        
        return func(request, *args, **kwargs)
    
    return wrapper
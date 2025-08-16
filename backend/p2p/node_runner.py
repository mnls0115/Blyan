#!/usr/bin/env python3
"""
Production P2P Node Runner for Blyan Network
Integrates all P2P components into a runnable node
"""
import asyncio
import argparse
import signal
import sys
import os
import yaml
import json
from pathlib import Path
from typing import Optional, Dict, Any
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.crypto.signing import KeyPair
from backend.crypto.hashing import compute_hash
from backend.consensus.block_validator import BlockValidator
from backend.consensus.chain_manager import ChainManager
from backend.network.serialization import NetworkSerializer
from backend.network.security import SecurityManager
from backend.p2p.dht_discovery import DHT
from backend.p2p.chain_sync import ChainSyncProtocol
from backend.ops.metrics import metrics_collector
from backend.ops.logging import get_logger
from backend.core.chain import Chain


class P2PNode:
    """Production P2P node implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize P2P node
        
        Args:
            config: Node configuration
        """
        self.config = config
        self.node_id = config['node']['id'] or self._generate_node_id()
        self.role = config['node']['role']
        self.data_dir = Path(config['node']['data_dir'])
        
        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = get_logger('node', config['logging']['level'])
        self.logger.info(f"Initializing P2P node", node_id=self.node_id, role=self.role)
        
        # Initialize components
        self.keypair = self._load_or_create_keypair()
        self.address = self.keypair.get_address()
        
        # Core components
        self.chain_manager = ChainManager()
        self.validator = BlockValidator(self.chain_manager)
        self.serializer = NetworkSerializer()
        self.security = SecurityManager(config.get('security', {}))
        
        # Network components
        self.dht: Optional[DHT] = None
        self.sync_protocol: Optional[ChainSyncProtocol] = None
        
        # Blockchain instances
        self.chains: Dict[str, Chain] = {}
        self._init_chains()
        
        # State
        self.running = False
        self.peers: Dict[str, Dict] = {}
        self.sync_task: Optional[asyncio.Task] = None
        self.metrics_task: Optional[asyncio.Task] = None
        
    def _generate_node_id(self) -> str:
        """Generate unique node ID"""
        import uuid
        return f"node_{uuid.uuid4().hex[:8]}"
    
    def _load_or_create_keypair(self) -> KeyPair:
        """Load existing keypair or create new one"""
        key_file = self.data_dir / "node.key"
        
        if key_file.exists():
            self.logger.info("Loading existing keypair")
            with open(key_file, 'rb') as f:
                private_key = f.read()
            return KeyPair(private_key)
        else:
            self.logger.info("Generating new keypair")
            keypair = KeyPair()
            with open(key_file, 'wb') as f:
                f.write(keypair.signing_key.encode())
            return keypair
    
    def _init_chains(self):
        """Initialize blockchain instances"""
        # Initialize meta-chain (A) and parameter-chain (B)
        for chain_id in ['A', 'B']:
            chain_dir = self.data_dir / f"chain_{chain_id}"
            self.chains[chain_id] = Chain(chain_dir, chain_id)
            
            # Load existing blocks into chain manager
            blocks = self.chains[chain_id].get_all_blocks()
            for block in blocks:
                # Convert to dict format expected by chain manager
                block_dict = {
                    'hash': block['hash'],
                    'height': block.get('index', 0),
                    'parent_hash': block.get('previous_hash', ''),
                    'timestamp': block.get('timestamp', time.time()),
                    'transactions': [],
                    'signature': block.get('signature', ''),
                    'public_key': block.get('public_key', ''),
                }
                self.chain_manager.add_block(block_dict)
        
        self.logger.info(f"Initialized chains", 
                        chain_a_height=len(self.chains['A'].get_all_blocks()),
                        chain_b_height=len(self.chains['B'].get_all_blocks()))
    
    async def start(self):
        """Start the P2P node"""
        self.logger.info("Starting P2P node", address=self.address)
        self.running = True
        
        try:
            # Initialize DHT if enabled
            if self.config['bootstrap']['dht']['enabled']:
                await self._init_dht()
            
            # Initialize sync protocol
            await self._init_sync()
            
            # Start background tasks
            self.sync_task = asyncio.create_task(self._sync_loop())
            self.metrics_task = asyncio.create_task(self._metrics_loop())
            
            # Start API server if enabled
            if self.config['api']['enabled']:
                await self._start_api_server()
            
            self.logger.info("P2P node started successfully")
            
            # Keep running
            while self.running:
                await asyncio.sleep(1)
                
        except Exception as e:
            self.logger.error(f"Error starting node: {e}")
            raise
    
    async def _init_dht(self):
        """Initialize DHT discovery"""
        bootstrap_nodes = self.config['bootstrap']['dht']['bootstrap_nodes']
        
        self.dht = DHT(
            node_id=self.node_id.encode(),
            bootstrap_nodes=bootstrap_nodes,
            storage_dir=self.data_dir / "dht"
        )
        
        self.logger.info("Bootstrapping DHT", bootstrap_nodes=bootstrap_nodes)
        await self.dht.bootstrap()
        
        # Announce ourselves
        await self.dht.announce_peer(
            self.node_id.encode(),
            {
                'address': self.address,
                'role': self.role,
                'chains': list(self.chains.keys()),
                'version': self.config['p2p']['protocol_version'],
            }
        )
    
    async def _init_sync(self):
        """Initialize chain sync protocol"""
        # For now, use bootstrap nodes directly
        bootstrap_nodes = self.config['bootstrap']['nodes']
        
        # Create sync protocol instance
        # Note: ChainSyncProtocol would need to be updated to use new components
        self.logger.info("Initializing chain sync", bootstrap_nodes=bootstrap_nodes)
        
        # Connect to bootstrap nodes
        for node_addr in bootstrap_nodes:
            try:
                # In production, this would establish actual network connections
                self.logger.info(f"Connecting to bootstrap node", node=node_addr)
                # await self.connect_to_peer(node_addr)
            except Exception as e:
                self.logger.warning(f"Failed to connect to bootstrap node", 
                                  node=node_addr, error=str(e))
    
    async def _sync_loop(self):
        """Background task for chain synchronization"""
        while self.running:
            try:
                # Update metrics
                for chain_id, chain in self.chains.items():
                    blocks = chain.get_all_blocks()
                    if blocks:
                        height = max(b.get('index', 0) for b in blocks)
                        metrics_collector.update_chain_height(height, chain_id)
                
                # Check for new blocks from peers
                # In production, this would sync with connected peers
                
                await asyncio.sleep(10)  # Sync every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Sync error: {e}")
                await asyncio.sleep(5)
    
    async def _metrics_loop(self):
        """Background task for metrics collection"""
        while self.running:
            try:
                # Update peer count
                full_peers = sum(1 for p in self.peers.values() if p.get('role') == 'FULL')
                light_peers = sum(1 for p in self.peers.values() if p.get('role') == 'LIGHT')
                seed_peers = sum(1 for p in self.peers.values() if p.get('role') == 'SEED')
                
                metrics_collector.update_peer_count(
                    full=full_peers,
                    light=light_peers,
                    seed=seed_peers
                )
                
                # Update sync progress
                # In production, calculate actual sync progress
                metrics_collector.update_sync_progress(100, 100)
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Metrics error: {e}")
                await asyncio.sleep(30)
    
    async def _start_api_server(self):
        """Start HTTP API server"""
        from aiohttp import web
        
        app = web.Application()
        
        # Health endpoint
        async def health(request):
            return web.json_response({
                'status': 'healthy',
                'node_id': self.node_id,
                'address': self.address,
                'role': self.role,
                'uptime': time.time() - self.start_time if hasattr(self, 'start_time') else 0
            })
        
        # Peers endpoint
        async def peers(request):
            return web.json_response({
                'count': len(self.peers),
                'peers': list(self.peers.values())
            })
        
        # Chain info endpoint
        async def chain_info(request):
            chain_id = request.match_info.get('chain_id', 'A')
            if chain_id not in self.chains:
                return web.json_response({'error': 'Invalid chain ID'}, status=400)
            
            blocks = self.chains[chain_id].get_all_blocks()
            return web.json_response({
                'chain_id': chain_id,
                'height': len(blocks),
                'latest_block': blocks[-1] if blocks else None
            })
        
        # Metrics endpoint
        async def metrics(request):
            metrics_bytes = metrics_collector.get_metrics()
            return web.Response(
                body=metrics_bytes,
                content_type='text/plain; version=0.0.4'
            )
        
        # Sync status endpoint
        async def sync_status(request):
            return web.json_response({
                'syncing': False,  # Would check actual sync state
                'current_block': self.chain_manager.get_latest_block(),
                'highest_block': self.chain_manager.get_latest_block(),
                'progress': 1.0
            })
        
        # Register routes
        app.router.add_get('/health', health)
        app.router.add_get('/api/peers', peers)
        app.router.add_get('/api/chain/{chain_id}', chain_info)
        app.router.add_get('/api/sync', sync_status)
        app.router.add_get('/metrics', metrics)
        
        # Start server
        host, port = self.config['api']['listen_addr'].split(':')
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, int(port))
        await site.start()
        
        self.logger.info(f"API server started", host=host, port=port)
    
    async def stop(self):
        """Stop the P2P node"""
        self.logger.info("Stopping P2P node")
        self.running = False
        
        # Cancel background tasks
        if self.sync_task:
            self.sync_task.cancel()
        if self.metrics_task:
            self.metrics_task.cancel()
        
        # Close DHT
        if self.dht:
            await self.dht.close()
        
        self.logger.info("P2P node stopped")
    
    def handle_signal(self, sig, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {sig}, shutting down")
        asyncio.create_task(self.stop())


def load_config(config_file: Path) -> Dict[str, Any]:
    """Load configuration from file"""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Expand environment variables
    def expand_env(value):
        if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            env_var = value[2:-1]
            return os.getenv(env_var, value)
        return value
    
    def expand_dict(d):
        for key, value in d.items():
            if isinstance(value, dict):
                expand_dict(value)
            elif isinstance(value, list):
                d[key] = [expand_env(v) for v in value]
            else:
                d[key] = expand_env(value)
    
    expand_dict(config)
    
    # Apply environment-specific overrides
    env = os.getenv('NODE_ENV', 'development')
    if env in config.get('environments', {}):
        env_config = config['environments'][env]
        # Deep merge environment config
        import copy
        
        def deep_merge(base, override):
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
        
        deep_merge(config, env_config)
    
    return config


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Blyan Network P2P Node')
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('config/node.yaml'),
        help='Configuration file path'
    )
    parser.add_argument(
        '--node-id',
        type=str,
        help='Override node ID'
    )
    parser.add_argument(
        '--role',
        choices=['SEED', 'FULL', 'LIGHT'],
        help='Override node role'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        help='Override data directory'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if not args.config.exists():
        print(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    config = load_config(args.config)
    
    # Apply command-line overrides
    if args.node_id:
        config['node']['id'] = args.node_id
    if args.role:
        config['node']['role'] = args.role
    if args.data_dir:
        config['node']['data_dir'] = str(args.data_dir)
    
    # Create and start node
    node = P2PNode(config)
    node.start_time = time.time()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, node.handle_signal)
    signal.signal(signal.SIGTERM, node.handle_signal)
    
    try:
        await node.start()
    except KeyboardInterrupt:
        pass
    finally:
        await node.stop()


if __name__ == '__main__':
    asyncio.run(main())
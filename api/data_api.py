"""
API endpoints for the Big Data Pool system.
Handles data upload, validation, distribution, and transparency.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional
from aiohttp import web
import aiofiles

from backend.data.pool import DataPool, DataQuality
from backend.data.ingest import DataIngester
from backend.data.scheduler import DataScheduler
from backend.data.harvest import DataHarvester

logger = logging.getLogger(__name__)


class DataPoolAPI:
    """API handler for data pool operations."""
    
    def __init__(self, chains: Dict, storage_dir: Path = None):
        """
        Initialize Data Pool API.
        
        Args:
            chains: Blockchain instances
            storage_dir: Storage directory
        """
        if storage_dir is None:
            storage_dir = Path("./data/pool")
        
        # Initialize components
        self.data_pool = DataPool(storage_dir, chains)
        self.ingester = DataIngester(self.data_pool)
        self.scheduler = DataScheduler(self.data_pool, storage_dir / "scheduler")
        self.harvester = DataHarvester(self.data_pool)
        
        logger.info("Data Pool API initialized")
    
    # ==================== Upload Endpoints ====================
    
    async def upload_data(self, request: web.Request) -> web.Response:
        """
        Upload data to the pool.
        
        POST /data/upload
        Multipart form with file and metadata
        """
        try:
            reader = await request.multipart()
            
            file_data = None
            metadata = {
                'license': 'unknown',
                'tags': [],
                'uploader': 'anonymous'
            }
            
            # Parse multipart data
            async for part in reader:
                if part.name == 'file':
                    file_data = await part.read()
                    filename = part.filename
                elif part.name == 'metadata':
                    metadata_str = await part.text()
                    metadata.update(json.loads(metadata_str))
            
            if not file_data:
                return web.json_response({
                    'error': 'No file provided'
                }, status=400)
            
            # Save temporary file
            temp_dir = Path("/tmp/data_upload")
            temp_dir.mkdir(exist_ok=True)
            temp_file = temp_dir / filename
            
            async with aiofiles.open(temp_file, 'wb') as f:
                await f.write(file_data)
            
            # Ingest file
            result = self.ingester.ingest_file(
                file_path=temp_file,
                license=metadata.get('license', 'unknown'),
                uploader=metadata.get('uploader', 'anonymous'),
                tags=metadata.get('tags', [])
            )
            
            # Clean up temp file
            temp_file.unlink()
            
            return web.json_response(result)
            
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return web.json_response({
                'error': str(e)
            }, status=500)
    
    async def upload_text(self, request: web.Request) -> web.Response:
        """
        Upload raw text to the pool.
        
        POST /data/upload_text
        {
            "text": "...",
            "source_url": "...",
            "license": "mit",
            "tags": ["tag1", "tag2"]
        }
        """
        try:
            data = await request.json()
            
            text = data.get('text')
            if not text:
                return web.json_response({
                    'error': 'No text provided'
                }, status=400)
            
            # Ingest text
            cids = self.data_pool.ingest_text(
                text=text,
                source_url=data.get('source_url', 'user_upload'),
                license=data.get('license', 'unknown'),
                uploader=data.get('uploader', 'anonymous'),
                tags=data.get('tags', [])
            )
            
            return web.json_response({
                'status': 'success',
                'chunks_created': len(cids),
                'cids': cids
            })
            
        except Exception as e:
            logger.error(f"Text upload failed: {e}")
            return web.json_response({
                'error': str(e)
            }, status=500)
    
    # ==================== Distribution Endpoints ====================
    
    async def get_training_data(self, request: web.Request) -> web.Response:
        """
        Get training data for an expert.
        
        GET /data/training/{expert_id}?tokens=100000&tags=medical,research
        """
        try:
            expert_id = request.match_info['expert_id']
            
            # Parse parameters
            target_tokens = int(request.query.get('tokens', 100000))
            tags = request.query.get('tags', '').split(',') if request.query.get('tags') else None
            min_quality = float(request.query.get('min_quality', 0.7))
            
            # Create epoch manifest
            manifest = self.scheduler.create_epoch_manifest(
                expert_id=expert_id,
                target_tokens=target_tokens,
                required_tags=tags,
                min_quality=min_quality
            )
            
            # Get chunk metadata
            chunks = []
            for cid in manifest.cids[:100]:  # Limit response size
                if cid in self.data_pool.chunk_index:
                    chunk = self.data_pool.chunk_index[cid]
                    chunks.append({
                        'cid': chunk.cid,
                        'size': chunk.size_bytes,
                        'tokens': chunk.token_count,
                        'quality': chunk.quality_score,
                        'tags': chunk.tags
                    })
            
            return web.json_response({
                'epoch_id': manifest.epoch_id,
                'expert_id': manifest.expert_id,
                'total_chunks': len(manifest.cids),
                'total_tokens': manifest.total_tokens,
                'chunks': chunks,
                'expires_at': manifest.expires_at
            })
            
        except Exception as e:
            logger.error(f"Failed to get training data: {e}")
            return web.json_response({
                'error': str(e)
            }, status=500)
    
    async def get_chunk_content(self, request: web.Request) -> web.Response:
        """
        Get actual content of a chunk.
        
        GET /data/chunk/{cid}
        """
        try:
            cid = request.match_info['cid']
            
            # Check if chunk exists
            if cid not in self.data_pool.chunk_index:
                return web.json_response({
                    'error': 'Chunk not found'
                }, status=404)
            
            chunk = self.data_pool.chunk_index[cid]
            
            # Load chunk content
            chunk_path = self.data_pool.chunks_path / chunk.shard_id / f"{cid}.bin"
            
            if not chunk_path.exists():
                return web.json_response({
                    'error': 'Chunk data not found'
                }, status=404)
            
            content = chunk_path.read_bytes()
            
            return web.Response(
                body=content,
                content_type='application/octet-stream',
                headers={
                    'X-Chunk-CID': cid,
                    'X-Chunk-License': chunk.license,
                    'X-Chunk-Quality': str(chunk.quality_score)
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to get chunk: {e}")
            return web.json_response({
                'error': str(e)
            }, status=500)
    
    # ==================== Harvesting Endpoints ====================
    
    async def harvest_dataset(self, request: web.Request) -> web.Response:
        """
        Harvest a dataset from internet sources.
        
        POST /data/harvest
        {
            "source": "huggingface",
            "dataset": "wikipedia",
            "max_samples": 1000
        }
        """
        try:
            data = await request.json()
            
            source = data.get('source', 'huggingface')
            dataset = data.get('dataset')
            max_samples = data.get('max_samples', 1000)
            
            if source == 'huggingface':
                result = self.harvester.huggingface.harvest_dataset(
                    dataset_name=dataset,
                    max_samples=max_samples
                )
            else:
                return web.json_response({
                    'error': f'Unknown source: {source}'
                }, status=400)
            
            return web.json_response(result)
            
        except Exception as e:
            logger.error(f"Harvest failed: {e}")
            return web.json_response({
                'error': str(e)
            }, status=500)
    
    async def list_datasets(self, request: web.Request) -> web.Response:
        """
        List available datasets for harvesting.
        
        GET /data/datasets
        """
        try:
            datasets = []
            
            # Get HuggingFace datasets
            for ds in self.harvester.huggingface.list_available_datasets():
                datasets.append({
                    'source': ds.source,
                    'name': ds.name,
                    'url': ds.url,
                    'license': ds.license,
                    'tags': ds.tags,
                    'quality': ds.quality_tier
                })
            
            return web.json_response({
                'datasets': datasets,
                'total': len(datasets)
            })
            
        except Exception as e:
            logger.error(f"Failed to list datasets: {e}")
            return web.json_response({
                'error': str(e)
            }, status=500)
    
    # ==================== Statistics & Transparency ====================
    
    async def get_pool_stats(self, request: web.Request) -> web.Response:
        """
        Get data pool statistics.
        
        GET /data/stats
        """
        try:
            pool_stats = self.data_pool.get_stats()
            harvest_stats = self.harvester.get_harvest_stats()
            
            return web.json_response({
                'pool': pool_stats,
                'harvest': harvest_stats,
                'timestamp': int(time.time())
            })
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return web.json_response({
                'error': str(e)
            }, status=500)
    
    async def get_expert_usage(self, request: web.Request) -> web.Response:
        """
        Get data usage statistics for an expert.
        
        GET /data/usage/{expert_id}
        """
        try:
            expert_id = request.match_info['expert_id']
            
            stats = self.scheduler.get_expert_stats(expert_id)
            
            return web.json_response(stats)
            
        except Exception as e:
            logger.error(f"Failed to get usage: {e}")
            return web.json_response({
                'error': str(e)
            }, status=500)
    
    async def search_chunks(self, request: web.Request) -> web.Response:
        """
        Search for chunks by tags and quality.
        
        GET /data/search?tags=medical&min_quality=0.8&limit=100
        """
        try:
            # Parse parameters
            tags = request.query.get('tags', '').split(',') if request.query.get('tags') else []
            min_quality = float(request.query.get('min_quality', 0.0))
            limit = int(request.query.get('limit', 100))
            
            # Search chunks
            results = []
            
            for cid, chunk in self.data_pool.chunk_index.items():
                # Check quality
                if chunk.quality_score < min_quality:
                    continue
                
                # Check tags
                if tags and not any(tag in chunk.tags for tag in tags):
                    continue
                
                results.append({
                    'cid': chunk.cid,
                    'source': chunk.source_url,
                    'license': chunk.license,
                    'quality': chunk.quality_score,
                    'tags': chunk.tags,
                    'size': chunk.size_bytes,
                    'tokens': chunk.token_count,
                    'usage_count': chunk.usage_count
                })
                
                if len(results) >= limit:
                    break
            
            return web.json_response({
                'results': results,
                'total': len(results)
            })
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return web.json_response({
                'error': str(e)
            }, status=500)
    
    def setup_routes(self, app: web.Application):
        """Setup API routes."""
        # Upload routes
        app.router.add_post('/data/upload', self.upload_data)
        app.router.add_post('/data/upload_text', self.upload_text)
        
        # Distribution routes
        app.router.add_get('/data/training/{expert_id}', self.get_training_data)
        app.router.add_get('/data/chunk/{cid}', self.get_chunk_content)
        
        # Harvesting routes
        app.router.add_post('/data/harvest', self.harvest_dataset)
        app.router.add_get('/data/datasets', self.list_datasets)
        
        # Stats & transparency routes
        app.router.add_get('/data/stats', self.get_pool_stats)
        app.router.add_get('/data/usage/{expert_id}', self.get_expert_usage)
        app.router.add_get('/data/search', self.search_chunks)
        
        logger.info("Data Pool API routes configured")
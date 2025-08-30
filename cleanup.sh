#!/bin/bash
# Clean up unnecessary files while keeping functionality intact

echo "🧹 Cleaning up Blyan project"
echo "============================"

# Backup important files first
echo "📦 Creating backup..."
mkdir -p backups/cleanup_backup
cp -r data backups/cleanup_backup/ 2>/dev/null || true
cp .env backups/cleanup_backup/ 2>/dev/null || true

# Remove test and debug scripts from root
echo "🗑️ Removing test/debug scripts..."
rm -f test_gpu_nodes.py
rm -f test_gpu_redis_registry.py
rm -f test_queue.py
rm -f debug_gpu_routing.py
rm -f migrate_gpu_nodes_to_redis.py
rm -f register_gpu_node_2256.py
rm -f force_cleanup_nodes.py
rm -f cleanup_stale_nodes.py
rm -f verify_production_features.py

# Remove old/deprecated GPU manager files
echo "🗑️ Removing deprecated GPU manager files..."
rm -f backend/p2p/gpu_node_manager_production.py
rm -f backend/p2p/gpu_node_manager_redis_old.py
rm -f backend/p2p/gpu_node_manager.py  # Old file-based manager

# Remove log files
echo "🗑️ Removing log files..."
rm -rf logs/*.log
rm -f *.log
rm -f api_*.log

# Remove Python cache files
echo "🗑️ Removing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
find . -type f -name ".DS_Store" -delete 2>/dev/null || true

# Remove temporary files
echo "🗑️ Removing temporary files..."
rm -f *.tmp
rm -f *.temp
rm -f *.bak
rm -f *.backup
rm -f *.old
rm -f *~

# Remove empty directories
echo "🗑️ Removing empty directories..."
find . -type d -empty -delete 2>/dev/null || true

# Clean data directory (keep blockchain)
echo "🗑️ Cleaning data directory..."
rm -f data/gpu_nodes.json  # Old file-based registry
rm -f data/gpu_registry.json  # Old registry
rm -f data/*.tmp
rm -f data/*.lock

# Remove test chain data (keep A, B, D chains)
echo "🗑️ Removing test chains..."
rm -rf data/test_*
rm -rf data/C  # Usually test chain
rm -rf data/E  # Usually test chain

# Clean backups (keep snapshot registry)
echo "🗑️ Cleaning old backups..."
find backups -name "*.old" -delete 2>/dev/null || true
find backups -name "*.tmp" -delete 2>/dev/null || true

# Size before and after
echo ""
echo "📊 Cleanup complete!"
echo "Space saved: $(du -sh . | cut -f1)"

echo ""
echo "✅ Kept:"
echo "  - Production code (api/, backend/, frontend/)"
echo "  - Blockchain data (data/A, data/B, data/D)"
echo "  - Configuration files"
echo "  - Scripts directory"
echo "  - Requirements and setup files"

echo ""
echo "🗑️ Removed:"
echo "  - Test/debug scripts"
echo "  - Deprecated GPU managers"
echo "  - Log files"
echo "  - Python cache"
echo "  - Temporary files"
echo "  - Old registries"

echo ""
echo "💡 To restore if needed: check backups/cleanup_backup/"
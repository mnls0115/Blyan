#!/bin/bash
# Final cleanup - remove all unnecessary files and streamline codebase

echo "🧹 Final Cleanup for Production"
echo "================================"

# Remove unnecessary test files from tests directory
echo "🗑️ Cleaning tests directory..."
rm -f tests/test_gpu_redis_registry.py
rm -f tests/test_dense_learning_e2e.py
# Keep essential tests only

# Remove old snapshot files if they exist
echo "🗑️ Cleaning old snapshots..."
find backups -name "snapshot_*.json" -mtime +7 -delete 2>/dev/null || true

# Remove example and template files that aren't needed
echo "🗑️ Removing example files..."
rm -f *.example
rm -f *.sample

# Clean up scripts directory
echo "🗑️ Cleaning scripts directory..."
rm -f scripts/test_*.py 2>/dev/null || true
rm -f scripts/demo_*.py 2>/dev/null || true
rm -f scripts/extract_individual_experts.py  # Old MoE script

# Remove any remaining .pyc files
echo "🗑️ Final Python cache cleanup..."
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Remove editor backup files
echo "🗑️ Removing editor files..."
find . -type f -name ".*.swp" -delete 2>/dev/null || true
find . -type f -name ".*.swo" -delete 2>/dev/null || true
find . -type f -name "*~" -delete 2>/dev/null || true

# Clean up git ignored files (careful with this)
echo "🗑️ Cleaning git ignored files..."
git clean -fdX --dry-run | head -20  # Show what would be removed
echo ""
read -p "Remove git ignored files? (y/N): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git clean -fdX
fi

# Update .gitignore to ensure we don't track unnecessary files
echo "📝 Updating .gitignore..."
cat >> .gitignore << 'EOF'

# Cleanup additions
*.pyc
__pycache__/
*.log
*.tmp
*.temp
*.bak
*.backup
*.old
.DS_Store
*.swp
*.swo
*~
test_*.py
debug_*.py
migrate_*.py
data/gpu_nodes.json
data/gpu_registry.json
EOF

echo ""
echo "✅ Final cleanup complete!"
echo ""
echo "📊 Project size: $(du -sh . | cut -f1)"
echo ""
echo "🎯 Ready for production deployment!"
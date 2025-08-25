# Blyan Network Operations Runbook

## 🚀 Quick Deploy

### Production Deploy (DigitalOcean)
```bash
# 1. Pull latest code
cd /root/blyan
git pull origin main

# 2. Install/update dependencies
pip install -r requirements.txt

# 3. Run migrations
python scripts/migrate_db.py

# 4. Restart services
systemctl restart aiblock
systemctl restart nginx

# 5. Verify health
curl http://localhost:8000/health
```

### Rollback
```bash
# Quick rollback to previous version
cd /root/blyan
git log --oneline -5  # Find previous commit
git checkout <commit-hash>
systemctl restart aiblock
```

## 🔑 API Key Rotation

### Rotate All Keys (Emergency)
```bash
# 1. Generate new master key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# 2. Update environment
vim /etc/systemd/system/aiblock.service
# Update: Environment="API_MASTER_KEY=new_key_here"

# 3. Reload and restart
systemctl daemon-reload
systemctl restart aiblock

# 4. Notify users via email/dashboard
```

### Rotate Individual Key
```bash
# Via API
curl -X POST https://blyan.com/api/keys/{key_id}/rotate \
  -H "X-API-Key: admin_key"
```

## 📈 Scale Nodes

### Add GPU Node
```bash
# On new GPU node
docker run -d \
  --gpus all \
  -e API_KEY=node_operator_key \
  -e NODE_ID=gpu_node_$(hostname) \
  -e MAIN_SERVER_URL=https://blyan.com/api \
  -p 8001:8001 \
  blyan/gpu-node:latest
```

### Remove Node Gracefully
```bash
# Send shutdown signal (allows current requests to complete)
curl -X DELETE https://blyan.com/api/p2p/nodes/{node_id} \
  -H "X-API-Key: admin_key"
```

## 🔄 Recovery Procedures

### Redis Recovery
```bash
# 1. Check Redis status
redis-cli ping

# 2. If down, restart
systemctl restart redis

# 3. If data corrupted, restore from backup
systemctl stop redis
cp /backup/redis/dump.rdb /var/lib/redis/dump.rdb
systemctl start redis

# 4. Clear specific cache if needed
redis-cli
> SELECT 5  # Free tier DB
> FLUSHDB
```

### PostgreSQL Recovery
```bash
# 1. Check connection
psql -U blyan -d blyandb -c "SELECT 1;"

# 2. If down, restart
systemctl restart postgresql

# 3. Restore from daily backup
systemctl stop aiblock
pg_restore -U blyan -d blyandb /backup/postgres/daily_$(date +%Y%m%d).dump
systemctl start aiblock

# 4. Run migrations after restore
python scripts/migrate_db.py
```

### Stuck Streaming Connections
```bash
# 1. Identify stuck connections
netstat -an | grep :8000 | grep ESTABLISHED | wc -l

# 2. If too many (>1000), restart with connection draining
systemctl reload aiblock  # Graceful reload
sleep 30  # Wait for connections to drain
systemctl restart aiblock  # Full restart if needed
```

## 🚨 Emergency Procedures

### High CPU/Memory
```bash
# 1. Check top processes
htop

# 2. Check specific service
systemctl status aiblock

# 3. Emergency restart with rate limiting
# Add to nginx.conf temporarily:
limit_req_zone $binary_remote_addr zone=emergency:10m rate=1r/s;
# Then in location block:
limit_req zone=emergency burst=5;

nginx -s reload
```

### DDoS Attack
```bash
# 1. Enable Cloudflare "Under Attack" mode
# Via Cloudflare dashboard

# 2. Block specific IPs
iptables -A INPUT -s <bad_ip> -j DROP

# 3. Increase rate limits
vim /etc/nginx/nginx.conf
# Reduce rate limits to 1r/s temporarily
nginx -s reload
```

### Database Lock
```bash
# 1. Find blocking queries
psql -U blyan -d blyandb
> SELECT pg_terminate_backend(pid) 
  FROM pg_stat_activity 
  WHERE state = 'idle in transaction' 
  AND query_start < now() - interval '5 minutes';

# 2. If severe, restart database
systemctl restart postgresql
```

## 📊 Monitoring Commands

### Check Service Health
```bash
# Overall health
curl http://localhost:8000/health

# Node status
curl https://blyan.com/api/p2p/nodes -H "X-API-Key: $KEY"

# Free tier stats
redis-cli -n 5 info stats

# Active connections
ss -s | grep estab
```

### View Logs
```bash
# API logs
tail -f /var/log/aiblock.log

# Error logs
tail -f /var/log/aiblock.error.log

# Nginx access logs
tail -f /var/log/nginx/access.log

# System logs
journalctl -u aiblock -f
```

### Performance Metrics
```bash
# CPU and Memory
htop

# Disk usage
df -h

# Network connections
netstat -an | grep :8000 | wc -l

# Redis memory
redis-cli info memory | grep used_memory_human

# PostgreSQL connections
psql -U blyan -d blyandb -c "SELECT count(*) FROM pg_stat_activity;"
```

## 🔧 Common Fixes

### "502 Bad Gateway"
```bash
# API server is down
systemctl status aiblock
systemctl restart aiblock
```

### "Too Many Connections"
```bash
# Increase connection limits
psql -U blyan -d blyandb -c "ALTER SYSTEM SET max_connections = 200;"
systemctl restart postgresql
```

### "Out of Memory"
```bash
# Clear Redis cache
redis-cli
> SELECT 3  # Leaderboard cache
> FLUSHDB
> SELECT 5  # Free tier cache  
> FLUSHDB
```

### "SSL Certificate Expired"
```bash
# Renew Let's Encrypt cert
certbot renew
systemctl restart nginx
```

## 📞 Escalation

1. **Level 1**: DevOps on-call (check PagerDuty)
2. **Level 2**: Backend lead (Slack: #backend-oncall)
3. **Level 3**: CTO (emergency phone list)

## 📝 Checklist After Incident

- [ ] Services restored and verified
- [ ] Root cause identified
- [ ] Temporary fixes reverted
- [ ] Monitoring alerts cleared
- [ ] Incident report filed
- [ ] User communication sent (if needed)
- [ ] Runbook updated with new procedures
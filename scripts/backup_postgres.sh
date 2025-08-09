#!/bin/bash
#
# PostgreSQL Backup Script for Blyan Network
# Automated backup with rotation and remote storage
#

set -euo pipefail

# Configuration
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-blyan_db}"
DB_USER="${DB_USER:-blyan_user}"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/blyan}"
REMOTE_BACKUP="${REMOTE_BACKUP:-s3://blyan-backups}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Ensure backup directory exists
mkdir -p "$BACKUP_DIR"
mkdir -p "$BACKUP_DIR/hourly"
mkdir -p "$BACKUP_DIR/daily"
mkdir -p "$BACKUP_DIR/weekly"

# Logging
LOG_FILE="$BACKUP_DIR/backup.log"
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Check dependencies
check_dependencies() {
    local deps=("pg_dump" "gzip" "aws")
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            log "ERROR: $dep is not installed"
            exit 1
        fi
    done
}

# Perform backup
perform_backup() {
    local backup_type="$1"
    local backup_file="$BACKUP_DIR/$backup_type/blyan_${backup_type}_${TIMESTAMP}.sql.gz"
    
    log "Starting $backup_type backup..."
    
    # Record start time
    local start_time=$(date +%s)
    
    # Perform pg_dump with compression
    PGPASSWORD="${DB_PASSWORD:-}" pg_dump \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        --clean \
        --if-exists \
        --no-owner \
        --no-privileges \
        --verbose 2>> "$LOG_FILE" | gzip -9 > "$backup_file"
    
    # Calculate backup size and duration
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local size=$(du -h "$backup_file" | cut -f1)
    
    log "Backup completed: $backup_file (Size: $size, Duration: ${duration}s)"
    
    # Create symlink to latest backup
    ln -sf "$backup_file" "$BACKUP_DIR/$backup_type/latest.sql.gz"
    
    echo "$backup_file"
}

# Upload to remote storage
upload_to_remote() {
    local backup_file="$1"
    local backup_type="$2"
    
    if [[ -z "$REMOTE_BACKUP" ]] || [[ "$REMOTE_BACKUP" == "none" ]]; then
        log "Remote backup disabled"
        return 0
    fi
    
    log "Uploading to remote storage..."
    
    if [[ "$REMOTE_BACKUP" == s3://* ]]; then
        # AWS S3 upload
        aws s3 cp "$backup_file" "$REMOTE_BACKUP/$backup_type/" \
            --storage-class GLACIER_IR \
            --metadata "timestamp=$TIMESTAMP,type=$backup_type" \
            2>> "$LOG_FILE"
    elif [[ "$REMOTE_BACKUP" == gs://* ]]; then
        # Google Cloud Storage upload
        gsutil cp "$backup_file" "$REMOTE_BACKUP/$backup_type/" 2>> "$LOG_FILE"
    else
        log "ERROR: Unsupported remote storage: $REMOTE_BACKUP"
        return 1
    fi
    
    log "Remote upload completed"
}

# Cleanup old backups
cleanup_old_backups() {
    local backup_type="$1"
    local retention_days="$2"
    
    log "Cleaning up old $backup_type backups (retention: $retention_days days)..."
    
    # Local cleanup
    find "$BACKUP_DIR/$backup_type" -name "*.sql.gz" -type f -mtime +$retention_days -delete
    
    # Remote cleanup (S3)
    if [[ "$REMOTE_BACKUP" == s3://* ]]; then
        aws s3 ls "$REMOTE_BACKUP/$backup_type/" | \
            awk '{print $4}' | \
            while read -r file; do
                file_date=$(echo "$file" | grep -oP '\d{8}' | head -1)
                if [[ -n "$file_date" ]]; then
                    file_age=$(( ($(date +%s) - $(date -d "$file_date" +%s)) / 86400 ))
                    if [[ $file_age -gt $retention_days ]]; then
                        aws s3 rm "$REMOTE_BACKUP/$backup_type/$file"
                        log "Deleted remote: $file"
                    fi
                fi
            done
    fi
}

# Verify backup integrity
verify_backup() {
    local backup_file="$1"
    
    log "Verifying backup integrity..."
    
    # Test gzip integrity
    if ! gzip -t "$backup_file" 2>> "$LOG_FILE"; then
        log "ERROR: Backup file is corrupted!"
        return 1
    fi
    
    # Test PostgreSQL dump (first 100 lines)
    if ! zcat "$backup_file" | head -100 | grep -q "PostgreSQL database dump"; then
        log "ERROR: Invalid PostgreSQL dump format!"
        return 1
    fi
    
    log "Backup verification passed"
    return 0
}

# Send notification
send_notification() {
    local status="$1"
    local message="$2"
    
    # Slack notification
    if [[ -n "${SLACK_WEBHOOK:-}" ]]; then
        local color="good"
        [[ "$status" == "ERROR" ]] && color="danger"
        
        curl -X POST "$SLACK_WEBHOOK" \
            -H 'Content-Type: application/json' \
            -d "{
                \"attachments\": [{
                    \"color\": \"$color\",
                    \"title\": \"Blyan Backup $status\",
                    \"text\": \"$message\",
                    \"footer\": \"$(hostname)\",
                    \"ts\": $(date +%s)
                }]
            }" 2>> "$LOG_FILE"
    fi
    
    # Email notification (if configured)
    if [[ -n "${ALERT_EMAIL:-}" ]]; then
        echo "$message" | mail -s "Blyan Backup $status" "$ALERT_EMAIL"
    fi
}

# Main execution
main() {
    local backup_type="${1:-hourly}"
    local retention_days
    
    # Set retention based on backup type
    case "$backup_type" in
        hourly)
            retention_days=2
            ;;
        daily)
            retention_days=30
            ;;
        weekly)
            retention_days=90
            ;;
        *)
            log "ERROR: Invalid backup type: $backup_type"
            exit 1
            ;;
    esac
    
    log "=== Starting Blyan PostgreSQL Backup ==="
    log "Type: $backup_type, Host: $DB_HOST, Database: $DB_NAME"
    
    # Check dependencies
    check_dependencies
    
    # Perform backup
    backup_file=$(perform_backup "$backup_type")
    
    # Verify backup
    if ! verify_backup "$backup_file"; then
        send_notification "ERROR" "Backup verification failed for $backup_file"
        exit 1
    fi
    
    # Upload to remote storage
    if ! upload_to_remote "$backup_file" "$backup_type"; then
        send_notification "WARNING" "Remote upload failed for $backup_file"
    fi
    
    # Cleanup old backups
    cleanup_old_backups "$backup_type" "$retention_days"
    
    # Calculate metrics
    local total_size=$(du -sh "$BACKUP_DIR" | cut -f1)
    local backup_count=$(find "$BACKUP_DIR" -name "*.sql.gz" | wc -l)
    
    log "=== Backup Complete ==="
    log "Total backups: $backup_count, Total size: $total_size"
    
    # Send success notification
    send_notification "SUCCESS" "Backup completed: $backup_file (Size: $(du -h "$backup_file" | cut -f1))"
}

# Handle script arguments
case "${1:-}" in
    hourly|daily|weekly)
        main "$1"
        ;;
    *)
        echo "Usage: $0 {hourly|daily|weekly}"
        echo ""
        echo "Environment variables:"
        echo "  DB_HOST         Database host (default: localhost)"
        echo "  DB_PORT         Database port (default: 5432)"
        echo "  DB_NAME         Database name (default: blyan_db)"
        echo "  DB_USER         Database user (default: blyan_user)"
        echo "  DB_PASSWORD     Database password"
        echo "  BACKUP_DIR      Local backup directory (default: /var/backups/blyan)"
        echo "  REMOTE_BACKUP   Remote storage URL (s3:// or gs://)"
        echo "  SLACK_WEBHOOK   Slack webhook URL for notifications"
        echo "  ALERT_EMAIL     Email address for alerts"
        exit 1
        ;;
esac
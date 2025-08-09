#!/bin/bash
#
# PostgreSQL Restore Rehearsal Script
# Tests disaster recovery with RTO measurement
#

set -euo pipefail

# Configuration
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-blyan_db}"
DB_USER="${DB_USER:-blyan_user}"
TEST_DB_NAME="${TEST_DB_NAME:-blyan_test}"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/blyan}"
LOG_FILE="${LOG_FILE:-/var/log/blyan_restore.log}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging
log() {
    echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}✗${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1" | tee -a "$LOG_FILE"
}

# Find latest backup
find_latest_backup() {
    local backup_type="${1:-daily}"
    local latest_backup="$BACKUP_DIR/$backup_type/latest.sql.gz"
    
    if [[ ! -f "$latest_backup" ]]; then
        # Try to find most recent backup file
        latest_backup=$(find "$BACKUP_DIR/$backup_type" -name "*.sql.gz" -type f 2>/dev/null | sort -r | head -1)
        
        if [[ -z "$latest_backup" ]]; then
            log_error "No backup found in $BACKUP_DIR/$backup_type"
            return 1
        fi
    fi
    
    echo "$latest_backup"
}

# Create test database
create_test_database() {
    log "Creating test database: $TEST_DB_NAME"
    
    PGPASSWORD="${DB_PASSWORD:-}" psql \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -d postgres \
        -c "DROP DATABASE IF EXISTS $TEST_DB_NAME;" 2>> "$LOG_FILE"
    
    PGPASSWORD="${DB_PASSWORD:-}" psql \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -d postgres \
        -c "CREATE DATABASE $TEST_DB_NAME WITH TEMPLATE template0 ENCODING 'UTF8';" 2>> "$LOG_FILE"
    
    log_success "Test database created"
}

# Restore backup
restore_backup() {
    local backup_file="$1"
    local target_db="$2"
    
    log "Restoring backup: $backup_file"
    log "Target database: $target_db"
    
    # Record start time for RTO measurement
    local start_time=$(date +%s)
    
    # Decompress and restore
    gunzip -c "$backup_file" | PGPASSWORD="${DB_PASSWORD:-}" psql \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -d "$target_db" \
        --single-transaction \
        -v ON_ERROR_STOP=1 \
        > /dev/null 2>> "$LOG_FILE"
    
    local restore_status=$?
    
    # Calculate RTO
    local end_time=$(date +%s)
    local rto=$((end_time - start_time))
    
    if [[ $restore_status -eq 0 ]]; then
        log_success "Restore completed in ${rto} seconds"
    else
        log_error "Restore failed after ${rto} seconds"
        return 1
    fi
    
    echo "$rto"
}

# Verify restored data
verify_restored_data() {
    local target_db="$1"
    
    log "Verifying restored data..."
    
    local checks_passed=0
    local checks_total=0
    
    # Check 1: Table existence
    ((checks_total++))
    local tables=$(PGPASSWORD="${DB_PASSWORD:-}" psql \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -d "$target_db" \
        -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" 2>> "$LOG_FILE")
    
    if [[ $tables -gt 0 ]]; then
        log_success "Tables exist: $tables tables found"
        ((checks_passed++))
    else
        log_error "No tables found"
    fi
    
    # Check 2: User balances table
    ((checks_total++))
    local user_count=$(PGPASSWORD="${DB_PASSWORD:-}" psql \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -d "$target_db" \
        -t -c "SELECT COUNT(*) FROM user_balances;" 2>> "$LOG_FILE" || echo "0")
    
    if [[ $user_count -gt 0 ]]; then
        log_success "User balances: $user_count users"
        ((checks_passed++))
    else
        log_warning "No user balances found"
    fi
    
    # Check 3: Transactions table
    ((checks_total++))
    local tx_count=$(PGPASSWORD="${DB_PASSWORD:-}" psql \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -d "$target_db" \
        -t -c "SELECT COUNT(*) FROM transactions;" 2>> "$LOG_FILE" || echo "0")
    
    if [[ $tx_count -gt 0 ]]; then
        log_success "Transactions: $tx_count transactions"
        ((checks_passed++))
    else
        log_warning "No transactions found"
    fi
    
    # Check 4: Data integrity - balance consistency
    ((checks_total++))
    local balance_check=$(PGPASSWORD="${DB_PASSWORD:-}" psql \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -d "$target_db" \
        -t -c "SELECT COUNT(*) FROM user_balances WHERE balance < 0;" 2>> "$LOG_FILE" || echo "-1")
    
    if [[ $balance_check -eq 0 ]]; then
        log_success "Balance integrity: No negative balances"
        ((checks_passed++))
    else
        log_error "Balance integrity: Found negative balances"
    fi
    
    # Check 5: Recent data
    ((checks_total++))
    local recent_tx=$(PGPASSWORD="${DB_PASSWORD:-}" psql \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -d "$target_db" \
        -t -c "SELECT COUNT(*) FROM transactions WHERE created_at > NOW() - INTERVAL '7 days';" 2>> "$LOG_FILE" || echo "0")
    
    if [[ $recent_tx -gt 0 ]]; then
        log_success "Recent data: $recent_tx transactions in last 7 days"
        ((checks_passed++))
    else
        log_warning "No recent transactions found"
    fi
    
    log "Verification complete: $checks_passed/$checks_total checks passed"
    
    if [[ $checks_passed -eq $checks_total ]]; then
        return 0
    else
        return 1
    fi
}

# Clean up test database
cleanup_test_database() {
    log "Cleaning up test database..."
    
    PGPASSWORD="${DB_PASSWORD:-}" psql \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -d postgres \
        -c "DROP DATABASE IF EXISTS $TEST_DB_NAME;" 2>> "$LOG_FILE"
    
    log_success "Test database cleaned up"
}

# Calculate RPO
calculate_rpo() {
    local backup_file="$1"
    
    # Get backup timestamp from filename
    local backup_timestamp=$(echo "$backup_file" | grep -oP '\d{8}_\d{6}' | head -1)
    
    if [[ -n "$backup_timestamp" ]]; then
        # Convert to epoch
        local year="${backup_timestamp:0:4}"
        local month="${backup_timestamp:4:2}"
        local day="${backup_timestamp:6:2}"
        local hour="${backup_timestamp:9:2}"
        local minute="${backup_timestamp:11:2}"
        local second="${backup_timestamp:13:2}"
        
        local backup_epoch=$(date -d "$year-$month-$day $hour:$minute:$second" +%s 2>/dev/null || echo "0")
        local current_epoch=$(date +%s)
        
        if [[ $backup_epoch -gt 0 ]]; then
            local rpo=$((current_epoch - backup_epoch))
            echo "$rpo"
            return 0
        fi
    fi
    
    # Fallback to file modification time
    local file_mod_time=$(stat -c %Y "$backup_file" 2>/dev/null || stat -f %m "$backup_file" 2>/dev/null)
    local current_time=$(date +%s)
    local rpo=$((current_time - file_mod_time))
    echo "$rpo"
}

# Generate report
generate_report() {
    local backup_file="$1"
    local rto="$2"
    local rpo="$3"
    local verification_status="$4"
    
    local report_file="$BACKUP_DIR/restore_rehearsal_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "========================================="
        echo "    DISASTER RECOVERY REHEARSAL REPORT"
        echo "========================================="
        echo ""
        echo "Date: $(date)"
        echo "Host: $(hostname)"
        echo ""
        echo "BACKUP DETAILS:"
        echo "  File: $backup_file"
        echo "  Size: $(du -h "$backup_file" | cut -f1)"
        echo "  Age: $(( rpo / 3600 )) hours $(( (rpo % 3600) / 60 )) minutes"
        echo ""
        echo "RECOVERY METRICS:"
        echo "  RPO (Recovery Point Objective): $(( rpo / 60 )) minutes"
        echo "  RTO (Recovery Time Objective): $(( rto / 60 )) minutes $(( rto % 60 )) seconds"
        echo ""
        echo "VERIFICATION:"
        if [[ $verification_status -eq 0 ]]; then
            echo "  Status: PASSED ✓"
        else
            echo "  Status: FAILED ✗"
        fi
        echo ""
        echo "RECOMMENDATIONS:"
        if [[ $rpo -gt 3600 ]]; then
            echo "  - RPO exceeds 1 hour. Consider more frequent backups."
        fi
        if [[ $rto -gt 1800 ]]; then
            echo "  - RTO exceeds 30 minutes. Consider performance optimization."
        fi
        if [[ $verification_status -ne 0 ]]; then
            echo "  - Data verification failed. Review backup integrity."
        fi
        echo ""
        echo "========================================="
    } | tee "$report_file"
    
    log_success "Report saved to: $report_file"
}

# Main execution
main() {
    log "=== Starting Disaster Recovery Rehearsal ==="
    
    # Find latest backup
    backup_type="${1:-daily}"
    backup_file=$(find_latest_backup "$backup_type")
    
    if [[ -z "$backup_file" ]] || [[ ! -f "$backup_file" ]]; then
        log_error "No backup file found"
        exit 1
    fi
    
    log "Using backup: $backup_file"
    log "Backup size: $(du -h "$backup_file" | cut -f1)"
    
    # Calculate RPO
    rpo=$(calculate_rpo "$backup_file")
    log "RPO: $(( rpo / 60 )) minutes"
    
    # Create test database
    create_test_database
    
    # Restore backup and measure RTO
    rto=$(restore_backup "$backup_file" "$TEST_DB_NAME")
    
    # Verify restored data
    verification_status=0
    if ! verify_restored_data "$TEST_DB_NAME"; then
        verification_status=1
        log_warning "Some verification checks failed"
    fi
    
    # Generate report
    generate_report "$backup_file" "$rto" "$rpo" "$verification_status"
    
    # Clean up
    if [[ "${KEEP_TEST_DB:-false}" != "true" ]]; then
        cleanup_test_database
    else
        log_warning "Test database kept: $TEST_DB_NAME"
    fi
    
    # Send notification
    if [[ -n "${SLACK_WEBHOOK:-}" ]]; then
        local status_emoji="✅"
        [[ $verification_status -ne 0 ]] && status_emoji="⚠️"
        
        curl -X POST "$SLACK_WEBHOOK" \
            -H 'Content-Type: application/json' \
            -d "{
                \"text\": \"$status_emoji DR Rehearsal Complete\",
                \"attachments\": [{
                    \"fields\": [
                        {\"title\": \"RPO\", \"value\": \"$(( rpo / 60 )) minutes\", \"short\": true},
                        {\"title\": \"RTO\", \"value\": \"$(( rto / 60 )) min $(( rto % 60 )) sec\", \"short\": true}
                    ]
                }]
            }" 2>> "$LOG_FILE"
    fi
    
    log "=== Disaster Recovery Rehearsal Complete ==="
    
    # Exit with appropriate status
    if [[ $verification_status -eq 0 ]] && [[ $rto -lt 1800 ]] && [[ $rpo -lt 3600 ]]; then
        log_success "All objectives met!"
        exit 0
    else
        log_warning "Some objectives not met"
        exit 1
    fi
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
        echo "  DB_HOST         Database host"
        echo "  DB_PORT         Database port"
        echo "  DB_NAME         Source database name"
        echo "  DB_USER         Database user"
        echo "  DB_PASSWORD     Database password"
        echo "  TEST_DB_NAME    Test database name (default: blyan_test)"
        echo "  BACKUP_DIR      Backup directory"
        echo "  KEEP_TEST_DB    Keep test database after rehearsal (true/false)"
        echo "  SLACK_WEBHOOK   Slack webhook for notifications"
        exit 1
        ;;
esac
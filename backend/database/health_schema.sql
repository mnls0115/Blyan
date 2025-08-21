-- Node Health & Heartbeat Tracking Schema
-- For production fault tolerance and automatic recovery

-- Node heartbeat tracking
CREATE TABLE IF NOT EXISTS node_heartbeats (
  node_id TEXT PRIMARY KEY REFERENCES gpu_nodes(node_id),
  last_heartbeat TIMESTAMPTZ NOT NULL DEFAULT now(),
  consecutive_failures INT DEFAULT 0,
  health_status TEXT DEFAULT 'healthy' CHECK (health_status IN ('healthy', 'degraded', 'unreachable', 'recovering')),
  last_failure_time TIMESTAMPTZ,
  last_recovery_time TIMESTAMPTZ,
  response_time_ms NUMERIC,
  success_rate NUMERIC DEFAULT 1.0  -- Rolling success rate (last 100 attempts)
);

CREATE INDEX IF NOT EXISTS idx_heartbeat_status ON node_heartbeats (health_status, last_heartbeat);

-- Round stage timeouts and SLA tracking
CREATE TABLE IF NOT EXISTS round_stage_metrics (
  round_id UUID REFERENCES learning_rounds(round_id),
  stage TEXT NOT NULL,
  started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  completed_at TIMESTAMPTZ,
  timeout_at TIMESTAMPTZ NOT NULL,
  nodes_total INT NOT NULL DEFAULT 0,
  nodes_responded INT NOT NULL DEFAULT 0,
  nodes_failed INT NOT NULL DEFAULT 0,
  nodes_reassigned INT NOT NULL DEFAULT 0,
  retry_count INT NOT NULL DEFAULT 0,
  timeout_occurred BOOLEAN DEFAULT FALSE,
  PRIMARY KEY(round_id, stage)
);

CREATE INDEX IF NOT EXISTS idx_stage_metrics_timeout ON round_stage_metrics (timeout_at);

-- Task reassignment tracking
CREATE TABLE IF NOT EXISTS task_reassignments (
  id BIGSERIAL PRIMARY KEY,
  round_id UUID REFERENCES learning_rounds(round_id),
  original_node_id TEXT REFERENCES gpu_nodes(node_id),
  new_node_id TEXT REFERENCES gpu_nodes(node_id),
  task_type TEXT NOT NULL,  -- 'data_shard', 'training', 'validation'
  task_id TEXT NOT NULL,
  reason TEXT NOT NULL,  -- 'timeout', 'node_failure', 'validation_failed'
  reassigned_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_reassignments_round ON task_reassignments (round_id, reassigned_at);

-- Node response history for success rate calculation
CREATE TABLE IF NOT EXISTS node_response_history (
  id BIGSERIAL PRIMARY KEY,
  node_id TEXT REFERENCES gpu_nodes(node_id),
  round_id UUID,
  request_type TEXT NOT NULL,
  success BOOLEAN NOT NULL,
  response_time_ms NUMERIC,
  error_message TEXT,
  recorded_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_response_history_node ON node_response_history (node_id, recorded_at DESC);

-- Function to update node health based on response
CREATE OR REPLACE FUNCTION update_node_health(
  p_node_id TEXT,
  p_success BOOLEAN,
  p_response_time_ms NUMERIC DEFAULT NULL
) RETURNS void AS $$
DECLARE
  v_recent_success_rate NUMERIC;
BEGIN
  -- Record the response
  INSERT INTO node_response_history (node_id, request_type, success, response_time_ms)
  VALUES (p_node_id, 'heartbeat', p_success, p_response_time_ms);
  
  -- Calculate recent success rate (last 100 attempts)
  SELECT AVG(CASE WHEN success THEN 1 ELSE 0 END)
  INTO v_recent_success_rate
  FROM (
    SELECT success FROM node_response_history
    WHERE node_id = p_node_id
    ORDER BY recorded_at DESC
    LIMIT 100
  ) recent;
  
  -- Update heartbeat table
  INSERT INTO node_heartbeats (node_id, last_heartbeat, consecutive_failures, response_time_ms, success_rate)
  VALUES (p_node_id, now(), CASE WHEN p_success THEN 0 ELSE 1 END, p_response_time_ms, COALESCE(v_recent_success_rate, 1.0))
  ON CONFLICT (node_id) DO UPDATE SET
    last_heartbeat = now(),
    consecutive_failures = CASE 
      WHEN p_success THEN 0 
      ELSE node_heartbeats.consecutive_failures + 1 
    END,
    response_time_ms = COALESCE(p_response_time_ms, node_heartbeats.response_time_ms),
    success_rate = COALESCE(v_recent_success_rate, node_heartbeats.success_rate),
    last_failure_time = CASE WHEN NOT p_success THEN now() ELSE node_heartbeats.last_failure_time END,
    last_recovery_time = CASE 
      WHEN p_success AND node_heartbeats.health_status IN ('degraded', 'unreachable') 
      THEN now() 
      ELSE node_heartbeats.last_recovery_time 
    END,
    health_status = CASE
      WHEN NOT p_success AND node_heartbeats.consecutive_failures >= 5 THEN 'unreachable'
      WHEN NOT p_success AND node_heartbeats.consecutive_failures >= 3 THEN 'degraded'
      WHEN p_success AND node_heartbeats.health_status = 'unreachable' THEN 'recovering'
      WHEN p_success AND node_heartbeats.consecutive_failures = 0 THEN 'healthy'
      ELSE node_heartbeats.health_status
    END;
END;
$$ LANGUAGE plpgsql;

-- Function to get healthy nodes for task assignment
CREATE OR REPLACE FUNCTION get_healthy_nodes(
  p_min_success_rate NUMERIC DEFAULT 0.8
) RETURNS TABLE(node_id TEXT, reputation_score NUMERIC) AS $$
BEGIN
  RETURN QUERY
  SELECT n.node_id, n.reputation_score
  FROM gpu_nodes n
  LEFT JOIN node_heartbeats h ON n.node_id = h.node_id
  WHERE n.status = 'active'
    AND (h.health_status IS NULL OR h.health_status IN ('healthy', 'recovering'))
    AND (h.success_rate IS NULL OR h.success_rate >= p_min_success_rate)
    AND (h.last_heartbeat IS NULL OR h.last_heartbeat > now() - INTERVAL '10 minutes')
  ORDER BY n.reputation_score DESC, h.success_rate DESC NULLS LAST;
END;
$$ LANGUAGE plpgsql;
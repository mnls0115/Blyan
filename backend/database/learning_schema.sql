-- Production Learning Cycle Schema for Blyan
-- Persistent storage for all learning states

-- 소각 원장 (Burn Ledger)
CREATE TABLE IF NOT EXISTS burns_ledger (
  id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ NOT NULL DEFAULT now(),
  user_addr TEXT NOT NULL,
  request_id TEXT NOT NULL,
  bly_amount NUMERIC(38, 18) NOT NULL,
  round_candidate BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_burns_ts ON burns_ledger (ts);
CREATE INDEX IF NOT EXISTS idx_burns_candidate ON burns_ledger (round_candidate, ts);

-- 학습 라운드 (Learning Rounds)
CREATE TABLE IF NOT EXISTS learning_rounds (
  round_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  state TEXT NOT NULL CHECK (state IN ('TRIGGERED', 'NOTIFYING', 'DATA_ALLOC', 'TRAINING', 'CONSENSUS', 'DELTA_CREATION', 'REWARD_DIST', 'FAILED')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  bly_sum NUMERIC(38,18) NOT NULL,
  min_interval_ok BOOLEAN NOT NULL,
  seed BYTEA NOT NULL,           -- Round deterministic seed (32 bytes)
  config JSONB NOT NULL,          -- Thresholds, quorum, datasets, budget
  target_expert TEXT,             -- Which expert to improve
  base_version TEXT              -- Base model version hash
);

CREATE INDEX IF NOT EXISTS idx_rounds_state ON learning_rounds (state, created_at);
CREATE INDEX IF NOT EXISTS idx_rounds_created ON learning_rounds (created_at DESC);

-- GPU 노드 레지스트리 (Node Registry)
CREATE TABLE IF NOT EXISTS gpu_nodes (
  node_id TEXT PRIMARY KEY,
  base_url TEXT NOT NULL,
  pubkey TEXT NOT NULL,           -- Node's public key for verification
  last_seen TIMESTAMPTZ,
  capabilities JSONB,             -- GPU info, memory, etc
  status TEXT DEFAULT 'active',
  reputation_score NUMERIC DEFAULT 1.0
);

CREATE INDEX IF NOT EXISTS idx_nodes_status ON gpu_nodes (status, last_seen);

-- 데이터 샤딩 할당 (Data Shard Assignments)
CREATE TABLE IF NOT EXISTS data_assignments (
  round_id UUID REFERENCES learning_rounds(round_id),
  node_id TEXT REFERENCES gpu_nodes(node_id),
  shard_id TEXT NOT NULL,
  shard_uri TEXT NOT NULL,
  checksum TEXT NOT NULL,
  assigned_at TIMESTAMPTZ DEFAULT now(),
  fetched_at TIMESTAMPTZ,
  PRIMARY KEY(round_id, node_id, shard_id)
);

CREATE INDEX IF NOT EXISTS idx_assignments_round ON data_assignments (round_id, node_id);

-- 델타 제출 (Delta Submissions)
CREATE TABLE IF NOT EXISTS deltas (
  round_id UUID REFERENCES learning_rounds(round_id),
  node_id TEXT REFERENCES gpu_nodes(node_id),
  delta_id TEXT NOT NULL,
  parent_model_hash TEXT NOT NULL,
  delta_hash TEXT NOT NULL,
  train_meta JSONB NOT NULL,      -- steps, lr, seed, walltime, device
  metrics JSONB NOT NULL,          -- val_loss, exact_match, accuracy
  proof_hash TEXT,                 -- Training log merkle root
  sig TEXT NOT NULL,               -- Node signature
  submitted_at TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY(round_id, node_id, delta_id)
);

CREATE INDEX IF NOT EXISTS idx_deltas_round ON deltas (round_id, submitted_at);

-- 검증/합의 (Validations/Consensus)
CREATE TABLE IF NOT EXISTS validations (
  round_id UUID REFERENCES learning_rounds(round_id),
  validator_node_id TEXT REFERENCES gpu_nodes(node_id),
  delta_id TEXT NOT NULL,
  score NUMERIC NOT NULL,
  report_hash TEXT NOT NULL,
  sig TEXT NOT NULL,
  validated_at TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY(round_id, validator_node_id, delta_id)
);

CREATE INDEX IF NOT EXISTS idx_validations_round ON validations (round_id, score DESC);

-- 보상 (Rewards)
CREATE TABLE IF NOT EXISTS rewards (
  round_id UUID REFERENCES learning_rounds(round_id),
  node_id TEXT REFERENCES gpu_nodes(node_id),
  amount NUMERIC(38,18) NOT NULL,
  basis JSONB NOT NULL,            -- Calculation basis
  tx_hash TEXT,                     -- Blockchain transaction hash
  distributed_at TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY(round_id, node_id)
);

CREATE INDEX IF NOT EXISTS idx_rewards_round ON rewards (round_id);

-- 멱등성 키 추적 (Idempotency Keys)
CREATE TABLE IF NOT EXISTS idempotency_keys (
  key TEXT PRIMARY KEY,
  request_id TEXT NOT NULL,
  response JSONB,
  created_at TIMESTAMPTZ DEFAULT now(),
  expires_at TIMESTAMPTZ DEFAULT now() + INTERVAL '24 hours'
);

CREATE INDEX IF NOT EXISTS idx_idempotency_expires ON idempotency_keys (expires_at);

-- 이벤트 로그 (Event Log for audit)
CREATE TABLE IF NOT EXISTS learning_events (
  id BIGSERIAL PRIMARY KEY,
  round_id UUID,
  event_type TEXT NOT NULL,
  payload JSONB NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_events_round ON learning_events (round_id, created_at);
CREATE INDEX IF NOT EXISTS idx_events_type ON learning_events (event_type, created_at);

-- Helper functions
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_learning_rounds_updated_at BEFORE UPDATE
    ON learning_rounds FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Cleanup old idempotency keys
CREATE OR REPLACE FUNCTION cleanup_expired_idempotency_keys() RETURNS void AS $$
BEGIN
    DELETE FROM idempotency_keys WHERE expires_at < now();
END;
$$ LANGUAGE plpgsql;
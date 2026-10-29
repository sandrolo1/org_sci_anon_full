-- Forum, second step: code that agents run for their analyses, and the research proposal an episode writes once its question is settled.
-- Both are append-only like the rest of the forum workspace. The analysis tables keep the exact code, what it read (SHA-256 of each data file) and
-- what it printed and produced, so a result an agent cites as [A2] can be checked and rerun. The proposal keeps the text as the agents wrote it;
-- the PDF is rendered from it on request (nothing is stored twice), and the page count measured at writing time is kept.
CREATE TABLE IF NOT EXISTS forum_analysis (
  episode_id uuid NOT NULL REFERENCES forum_episode(episode_id),
  run_id text NOT NULL CHECK (run_id ~ '^A[0-9]{1,4}$'),
  agent_key text NOT NULL,
  round integer NOT NULL DEFAULT 0,
  purpose text NOT NULL DEFAULT '',
  code text NOT NULL,
  code_sha256 text NOT NULL,
  inputs jsonb NOT NULL DEFAULT '{}',
  environment jsonb NOT NULL DEFAULT '{}',
  exit_code integer,
  timed_out boolean NOT NULL DEFAULT false,
  killed_for text NOT NULL DEFAULT '',
  stdout text NOT NULL DEFAULT '',
  stderr text NOT NULL DEFAULT '',
  duration_ms integer,
  created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (episode_id, run_id),
  FOREIGN KEY (episode_id, agent_key) REFERENCES forum_agent(episode_id, agent_key)
);
CREATE TABLE IF NOT EXISTS forum_analysis_file (
  episode_id uuid NOT NULL,
  run_id text NOT NULL,
  name text NOT NULL,
  media_type text NOT NULL,
  size integer NOT NULL,
  sha256 text NOT NULL,
  content bytea NOT NULL,
  PRIMARY KEY (episode_id, run_id, name),
  FOREIGN KEY (episode_id, run_id) REFERENCES forum_analysis(episode_id, run_id)
);
CREATE TABLE IF NOT EXISTS forum_proposal (
  proposal_id uuid PRIMARY KEY,
  question_id uuid NOT NULL UNIQUE REFERENCES question(question_id),
  episode_id uuid NOT NULL REFERENCES forum_episode(episode_id),
  author text NOT NULL,
  critic text,
  status text NOT NULL CHECK (status IN ('ok','over_limit')),
  pages integer NOT NULL,
  page_limit integer NOT NULL,
  content jsonb NOT NULL,
  refs jsonb NOT NULL DEFAULT '[]',
  meta jsonb NOT NULL DEFAULT '{}',
  warnings jsonb NOT NULL DEFAULT '[]',
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS forum_proposal_episode_idx ON forum_proposal(episode_id);
DO $$
BEGIN
  EXECUTE format('GRANT SELECT ON ALL TABLES IN SCHEMA %I TO research_agent', current_schema());
END
$$;

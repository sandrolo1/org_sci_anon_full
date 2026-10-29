-- Discovery layer (the proposal's separate Phase 4): the forum where specialist agents work on a validated release.
-- The release itself is never written. An episode is one run of one condition (0, A-F) with a frozen configuration; its messages are
-- append-only (the order is the sequence number) and the questions it produced go to the `question` archive of Appendix D.
-- Only the lifecycle columns of forum_episode (status, status_detail, started_at, finished_at) change after creation.
CREATE TABLE IF NOT EXISTS forum_episode (
  episode_id uuid PRIMARY KEY,
  release_id uuid NOT NULL REFERENCES release(release_id),
  condition text NOT NULL CHECK (condition IN ('0','A','B','C','D','E','F')),
  title text NOT NULL DEFAULT '',
  provider text NOT NULL,
  seed integer NOT NULL,
  config jsonb NOT NULL,
  config_sha256 text NOT NULL,
  budgets_sha256 text NOT NULL,
  prompt_version text NOT NULL,
  status text NOT NULL DEFAULT 'created' CHECK (status IN ('created','running','stopping','completed','stopped','failed','budget_exhausted')),
  status_detail text,
  created_by text NOT NULL CHECK (length(btrim(created_by)) > 0),
  created_at timestamptz NOT NULL DEFAULT now(),
  started_at timestamptz,
  finished_at timestamptz
);
CREATE INDEX IF NOT EXISTS forum_episode_release_idx ON forum_episode(release_id, created_at DESC);

-- The agents of an episode and the models chosen for each of their tasks.
CREATE TABLE IF NOT EXISTS forum_agent (
  episode_id uuid NOT NULL REFERENCES forum_episode(episode_id),
  agent_key text NOT NULL CHECK (agent_key ~ '^[a-z][a-z0-9_]{0,19}$'),
  label text NOT NULL,
  community text CHECK (community IN ('CTO','TIM')),
  role text NOT NULL CHECK (role IN ('baseline','generalist','specialist')),
  profile_id text NOT NULL,
  settings jsonb NOT NULL,
  PRIMARY KEY (episode_id, agent_key)
);

-- Everything said or done in an episode, in order. audience is 'all' (every agent of the episode reads it) or one agent_key (only that agent does).
-- kind: system (the orchestrator's protocol notes), encounter (random items handed over in condition C), reflection (private notes), tool_call,
-- tool_result, message (a public contribution), question (a submitted research question), note (budget and error notices).
CREATE TABLE IF NOT EXISTS forum_message (
  episode_id uuid NOT NULL REFERENCES forum_episode(episode_id),
  seq integer NOT NULL,
  agent_key text,
  kind text NOT NULL CHECK (kind IN ('system','encounter','reflection','tool_call','tool_result','message','question','note')),
  audience text NOT NULL DEFAULT 'all',
  round integer NOT NULL DEFAULT 0,
  content text NOT NULL DEFAULT '',
  data jsonb NOT NULL DEFAULT '{}',
  model text,
  input_tokens integer NOT NULL DEFAULT 0,
  output_tokens integer NOT NULL DEFAULT 0,
  reasoning_tokens integer NOT NULL DEFAULT 0,
  llm_calls integer NOT NULL DEFAULT 0,
  latency_ms integer,
  created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (episode_id, seq),
  FOREIGN KEY (episode_id, agent_key) REFERENCES forum_agent(episode_id, agent_key)
);

-- The archive of Appendix D: a motivated research question with its argumentative chain, predictions and verification plan.
-- `evaluation` stays empty here: the blind expert panels rate questions outside this system.
CREATE TABLE IF NOT EXISTS question (
  question_id uuid PRIMARY KEY,
  episode_id uuid NOT NULL REFERENCES forum_episode(episode_id),
  release_id uuid NOT NULL REFERENCES release(release_id),
  condition text NOT NULL,
  agents text[] NOT NULL,
  rq text NOT NULL CHECK (length(btrim(rq)) > 0),
  content jsonb NOT NULL,
  evidence jsonb NOT NULL DEFAULT '[]',
  evaluation jsonb NOT NULL DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS question_episode_idx ON question(episode_id);

-- The discovery workspace is written by the application, not by the read-only agent role, which reads it like everything else.
DO $$
BEGIN
  EXECUTE format('GRANT SELECT ON ALL TABLES IN SCHEMA %I TO research_agent', current_schema());
END
$$;

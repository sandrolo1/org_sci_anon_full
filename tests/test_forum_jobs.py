"""The jobs of the forum server: a batch runs its episodes one after the other or, with FORUM_BATCH_WORKERS, several at a time. No database, no model: the Runner is a stand-in."""
import threading
import time
import uuid

import pytest

from research0 import forum_api, forum_scoring as fscore
from research0.forum_providers import Cancelled


class FakeCon:
    status = "created"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        status = self.status
        return type("Result", (), {"fetchone": lambda self_: (status,)})()


class Rig:
    """Runners that report they started, then wait to be released; the rig counts how many were running at once."""

    def __init__(self, outcomes=None):
        self.lock, self.started, self.running, self.peak = threading.Lock(), [], 0, 0
        self.release = threading.Event()
        self.outcomes = outcomes or {}
        self.stops = []

    def runner(self, episode_id, provider, stop=None, on_activity=None):
        rig = self

        class Runner:
            def run(self):
                with rig.lock:
                    rig.started.append(str(episode_id))
                    rig.running += 1
                    rig.peak = max(rig.peak, rig.running)
                on_activity(f"thinking about {str(episode_id)[:4]}")
                rig.release.wait(5)
                on_activity("")
                with rig.lock:
                    rig.running -= 1
                rig.stops.append(stop.is_set())
                return rig.outcomes.get(str(episode_id), "completed")
        return Runner()


@pytest.fixture
def rig(monkeypatch):
    the_rig = Rig()
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")
    monkeypatch.setattr(forum_api.psycopg, "connect", lambda *a, **k: FakeCon())
    monkeypatch.setattr(forum_api.forum, "Runner", the_rig.runner)
    monkeypatch.setattr(forum_api, "get_provider", lambda name: object())
    return the_rig


def wait_for(condition, seconds=5):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


def finish(jobs):
    jobs.current["thread"].join(5)
    assert not jobs.current["thread"].is_alive()


def episodes(n):
    return [uuid.uuid4() for _ in range(n)]


def test_the_number_of_workers_is_read_from_the_environment_and_kept_between_1_and_4(monkeypatch):
    monkeypatch.delenv("FORUM_BATCH_WORKERS", raising=False)
    assert forum_api.batch_workers() == 1                                       # one after the other unless someone asks for more
    for value, expected in (("2", 2), ("9", forum_api.MAX_BATCH_WORKERS), ("0", 1), ("-3", 1), ("two", 1), ("", 1)):
        monkeypatch.setenv("FORUM_BATCH_WORKERS", value)
        assert forum_api.batch_workers() == expected
    monkeypatch.setenv("FORUM_BATCH_WORKERS", "3")
    assert forum_api.ForumJobs().workers == 3 and forum_api.ForumJobs(workers=2).workers == 2 and forum_api.ForumJobs(workers=99).workers == forum_api.MAX_BATCH_WORKERS


def test_with_one_worker_the_episodes_of_a_batch_run_one_after_the_other(rig):
    jobs, ids = forum_api.ForumJobs(workers=1), episodes(3)
    rig.release.set()
    jobs.start_batch(uuid.uuid4(), ids, "openai")
    finish(jobs)
    assert rig.peak == 1 and rig.started == [str(i) for i in ids]               # in order
    snap = jobs.snapshot()
    assert snap["running"] is None and snap["running_ids"] == [] and snap["batch"]["done"] == 3 and snap["batch"]["running"] is False


def test_with_two_workers_two_episodes_run_at_once_and_the_page_can_follow_both(rig):
    jobs, ids = forum_api.ForumJobs(workers=2), episodes(4)
    jobs.start_batch(uuid.uuid4(), ids, "openai")
    assert wait_for(lambda: rig.running == 2)
    snap = jobs.snapshot()
    assert sorted(snap["running_ids"]) == sorted(str(i) for i in ids[:2]) and snap["running"] in snap["running_ids"] and snap["batch"]["workers"] == 2
    for episode_id in ids[:2]:                                                  # the episode page asks about its own episode and gets its own activity
        mine = jobs.snapshot(episode_id)
        assert mine["running"] == str(episode_id) and mine["activity"] == f"thinking about {str(episode_id)[:4]}" and "stopping" in mine
    other = jobs.snapshot(ids[3])                                               # an episode that has not started: somebody else's, with no activity
    assert other["running"] in snap["running_ids"] and other["activity"] == "" and "stopping" not in other
    with pytest.raises(ValueError, match="Another episode is running"):
        jobs.start_batch(uuid.uuid4(), episodes(1), "openai")
    rig.release.set()
    finish(jobs)
    assert rig.peak == 2 and sorted(rig.started) == sorted(str(i) for i in ids) and jobs.snapshot()["batch"]["done"] == 4


def test_workers_are_never_more_than_the_episodes_of_the_batch(rig):
    jobs = forum_api.ForumJobs(workers=4)
    rig.release.set()
    jobs.start_batch(uuid.uuid4(), episodes(2), "openai")
    finish(jobs)
    assert jobs.snapshot()["batch"]["workers"] == 2 and rig.peak <= 2


def test_stopping_one_episode_stops_the_batch_and_the_episodes_not_reached_stay_created(rig):
    jobs, ids = forum_api.ForumJobs(workers=2), episodes(6)
    jobs.start_batch(uuid.uuid4(), ids, "openai")
    assert wait_for(lambda: rig.running == 2)
    jobs.stop(ids[1])                                                           # either of the two that run
    rig.release.set()
    finish(jobs)
    assert sorted(rig.started) == sorted(str(i) for i in ids[:2]) and all(rig.stops)
    with pytest.raises(ValueError, match="not running"):
        jobs.stop(ids[1])


def test_a_batch_halts_after_three_failures_in_a_row_and_lets_the_running_episodes_end(rig):
    ids = episodes(8)
    rig.outcomes = {str(i): "failed" for i in ids}
    rig.release.set()
    jobs = forum_api.ForumJobs(workers=2)
    jobs.start_batch(uuid.uuid4(), ids, "openai")
    finish(jobs)
    batch = jobs.snapshot()["batch"]
    assert "3 episodes failed in a row" in batch["halted"] and 3 <= batch["done"] <= 4 < len(ids) and len(rig.started) == batch["done"]


def test_a_success_between_failures_resets_the_count(rig):
    ids = episodes(6)
    rig.outcomes = {str(ids[i]): "failed" for i in (0, 1, 3, 4)}                # fail, fail, ok, fail, fail, ok: never three in a row
    rig.release.set()
    jobs = forum_api.ForumJobs(workers=1)
    jobs.start_batch(uuid.uuid4(), ids, "openai")
    finish(jobs)
    assert jobs.snapshot()["batch"]["halted"] is None and jobs.snapshot()["batch"]["done"] == 6


def test_an_episode_that_is_no_longer_created_is_skipped_and_an_error_counts_as_a_failure(rig, monkeypatch):
    jobs = forum_api.ForumJobs(workers=2)
    monkeypatch.setattr(FakeCon, "status", "completed")
    jobs.start_batch(uuid.uuid4(), episodes(3), "openai")
    finish(jobs)
    assert rig.started == [] and jobs.snapshot()["batch"]["done"] == 3 and jobs.snapshot()["batch"]["halted"] is None
    monkeypatch.setattr(FakeCon, "status", "created")
    monkeypatch.setattr(forum_api, "get_provider", lambda name: (_ for _ in ()).throw(ValueError("no key")))
    broken = forum_api.ForumJobs(workers=2)
    broken.start_batch(uuid.uuid4(), episodes(5), "openai")
    finish(broken)
    batch = broken.snapshot()["batch"]
    assert "ValueError: no key" in batch["error"] and "failed in a row" in batch["halted"] and rig.started == []


def test_a_single_episode_still_runs_alone_and_can_be_stopped(rig):
    jobs, episode_id = forum_api.ForumJobs(workers=2), uuid.uuid4()
    jobs.start(episode_id, "openai")
    assert wait_for(lambda: rig.running == 1)
    snap = jobs.snapshot(episode_id)
    assert snap["running"] == str(episode_id) and snap["running_ids"] == [str(episode_id)] and snap["batch"] is None and snap["activity"].startswith("thinking")
    with pytest.raises(ValueError, match="Another episode is running"):
        jobs.start(uuid.uuid4(), "openai")
    jobs.stop(episode_id)
    rig.release.set()
    finish(jobs)
    assert rig.stops == [True] and jobs.snapshot()["running"] is None


# --- the evaluations take turns -------------------------------------------------------------------------------------------------------------

class Evaluation:
    """Stands in for the body of the evaluation: it says it started and holds the lock until released."""

    def __init__(self, monkeypatch):
        self.lock, self.inside, self.peak, self.entered = threading.Lock(), 0, 0, threading.Event()
        self.release = threading.Event()
        monkeypatch.setattr(fscore, "_score_proposals", self.body)

    def body(self, con, proposal_ids, provider, **kwargs):
        with self.lock:
            self.inside += 1
            self.peak = max(self.peak, self.inside)
        self.entered.set()
        self.release.wait(5)
        with self.lock:
            self.inside -= 1
        return {"stored": len(proposal_ids)}


def in_thread(results, key, **kwargs):
    thread = threading.Thread(target=lambda: results.__setitem__(key, fscore.score_proposals(None, [1, 2], None, **kwargs)), daemon=True)
    thread.start()
    return thread


def test_two_evaluations_never_run_at_once_and_the_second_says_it_waits(monkeypatch):
    evaluation, results, waited = Evaluation(monkeypatch), {}, []
    first = in_thread(results, "first")
    assert evaluation.entered.wait(5)
    second = in_thread(results, "second", on_wait=lambda: waited.append(1))
    assert wait_for(lambda: waited == [1])
    time.sleep(0.3)
    assert evaluation.inside == 1 and "second" not in results                    # still waiting its turn
    evaluation.release.set()
    first.join(5), second.join(5)
    assert results == {"first": {"stored": 2}, "second": {"stored": 2}} and evaluation.peak == 1 and waited == [1]


def test_a_stop_ends_the_wait_without_evaluating_and_the_lock_is_free_afterwards(monkeypatch):
    evaluation, results, checks = Evaluation(monkeypatch), {}, []
    first = in_thread(results, "first")
    assert evaluation.entered.wait(5)

    def gate():
        checks.append(1)
        if len(checks) >= 2:
            raise Cancelled("stopped")
    second = in_thread(results, "second", gate=gate)
    second.join(8)
    assert results["second"] == {"stopped_by": "stopped while another evaluation was running", "stored": 0} and evaluation.peak == 1
    evaluation.release.set()
    first.join(5)
    assert fscore.score_proposals(None, [1], None) == {"stored": 1}              # nothing is left locked


def test_an_error_in_an_evaluation_releases_the_lock(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("boom")
    monkeypatch.setattr(fscore, "_score_proposals", broken)
    with pytest.raises(RuntimeError):
        fscore.score_proposals(None, [1], None)
    monkeypatch.setattr(fscore, "_score_proposals", lambda *a, **k: {"stored": 1})
    assert fscore.score_proposals(None, [1], None) == {"stored": 1}

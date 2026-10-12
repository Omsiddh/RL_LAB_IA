"""Check real checkpoint inference against previously recorded held-out results."""

import hashlib

import pytest

from main import DEFAULT_CHECKPOINT, console_output, run_demo


def test_saved_checkpoint_reproduces_both_report_case_studies():
    if not DEFAULT_CHECKPOINT.exists():
        pytest.skip("Saved checkpoint is supplied in the submission archive")
    digest_before = hashlib.sha256(DEFAULT_CHECKPOINT.read_bytes()).hexdigest()
    for workload_id, dqn_waiting, srtf_waiting in [(69, 10.2, 10.2), (201, 67 / 6, 61 / 6)]:
        workload, results = run_demo(workload_id=workload_id)
        assert results["DQN"]["mean_waiting_time"] == pytest.approx(dqn_waiting)
        assert results["SRTF"]["mean_waiting_time"] == pytest.approx(srtf_waiting)
        output = console_output(workload, results, workload_id, DEFAULT_CHECKPOINT)
        assert f"Held-out workload {workload_id}" in output
        for result in results.values():
            assert result["total_reward"] == -sum(p["waiting_time"] for p in result["processes"].values())
            assert all(p["completion_time"] is not None for p in result["processes"].values())
    assert hashlib.sha256(DEFAULT_CHECKPOINT.read_bytes()).hexdigest() == digest_before

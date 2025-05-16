from argparse import Namespace

from lib.tracking import ExperimentRun, load_run, save_run


def test_save_and_load_run(tmp_path):
    """Test that save_run and load_run work correctly with minimal input."""
    # Setup test data
    git_info = {"commit": "test-commit", "branch": "test-branch"}
    args = Namespace(env_name="test-env", batch_size=16)
    rewards = [[1.0, 2.0], [3.0, 4.0]]
    run_path = tmp_path / "test_run" / "experiment.json"

    # Save the run
    save_run(run_path, args, git_info, rewards)
    assert run_path.exists()

    # Test loading
    loaded_run = load_run(run_path)
    assert isinstance(loaded_run, ExperimentRun)
    assert loaded_run.parameters["env_name"] == "test-env"
    assert loaded_run.parameters["batch_size"] == 16
    assert loaded_run.git_info == git_info
    assert loaded_run.rewards == rewards

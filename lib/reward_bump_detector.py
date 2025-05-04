class RewardBumpDetector:
    """
    This class tracks training statistics, and if a reward bump is detected, it will trigger a recentering of the annealing process.

    The detector initially measures the average reward for the first min_good_samples, and writes this as the prev_best.
    The detector then resets itself, though this does not trigger a recentering.

    After the initial period, the detector tries to find n consecutive samples where the reward is greater than prev_best + bump_threshold.
    If it finds n > min_good_samples such samples, it will recenter the annealing process, and
    """

    def __init__(self, bump_threshold: float, initial_averaging_episodes: int, min_good_samples: int):
        self.bump_threshold = bump_threshold
        self.initial_averaging_episodes = initial_averaging_episodes
        self.min_good_samples = min_good_samples

        self.rewards: list[float] = []
        self.prev_best: float | None = None

    def update(self, reward: float) -> bool:
        """
        Update the detector with a new reward. Returns True if a bump is detected.
        """
        self.rewards.append(reward)

        if len(self.rewards) < self.initial_averaging_episodes:
            return False

        if self.prev_best is None:
            self.prev_best = sum(self.rewards[: self.initial_averaging_episodes]) / self.initial_averaging_episodes

        if len(self.rewards) > self.min_good_samples:
            min_consecutive_good = min(*self.rewards[-self.min_good_samples :])
            if min_consecutive_good > self.prev_best + self.bump_threshold:
                self.prev_best = min_consecutive_good
                return True

        return False

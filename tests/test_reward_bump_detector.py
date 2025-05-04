from lib.reward_bump_detector import RewardBumpDetector


def test_reward_bump_detector():
    detector = RewardBumpDetector(bump_threshold=0.1, initial_averaging_episodes=3, min_good_samples=2)
    assert not detector.update(0.7)
    assert not detector.update(0.5)
    assert not detector.update(0.3)

    # The baseline reward value is now set to 0.5
    assert not detector.update(0.7)  # Not yet, as min(0.3, 0.7) = 0.3 < 0.5 + 0.1
    assert not detector.update(0.5)  # Not yet, as min(0.7, 0.5) = 0.5 < 0.5 + 0.1
    assert not detector.update(0.7)  # Now, as min(0.5, 0.7) = 0.5 < 0.5 + 0.1
    assert detector.update(0.7)  # Bump detected, as min(0.7, 0.7) = 0.7 > 0.5 + 0.1

    # The baseline reward value is now set to 0.7
    assert not detector.update(0.9)  # Not yet, as min(0.7, 0.9) = 0.7 < 0.7 + 0.1
    assert detector.update(1.0)  # Bump detected, as min(0.9, 1.0) = 0.9 > 0.7 + 0.1

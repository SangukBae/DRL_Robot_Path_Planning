"""ROS-free unit tests for ObsTimeContext (actor-visible obs_state stacking).

Verifies the stacked-state contract used by both training and inference:
  * disabled / single-frame -> exact base [obs, agent] vector (87-D), dim stable
  * enabled stacked_state_dim matches the appended layout (obs-only vs full frame)
  * episode start uses FIRST-FRAME REPEAT padding (no zeros)
  * the current frame is ALWAYS first (state[:obs_dim]=current obs,
    state[obs_dim]=goal_dist) so baseline index-based readers keep working
  * history advances correctly and the stacked width is constant every step
"""

import numpy as np

from drl_agent.env.observation.obs_time_context import ObsTimeContext

O, A = 80, 7   # environment_dim (obs), agent_dim


def _obs(val):
    return np.full(O, float(val), dtype=np.float32)


def _agent(goal):
    a = np.zeros(A, dtype=np.float32)
    a[0] = goal
    return a


def test_disabled_returns_base_vector():
    otc = ObsTimeContext(O, A, enabled=False)
    assert otc.stacked_state_dim() == O + A
    s = otc.assemble(_obs(1.0), _agent(5.0))
    assert s.shape == (O + A,)
    assert s[O] == 5.0                       # goal_dist still at index O


def test_single_frame_is_treated_as_disabled():
    otc = ObsTimeContext(O, A, enabled=True, obs_frame_stack=1)
    assert otc.enabled is False
    assert otc.stacked_state_dim() == O + A


def test_stacked_dim_obs_only():
    otc = ObsTimeContext(O, A, enabled=True, obs_frame_stack=4,
                         stack_agent_state=False)
    assert otc.stacked_state_dim() == O * 4 + A      # 327


def test_stacked_dim_full_frame():
    otc = ObsTimeContext(O, A, enabled=True, obs_frame_stack=4,
                         stack_agent_state=True)
    assert otc.stacked_state_dim() == (O + A) * 4    # 348


def test_reset_uses_first_frame_repeat_not_zeros():
    otc = ObsTimeContext(O, A, enabled=True, obs_frame_stack=4)
    otc.reset(_obs(2.0), _agent(9.0))
    s0 = otc.assemble(_obs(2.0), _agent(9.0), advance=False)
    assert s0.shape == (O * 4 + A,)
    # current frame first: [obs(2.0) x80, agent(goal=9) x7], then 3 repeats of obs(2.0)
    assert np.all(s0[:O] == 2.0)
    assert s0[O] == 9.0
    assert np.all(s0[O + A:] == 2.0)          # history == first-frame repeat (not 0)


def test_history_advances_in_temporal_order():
    otc = ObsTimeContext(O, A, enabled=True, obs_frame_stack=3)  # current + 2 history
    otc.reset(_obs(0.0), _agent(0.0))
    # step 1: history is the seeded reset frame (0) twice
    s1 = otc.assemble(_obs(1.0), _agent(1.0))
    assert np.all(s1[:O] == 1.0) and s1[O] == 1.0
    assert np.all(s1[O + A:O + A + O] == 0.0)         # t-1 = reset frame
    assert np.all(s1[O + A + O:] == 0.0)              # t-2 = reset frame (repeat)
    # step 2: t-1 should now be frame 1, t-2 the reset frame
    s2 = otc.assemble(_obs(2.0), _agent(2.0))
    assert np.all(s2[:O] == 2.0)
    assert np.all(s2[O + A:O + A + O] == 1.0)         # t-1 = step-1 obs
    assert np.all(s2[O + A + O:] == 0.0)              # t-2 = reset obs
    # step 3: t-1 = frame 2, t-2 = frame 1 (reset frame dropped by maxlen)
    s3 = otc.assemble(_obs(3.0), _agent(3.0))
    assert np.all(s3[O + A:O + A + O] == 2.0)
    assert np.all(s3[O + A + O:] == 1.0)


def test_width_constant_every_step():
    otc = ObsTimeContext(O, A, enabled=True, obs_frame_stack=4)
    otc.reset(_obs(0.0), _agent(0.0))
    w = otc.stacked_state_dim()
    for t in range(10):
        s = otc.assemble(_obs(t), _agent(t))
        assert s.shape == (w,)


def test_full_frame_history_carries_agent_state():
    otc = ObsTimeContext(O, A, enabled=True, obs_frame_stack=2,
                         stack_agent_state=True)
    otc.reset(_obs(0.0), _agent(0.0))
    otc.assemble(_obs(1.0), _agent(11.0))            # pushes full frame 1
    s2 = otc.assemble(_obs(2.0), _agent(22.0))
    # layout: [obs2(80), agent2(7), frame1(87)] where frame1 = [obs1(80), agent1(7)]
    assert s2.shape == ((O + A) * 2,)
    assert s2[O] == 22.0                              # current agent goal
    assert s2[O + A + O] == 11.0                      # history frame's agent goal

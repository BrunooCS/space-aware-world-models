"""Simulation environments and rendering utilities."""


def make_simulator(env):
    """Interactive simulator ("dmlab") serving raw states, observations o_t, and scene maps m as recorded."""
    from src.envs.dm_lab_maze import DMLabSimulator  # load DeepMind Lab only when a simulator is requested
    return {"dmlab": DMLabSimulator}[env]()

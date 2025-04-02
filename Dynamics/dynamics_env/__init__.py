from gymnasium.envs.registration import register

register(
    id="Dynamics-v0",
    entry_point="dynamics_env.dynamics_env:Dynamics",
)
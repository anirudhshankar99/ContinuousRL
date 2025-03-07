from gymnasium.envs.registration import register

register(
    id="ContinuousCartPole-v0",
    entry_point="continuouscartpole_env.continuouscartpole_env:ContinuousCartPoleEnv",
)
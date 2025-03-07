from gymnasium.envs.registration import register

register(
    id="Squash-single-v0",
    entry_point="squash_env.squash_env:SquashEnv",
)

# from squash_env import SquashEnv
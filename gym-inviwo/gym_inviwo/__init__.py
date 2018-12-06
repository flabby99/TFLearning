from gym.envs.registration import register

register(
    id='inviwo-v0',
    entry_point='gym_foo.envs:InviwoEnv',
)
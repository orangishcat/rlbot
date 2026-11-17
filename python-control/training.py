import gymnasium as gym


class GdEnv(gym.Env): ...


gym.register(id="GeometryDash", entry_point=GdEnv, max_episode_steps=100)  # pyright: ignore[reportArgumentType]

gd_env = gym.make("GeometryDash")

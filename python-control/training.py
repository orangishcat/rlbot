from training import environment, model
from training.checkpoints import BestSurvivalCheckpoint, load_latest_checkpoint

CHECKPOINT_DIR = "training/results/checkpoints"


def main():
    print("Creating environment")
    gd_env = environment.create_env()
    print("Creating model")
    gd_model = model.create_model(gd_env)
    resumed_steps = load_latest_checkpoint(gd_model, CHECKPOINT_DIR)
    print("Beginning training")
    gd_model.learn(
        total_timesteps=3e6 - resumed_steps,
        progress_bar=True,
        reset_num_timesteps=resumed_steps <= 0,
        callback=BestSurvivalCheckpoint(CHECKPOINT_DIR, every=4096),
    )


if __name__ == "__main__":
    main()

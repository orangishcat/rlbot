from training import environment, model


def main():
    print("Creating environment")
    gd_env = environment.create_env()
    print("Creating model")
    gd_model = model.create_model(gd_env)
    print("Beginning training")
    gd_model.train()


if __name__ == "__main__":
    main()

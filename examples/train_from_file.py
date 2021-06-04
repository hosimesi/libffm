import ffm


def main():
    train_data = ffm.Dataset.read_ffm_data(
        "./bigdata.tr.txt",
        weights_path="./bigdata.iw.txt"
    )
    valid_data = ffm.Dataset.read_ffm_data(
        "./bigdata.te.txt",
    )

    model = ffm.train(
        train_data,
        valid_data=valid_data,
        auto_stop=True,
        auto_stop_threshold=3,
        quiet=False,
    )
    print("Best iteration:", model.best_iteration)

    with open("./model/prod-cvr.model", 'w') as f:
        model.dump_model(f, key_prefix="key")


if __name__ == '__main__':
    main()

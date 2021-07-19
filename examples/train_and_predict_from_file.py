import ffm


def main():
    # Prepare the data
    train_data = ffm.Dataset.read_ffm_data(
        "./bigdata.tr.txt",
        weights_path="./bigdata.iw.txt"
    )
    valid_data = ffm.Dataset.read_ffm_data(
        "./bigdata.te.txt",
    )

    # Train FFM model
    model = ffm.train(
        train_data,
        valid_data=valid_data,
        auto_stop=True,
        auto_stop_threshold=3,
        quiet=False,
    )
    print("Best iteration:", model.best_iteration)

    # Predict
    test_data = ffm.Dataset.read_ffm_data("./bigdata.te.txt")
    for x in test_data.data:
        pred_y = model.predict(x)
        print(pred_y)

    with open("./model/prod-cvr.model", 'w') as f:
        model.dump_model(f)


if __name__ == '__main__':
    main()

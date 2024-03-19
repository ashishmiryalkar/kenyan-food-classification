def data_loader(train, transforms, batch_size, shuffle, num_workers):
    dataset = KenyanFood13Dataset(
        data_root=data_root, train=train, image_shape=256, transform=transforms
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle
    )
    return loader


def get_mean_std():
    mean, std = [0.4837, 0.1418, -0.1151], [1.2656, 1.3080, 1.3844]
    return (mean, std)


def get_data(batch_size, data_root, num_workers=1, data_augmentation=False):
    # common transorms
    mean, std = get_mean_std()
    common_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    train_transforms = common_transforms
    if data_augmentation:
        train_transforms = transforms.Compose(
            [
                common_transforms,
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        )
    train_loader = data_loader(
        train=True,
        transforms=train_transforms,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = data_loader(
        train=False,
        transforms=common_transforms,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader


def save_model(
    model, device, model_dir="models", model_file_name="cat_dog_panda_classifier.pt"
):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, model_file_name)

    # make sure you transfer the model to cpu.
    if device == "cuda":
        model.to("cpu")

    # save the state_dict
    torch.save(model.state_dict(), model_path)

    if device == "cuda":
        model.to("cuda")

    return


def get_optimizer_and_scheduler(model):
    train_config = TrainingConfiguration()

    init_learning_rate = train_config.init_learning_rate

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=init_learning_rate, momentum=0.9)

    factor = 0.5  # reduce by factor 0.5
    patience = 2  # epochs
    threshold = 0.1
    verbose = True

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=factor,
        patience=patience,
        verbose=verbose,
        threshold=threshold,
    )

    return optimizer, scheduler
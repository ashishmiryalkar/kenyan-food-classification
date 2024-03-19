import time


def main(
    model,
    optimizer,
    tb_writer,
    scheduler=None,
    system_configuration=SystemConfiguration(),
    training_configuration=TrainingConfiguration(),
    data_augmentation=True,
):

    # system configuration
    setup_system(system_configuration)

    # batch size
    batch_size_to_set = training_configuration.batch_size
    # num_workers
    num_workers_to_set = training_configuration.num_workers
    # epochs
    epoch_num_to_set = training_configuration.epochs_count

    # if GPU is available use training config,
    # else lowers batch_size, num_workers and epochs count
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        batch_size_to_set = 16
        num_workers_to_set = 4

    # data loader
    train_loader, test_loader = get_data(
        batch_size=batch_size_to_set,
        data_root=training_configuration.data_root,
        num_workers=num_workers_to_set,
        data_augmentation=data_augmentation,
    )

    # Update training configuration
    training_configuration = TrainingConfiguration(
        device=device, batch_size=batch_size_to_set, num_workers=num_workers_to_set
    )

    # send model to device (GPU/CPU)
    model.to(training_configuration.device)

    best_loss = torch.tensor(np.inf)

    # epoch train/test loss
    epoch_train_loss = np.array([])
    epoch_test_loss = np.array([])

    # epch train/test accuracy
    epoch_train_acc = np.array([])
    epoch_test_acc = np.array([])

    # Calculate Initial Test Loss
    init_val_loss, init_val_accuracy = validate(
        training_configuration, model, test_loader
    )
    print(
        "Initial Test Loss : {:.6f}, \nInitial Test Accuracy : {:.3f}%\n".format(
            init_val_loss, init_val_accuracy * 100
        )
    )

    # trainig time measurement
    t_begin = time.time()
    for epoch in range(training_configuration.epochs_count):

        # Train
        train_loss, train_acc = train(
            training_configuration, model, optimizer, train_loader, epoch, tb_writer
        )

        epoch_train_loss = np.append(epoch_train_loss, [train_loss])

        epoch_train_acc = np.append(epoch_train_acc, [train_acc])

        elapsed_time = time.time() - t_begin
        speed_epoch = elapsed_time / (epoch + 1)
        speed_batch = speed_epoch / len(train_loader)
        eta = speed_epoch * training_configuration.epochs_count - elapsed_time

        print(
            "Elapsed {:.2f}s, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s".format(
                elapsed_time, speed_epoch, speed_batch, eta
            )
        )

        # Validate
        if epoch % training_configuration.test_interval == 0:
            current_loss, current_accuracy = validate(
                training_configuration, model, test_loader
            )

            epoch_test_loss = np.append(epoch_test_loss, [current_loss])

            epoch_test_acc = np.append(epoch_test_acc, [current_accuracy])

            tb_writer.add_scalar("Loss/Validation", current_loss, epoch)
            tb_writer.add_scalar("Accuracy/Validation", current_accuracy, epoch)

            # add scalars (loss/accuracy) to tensorboard
            tb_writer.add_scalars(
                "Loss/train-val",
                {"train": train_loss, "validation": current_loss},
                epoch,
            )
            tb_writer.add_scalars(
                "Accuracy/train-val",
                {"train": train_acc, "validation": current_accuracy},
                epoch,
            )

            if current_loss < best_loss:
                best_loss = current_loss
                print("Model Improved. Saving the Model...\n")
                save_model(model, device=training_configuration.device)

            # scheduler step/ update learning rate
        if scheduler is not None:
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                scheduler.step(current_loss)
                print("Bad Epochs:{}".format(scheduler.num_bad_epochs))
                print("last LR = {}".format(scheduler._last_lr))
            else:
                scheduler.step()

    print(
        "Total time: {:.2f}, Best Loss: {:.3f}".format(time.time() - t_begin, best_loss)
    )

    return model, epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc

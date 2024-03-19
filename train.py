def train(
    train_config: TrainingConfiguration,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    epoch_idx: int,
    tb_writer: SummaryWriter,
) -> None:

    # change model in training mood
    model.train()

    # to get batch loss
    batch_loss = np.array([])

    # to get batch accuracy
    batch_acc = np.array([])

    for batch_idx, (data, target) in enumerate(train_loader):

        # clone target
        indx_target = target.clone()
        # send data to device (its is medatory if GPU has to be used)
        data = data.to(train_config.device)
        # send target to device
        target = target.to(train_config.device)

        # reset parameters gradient to zero
        optimizer.zero_grad()

        # forward pass to the model
        output = model(data)
        # print(output.size(), 'from train output.size()')
        # cross entropy loss
        loss = F.cross_entropy(output, target)
        # loss = Variable(loss, requires_grad = True)
        # find gradients w.r.t training parameters
        loss.backward()
        # Update parameters using gardients
        optimizer.step()

        batch_loss = np.append(batch_loss, [loss.item()])

        # Score to probability using softmax
        prob = F.softmax(output, dim=1)

        # get the index of the max probability
        pred = prob.data.max(dim=1)[1]

        # correct prediction
        correct = pred.cpu().eq(indx_target).sum()

        # accuracy
        acc = float(correct) / float(len(data))

        batch_acc = np.append(batch_acc, [acc])

        if batch_idx % train_config.log_interval == 0 and batch_idx > 0:

            total_batch = (
                epoch_idx * len(train_loader.dataset) / train_config.batch_size
                + batch_idx
            )
            tb_writer.add_scalar("Loss/train-batch", loss.item(), total_batch)
            tb_writer.add_scalar("Accuracy/train-batch", acc, total_batch)

    epoch_loss = batch_loss.mean()
    epoch_acc = batch_acc.mean()
    tb_writer.add_scalar("Loss/train", epoch_loss.item(), total_batch)
    tb_writer.add_scalar("Accuracy/train", epoch_acc.item(), total_batch)

    print(
        "Epoch: {} \nTrain Loss: {:.6f} Acc: {:.4f}".format(
            epoch_idx, epoch_loss, epoch_acc
        )
    )
    return epoch_loss, epoch_acc

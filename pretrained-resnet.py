def pretrained_resnet18(transfer_learning=True, num_class=13):
    resnet = models.resnet50(pretrained=True)

    if transfer_learning:
        for param in resnet.parameters():
            param.requires_grad = False

    last_layer_in = resnet.fc.in_features
    resnet.fc = nn.Sequential(
        nn.Linear(last_layer_in, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 13)
    )

    return resnet

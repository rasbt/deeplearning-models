test_dataloader = data_module.test_dataloader()
acc = torchmetrics.Accuracy()

for batch in test_dataloader:
    features, true_labels = batch

    with torch.no_grad():
        logits = lightning_model(features)

    predicted_labels = torch.argmax(logits, dim=1)
    acc(predicted_labels, true_labels)

predicted_labels[:5]

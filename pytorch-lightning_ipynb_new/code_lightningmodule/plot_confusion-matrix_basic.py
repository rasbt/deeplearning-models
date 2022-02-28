from torchmetrics import ConfusionMatrix
from helper_plotting import plot_confusion_matrix


cmat = ConfusionMatrix(num_classes=len(class_dict))

for x, y in test_dataloader:

    with torch.no_grad():
        pred = lightning_model(x)
    cmat(pred, y)

cmat_tensor = cmat.compute()

plot_confusion_matrix(conf_mat=cmat_tensor.numpy(), class_names=class_dict.values())
plt.show()

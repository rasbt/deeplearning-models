from torchmetrics import ConfusionMatrix
import matplotlib
from mlxtend.plotting import plot_confusion_matrix


cmat = ConfusionMatrix(num_classes=len(class_dict))

for x, y in test_dataloader:

    with torch.no_grad():
        pred = lightning_model(x)
    cmat(pred, y)

cmat_tensor = cmat.compute()
cmat = cmat_tensor.numpy()

fig, ax = plot_confusion_matrix(
    conf_mat=cmat,
    class_names=class_dict.values(),
    norm_colormap=matplotlib.colors.LogNorm()  
    # normed colormaps highlight the off-diagonals 
    # for high-accuracy models better
)

plt.show()
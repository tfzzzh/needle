import needle as ndl
from models import ResNet9
from simple_ml import train_cifar10, evaluate_cifar10

device = ndl.default_device()
dataset = ndl.data.CIFAR10Dataset("../data/cifar-10-batches-py", train=True)
dataloader = ndl.data.DataLoader(\
         dataset=dataset,
         batch_size=128,
         shuffle=True,)
model = ResNet9(device=device, dtype="float32")
train_cifar10(model, dataloader, n_epochs=10, optimizer=ndl.optim.Adam,
      lr=0.001, weight_decay=0.001)

dataset_valid = ndl.data.CIFAR10Dataset("../data/cifar-10-batches-py", train=False)
dataloader_valid = ndl.data.DataLoader(
    dataset = dataset,
    batch_size = 1000
)
valid_error, valid_loss = evaluate_cifar10(model, dataloader_valid)

print(f"valid_error = {valid_error}, valid_loss = {valid_loss}")
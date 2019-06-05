import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import os

os.environ['TORCH_MODEL_ZOO'] = '/sgoinfre/goinfre/Perso/malluin'

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.486, 0.456, 0.406), (0.229, 0.224, 0.225))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=8,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def img_show(image):
	npimage = image.numpy()
	print(image.shape)
	plt.imshow(np.transpose(npimage, (1, 2, 0)))
	# print(np.transpose(npimage, (1, 0, 2)).shape)
	plt.show()

# dataiter = iter(trainloader)
# images, labels = dataiter.next()

# show images
# img_show(torchvision.utils.make_grid(images))

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 25)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params = net.parameters(), lr=0.01, momentum=0.9)

var = input("Load existing Model? (y/n) (default:y)\n")

if os.path.isfile("model_save.txt") and var != "n":
	# net.load_state_dict(torch.load("model_save.txt"))
	net = models.vgg19_bn(pretrained=True)
	print("loading successful")



var2 = input("Train Model? (y/n) (default:y)\n")
if var2 != "n":
	epoch_nb = input("How many epochs? (default:2)\n")
	if epoch_nb.isdigit() == False:
		epoch_nb = 2
	else:
		epoch_nb = int(epoch_nb)
	for epoch in range(epoch_nb):
		running_loss = 0.0
		for i, data in enumerate(trainloader):
			inputs, labels = data
			# print(inputs.shape, labels.shape)
			optimizer.zero_grad()
			output = net(inputs)
			loss = criterion(output, labels)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
			if i % 200 == 199:
				print("epoch: {} loss {}".format(epoch, running_loss / 200))
				running_loss = 0
	print('Training Finished')

var = input("Save model ? (y/n)(default:y)\n")
if var != "n":
	torch.save(net.state_dict(), "model_save.txt")

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
net.eval()
# img_show(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
# outputs = net(images)
# _, predicted = torch.max(outputs, 1)
# print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels)
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

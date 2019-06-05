import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()
		#convolutions de 5
		self.conv1 = nn.Conv2d(1, 6, 5)
		self.conv2 = nn.Conv2d(6, 16, 5)
		#fully connected layers
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		#convolution de 5 => passe l'image de 32 a 28 et 6 channels
		#relu sur la couche de convolution change pas la taille
		#pooling de (2 ,2) => passe l'image de 28 a 14, tj 6 channels
		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
		#convolution de 5 => passe l'image de 14 a 10 et 16 channels
		#relu sur la couche de convolution change pas la taille
		#pooling de 2 // (2,2) et 2 sont pareils => passe l'image de 10 a 5 et 16 channels
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		#resize de 5 * 5 * 16 a 400
		x = x.view(-1, self.num_flat_features(x))
		#passe dans la FC1 + relu
		x = F.relu(self.fc1(x))
		#passe dans la FC2 + relu
		x = F.relu(self.fc2(x))
		#passe dans la FC3 => classifie en 10 categories
		x = self.fc3(x)
		return x

	def num_flat_features(self, x):
		size = x.size()[1:]
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

net = Net()
print(net)
params = list(net.parameters())
# print(params[0].size())
# print(len(params))
# for param in params:
# 	print(param.size())
# print(params)
input = torch.randn(1, 1, 32, 32)
output = net(input)

# print(output)
net.zero_grad()
# out.backward(torch.randn(1, 10))

target = torch.randn(10)
target = target.view(1, -1)
criterion = nn.MSELoss()

loss = criterion(output, target)
# print(net.conv1.bias.grad)
# print(loss)
# print(loss.grad_fn)
# print(loss.grad_fn.next_functions[0][0])
# print(loss.grad_fn.next_functions[0][0].next_functions[0][0])

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

learning_rate = 0.01
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
iter = 1000
for i in range(iter):
	output = net(input)
	loss = criterion(output, target)
	loss.backward()
	optimizer.step()
	print(net.conv1.bias.grad)

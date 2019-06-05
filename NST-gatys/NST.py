import copy
import os
import sys
import time
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

parser = argparse.ArgumentParser(description='Laplacian-Steered Neural Style Transfer')
parser.add_argument('--steps', type=int, default=800, metavar='N',
                    help='number of steps to train (default: 800)')
parser.add_argument('--sw', type=int, default=1000000, metavar='N',
                    help='Style weight (default: 1000000)')
parser.add_argument('--cw', type=int, default=1, metavar='N',
                    help='Content weight (default: 20)')
parser.add_argument('--style', type=str, default='starry_night.jpg', metavar='X.jpg',
                    help='Style image to use')
parser.add_argument('--content', type=str, default='lozere.jpg', metavar='X.jpg',
                    help='Content image to use')
parser.add_argument('--random', type=int, default=0 , metavar='0-1',
                    help='Initialize generated image at random (default 0: False)')
parser.add_argument('--res', type=int, default=512 , metavar='N',
                    help='Generate image at a resolution of NxN (default 512)')
dargs = vars(parser.parse_args())

print(dargs['steps']) 

#Timing program
begin = time.time()

#Defines directory for pretrained models download
# os.environ['TORCH_MODEL_ZOO'] = '/sgoinfre/goinfre/Perso/malluin'

#If GPU available use bigger image size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count():
	print("Using GPU: {}\n".format(torch.cuda.get_device_name(device)))
else:
	print("Using CPU")
imsize = dargs['res'] if torch.cuda.is_available() else 256

#Load input image to a Tensor and resize to desired shape, in line with device computational power.
loader = transforms.Compose([transforms.Resize((imsize, imsize)), transforms.ToTensor()])
#Unloader to transform back tensors to pillow images in order to save and plot output images.
unloader = transforms.ToPILImage()

def image_loader(image_name):
    image_load = Image.open(image_name)
    old_size = image_load.size
    image_load = loader(image_load).unsqueeze(0)
    new_size = (image_load.size()[2], image_load.size()[3])
    print("Rescaling from {}x{} to {}x{}".format(old_size[0], old_size[1], new_size[0], new_size[1]))
    if image_load.size()[1] == 1:
        image_load = image_load.repeat(1,3,1,1)
    show_image(image_load, 1)
    return image_load.to(device, torch.float)

def show_image(tensor, i, row = 1, col = 2):
	tensor = tensor.squeeze(0)
	image = unloader(tensor)
	plt.subplot(row, col, i)
	plt.imshow(image)

#Define content and image_style, if provided use arguments in command line to define images
argc = len(sys.argv)
default_content = "../content_images/" + dargs['content']
default_style = "../style_images/"  + dargs['style']
style_img_path = default_style
content_img_path = default_content
print("Content image: {} \nStyle image: {} \n".format(content_img_path, style_img_path))

#Load both style and content images, save original size for rescaling output
style_img = image_loader(style_img_path)
content_img = image_loader(content_img_path)
old_size = reversed(Image.open(content_img_path).size)
old_size = tuple(old_size)

print(style_img.size(), content_img.size())
# print(style_img.size(), content_img.size())
assert style_img.size() == content_img.size()

#loader to resize output image to its original size
load_resize = transforms.Compose([transforms.Resize(old_size), transforms.ToTensor()])
def scale_up_save(tensor, i):
	resized = tensor.squeeze(0)
	resized = unloader(resized.cpu())
	resized = load_resize(resized)
	resized = resized.squeeze(0)
	resized = unloader(resized)
	resized.save('results_NST/output' + str(i) +'.jpg')
	print('saving results_NST/output' + str(i) +'.jpg...')

#Optional show style and content images
# show_image(style_img.cpu(), 1)
# show_image(content_img.cpu(), 2)
# plt.show()

#Content loss function which inherits from pytorch nn.nodule
#This function defines the mean squared error between the generated output and the input content image.
#It is computed at each iteration and is used in the loss function.
class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

#Calculation of gram matrix to compute the style loss.
def gram_matrix(input):
	a, b, c, d = input.size()
	# a=batch size(=1) / b=number of feature maps / (c,d)=dimensions of a f. map (N=c*d)
	features = input.view(a * b, c * d)
	#Torch.mm computes the gram product of the matrix by performing a dot product between the matrix and its transpose
	G = torch.mm(features, features.t())
	# we 'normalize' the values of the gram matrix by dividing by the number of element in each feature maps.
	return G.div(a * b * c * d)

#Style loss function which inherits from pytorch nn.nodule
#This function calculate the mean squared difference between the gram matrix of the input and the gram matrix of the generated output image.
#It is computed at each iteration and is used in the loss function paired with the content loss.
class StyleLoss(nn.Module):

    def __init__(self, target,):
        super(StyleLoss, self).__init__()
        # we 'detach' the target content from the tree used to dynamically compute the gradient: this is a stated value, not a variable.
		# Otherwise the forward method of the criterion will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(gram_matrix(input), gram_matrix(self.target))
        return input

#Loading of a pretrained vgg19 model, the model parameters are downloaded the first time this code is run.
cnn = models.vgg19(pretrained = True).features.to(device).eval()
#Normalization of input image is necessary for this pretrained network.
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
	def __init__(self, mean, std):
		super(Normalization, self).__init__()
		self.mean = torch.tensor(mean).view(-1, 1, 1)
		self.std = torch.tensor(std).view(-1, 1, 1)

	def forward(self, input):
		return ((input - self.mean) / self.std)

#Defining the content and style layers that will be used to compute the loss function. Best results are achieved with early conv layers.
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
# style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5', 'conv_6', 'conv_7', 'conv_8', 'conv_9', 'conv_10', 'conv_11', 'conv_12', 'conv_13', 'conv_14', 'conv_15', 'conv_16']

#Defining model
def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img, content_layers=content_layers_default,
                               style_layers=style_layers_default):
	#Copies instead of referencing CNN parameters, it seems that it doesnt have much effect on memory usage
	cnn = copy.deepcopy(cnn)
	normalization = Normalization(normalization_mean, normalization_std).to(device)
	content_losses = []
	style_losses = []
	# assuming that cnn is a nn.Sequential, so we make a new nn.Sequential to put in modules that are supposed to be activated sequentially
	model = nn.Sequential(normalization)
	i = 0
	#Iterating through pretrained CNN model layers, we add each layer to our own model and build additionnal style and content layers depending on previously defined layers
	for layer in cnn.children():
		if isinstance(layer, nn.Conv2d):
			i += 1
			name = 'conv_{}'.format(i)
			# print(name)
		elif isinstance(layer, nn.ReLU):
			name = 'relu_{}'.format(i)
			# The in-place version doesn't play very nicely with the ContentLoss
			# and StyleLoss we insert below. So we replace with out-of-place
			# ones here.
			layer = nn.ReLU(inplace=False)
		elif isinstance(layer, nn.MaxPool2d):
			name = 'pool_{}'.format(i)
		elif isinstance(layer, nn.BatchNorm2d):
			name = 'bn_{}'.format(i)
		else:
			raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
		#Add CNN layer to our model
		model.add_module(name, layer)
		if name in content_layers:
            #Add content loss layer to our model which is defined by passing through the content loss function the input image which has been processed by previous layers of the CNN
			target = model(content_img).detach()
			content_loss = ContentLoss(target)
			model.add_module("content_loss_{}".format(i), content_loss)
			content_losses.append(content_loss)

		if name in style_layers:
			#Add style loss layer to our model
			target_feature = model(style_img).detach()
			style_loss = StyleLoss(target_feature)
			model.add_module("style_loss_{}".format(i), style_loss)
			style_losses.append(style_loss)
    #Removing additional layers
	for i in range(len(model) - 1, -1, -1):
		if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
			break
	model = model[:(i + 1)]
	return model, style_losses, content_losses

#Define input image as a clone of content image to speed up convergence
if dargs['random'] == 0:
    input_img = content_img.clone()
else:
    input_img = torch.randn(content_img.data.size(), device=device)

#Define LBFGS optimizer which converges quickly and efficiently compared to other algorithms.
#Adam uses slightly less memory but is much slower and has trouble converging.
def get_input_optimizer(input_img):
	optimizer = optim.LBFGS([input_img.requires_grad_()], lr = 1, max_iter = 20, history_size = 10)
	# optimizer = optim.Adam([input_img.requires_grad_()], lr = 1e-2)
	return optimizer

def run_style_transfer(cnn, normalization_mean, normalization_std, content_img, style_img, input_img, 
						num_steps=dargs['steps'], style_weight=dargs['sw'], content_weight=dargs['cw']):
	print('\n\nBuilding the style transfer model..')
	model, style_losses, content_losses = get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img)
	###var = input("\n\nLoad existing Model? (y/n) (default:y)\n")
	### if  var != "n":
	### 	if os.path.isfile("model_save.txt"):
	### 		model.load_state_dict(torch.load("model_save.txt"))
	### 		print("loading successful\n")
	### 	else:
	### 		print("No data available\n")
	optimizer = get_input_optimizer(input_img)
	print('Optimizing....')
	run = [0]
	#Iterate on num_steps
	### var2 = input("\nTrain Model? (y/n) (default:y)\n")
	### if var2 == "n":
	### 	num_steps = 100
	while run[0] <= num_steps:
		def closure():
			# correct the values of updated input image
			input_img.data.clamp_(0, 1)

			#??
			optimizer.zero_grad()
			#Run the input image through the model
			model(input_img)

			#Compute sum of style and content losses.
			style_score = 0
			content_score = 0
			for sl in style_losses:
				style_score += sl.loss
			for cl in content_losses:
				content_score += cl.loss
			style_score *= style_weight
			content_score *= content_weight
			loss = style_score + content_score

			#Compute gradients based on loss function
			loss.backward()
			run[0] += 1
			if run[0] % 5 == 4:
				print("run {}:".format(run))
				print('Style Loss : {:4f} Content Loss: {:4f}'.format(
				style_score.item(), content_score.item()))
				print('Memory usage: {} Mo'.format(round(torch.cuda.memory_allocated(device) / 1000000, 2)))
				print('Memory cached: {} Mo'.format(round(torch.cuda.memory_cached(device) / 1000000, 2)))
				print()
				#Save image every 50 iterations
				if run[0] % 50 == 49:
					tmp = copy.deepcopy(input_img.data.clamp_(0, 1))
					scale_up_save(tmp, run[0] / 5)
			return style_score + content_score
		optimizer.step(closure)
	input_img.data.clamp_(0, 1)
	print('Training Finished')
	### save = input("Save model ? (y/n)(default:y)\n")
	### if save != "n":
	### 	torch.save(model.state_dict(), "model_save.txt")
	return input_img

output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std, content_img, style_img, input_img)
i = 1
#Show final output and print time
plt.figure(figsize=(20,20))
show_image(output.detach().cpu(), 1, 1, 1)
print("Program has taken {}s to compute.".format(round(time.time() - begin, 2)))
plt.show()
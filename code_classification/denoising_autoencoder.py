import torch
from torch import nn
import numpy as np
import os
import time
from collections import OrderedDict
from max_pool_out_size import get_h_w_out, get_h_w_tran_out
from spacetimecnn import get_data, get_loaders, make_optimizer
from torch.autograd import Variable
from utils import root_dir
random_state = random_seed = 99 

#shape in, padding, dilation, kernel, stride
#print(get_h_w_out((25,513),(0,0),(0,0),(1,3),(1,3)))

def is_conv_layer(layer):
    if "conv" in type(layer).__name__.lower():  
        return True
    else:
        return False

def is_pool_layer(layer):
    if "pool" in type(layer).__name__.lower():  
        return True
    else:
        return False

def is_transpose_layer(layer):
    if "transpose" in type(layer).__name__.lower():
        return True
    else:
        return False

def is_nonlinear_layer(layer):
    if "nonlin" in type(layer).__name__.lower():
        return True
    else:
        return False


def make_layer_dict(layer_list, input_size, super_layer, output=None, linear_shrink=0.75):
    """Takes in list of layer classes, returns an OrderedDict of layers"""

    layer_dict = OrderedDict()

    size_list = []

    for i, layer in enumerate(layer_list):
        if is_transpose_layer(layer):
            output_size = get_h_w_tran_out(input_size, stride=layer.stride, kernel=layer.kernel_size, padding=layer.padding, dilation=layer.dilation)
            print(layer.stride, layer.kernel_size, layer.padding, layer.dilation, layer.padding_mode)
        
        elif is_conv_layer(layer): #If the layer is convolutional, we calculate what the size of its output will be.
            output_size = get_h_w_out(input_size, stride=layer.stride, kernel=layer.kernel_size, padding=layer.padding, dilation=layer.dilation)
            print(layer.stride, layer.kernel_size, layer.padding, layer.dilation, layer.padding_mode)

        elif is_pool_layer(layer): #If the layer is convolutional, we calculate what the size of its output will be.
            output_size = get_h_w_out(input_size, stride=layer.stride, kernel=layer.kernel_size, padding=layer.padding, dilation=layer.dilation)
        elif is_nonlinear_layer(layer):
            output_size = input_size
            
        elif type(layer) == Flatten:
            output_size = (input_size[0] * input_size[1],1)

        elif type(layer) == GaussianLayer:
            output_size = input_size
        
        else:                     #Otherwise, we assume that it's feedforward and give the output size as equal to the input size.
            #print("it's a feedforward")
            if i+1 == len(layer_list) and output != None:
                output_size = output
            else:
                output_size = (int(input_size[0] * linear_shrink),1)
            if output_size[0] * output_size[1] == 0:
                return(layer_dict, size_list, False)
            layer = layer(input_size[0], output_size[0])

        layer_dict[type(layer).__name__.lower() + str(super_layer)] = layer

        size_list.append(output_size)

        input_size = output_size

    return(layer_dict, size_list, True)


def make_full_dict(layer_list, input_size, output=(128,1), linear_shrink=0.75):

    list_of_dicts = []
    full_size_list = []

    for i, layer_type in enumerate(layer_list):
        if type(layer_type[0]) == OrderedDict:
            list_of_dicts.append(layer_type[0])
            full_size_list += layer_type[1]
            continue
        layers = layer_type[0]
        n_layers = layer_type[1]
        for layer_n in range(n_layers):
            if i+1 == len(layer_list) and layer_n+1 == n_layers:
                layer_dict, size_list, add_more = make_layer_dict(layers, input_size, layer_n, linear_shrink=linear_shrink, output=output)
            else:
                layer_dict, size_list, add_more = make_layer_dict(layers, input_size, layer_n, linear_shrink=linear_shrink)
            full_size_list += size_list
            list_of_dicts.append(layer_dict)
            if not add_more:
                print("Did not finish")
                break
            input_size = size_list[-1]

    full_dict = OrderedDict()

    for dic in list_of_dicts:
        for k in dic.keys():
            full_dict[k] = dic[k]

    return(full_dict, full_size_list, add_more)

#def make_full_dict(super_layers_dict, input_size):


#def make_full_dict(layer_list, input_size, n_layers):

 #   list_of_dicts = []
#    full_size_list = []

#    for layer_n in range(n_layers):
#        layer_dict, size_list = make_layer_dict(layer_list, input_size, layer_n)
#        input_size = size_list[-1]
#        full_size_list += size_list
#        list_of_dicts.append(layer_dict)

#    full_dict = OrderedDict()

#    for dic in list_of_dicts:
#        for k in dic.keys():
#            full_dict[k] = dic[k]

#    return(full_dict, full_size_list)


def copy_weights(self, encoder: torch.nn.Linear, decoder: torch.nn.Linear) -> None:
    """
    Utility method to copy the weights of self into the given encoder and decoder, where
    encoder and decoder should be instances of torch.nn.Linear.
    :param encoder: encoder Linear unit
    :param decoder: decoder Linear unit
    :return: None
    """
    encoder.weight.data.copy_(self.encoder_weight)
    encoder.bias.data.copy_(self.encoder_bias)
    decoder.weight.data.copy_(self.decoder_weight)
    decoder.bias.data.copy_(self.decoder_bias)


class GaussianLayer(nn.Module):
    def __init__(self, noise_level=0.5):
        super(GaussianLayer, self).__init__()
        
        #self.embedding_dimension = embedding_dimension
        #self.hidden_dimension = hidden_dimension
        self.noise_level = noise_level

    def forward(self, x): 
        try:
            noise = np.random.normal(loc=0.0, scale=self.noise_level, size=x.size()).astype(np.float32)
        except AttributeError:
            import pdb; pdb.set_trace()
        x += torch.Tensor(noise)
        return x


class Flatten(nn.Module):
    def forward(self, input):
        #print(input.shape)
        return input.view(input.size(0), -1)


class NonlinElu(nn.Module):
    def __init__(self):
        super(NonlinElu, self).__init__()
        
        #self.embedding_dimension = embedding_dimension
        #self.hidden_dimension = hidden_dimension
        self.noise_level = noise_level

    def forward(self, input):
        #print(input.shape)
        return nn.ELU(input)

class NonlinRelu(nn.Module):
    def __init__(self):
        super(NonlinRelu, self).__init__()
        self.nonlinearity = nn.ReLU()

    def forward(self, input):
        #print(input.shape)
        return self.nonlinearity(input)

class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_size: (int,int), output_size: (int,int), noise_level=0.5):
        super(DenoisingAutoencoder, self).__init__()

        self.embedding_dimension = embedding_dimension
        self.hidden_dimension = hidden_dimension
        self.noise_level = noise_level

        self.noise_layer = GaussianLayer(input_size, noise_level=noise_level)
        self.conv_layer = CNN2D(input_channels, output_channels, kernel_size)


class ArbitraryNet(nn.Module):
    def __init__(self, layers_dict: OrderedDict):
        super(ArbitraryNet, self).__init__()

        self.layers = nn.Sequential(layers_dict)
    def forward(self, x):
        return self.layers(x)


def make_reverse_layer(conv_layer, input_size):

    layer = conv_layer[0]
    n_layers = conv_layer[1]

    full_size_list = [input_size]

    for layer_n in range(n_layers):
        layer_dict, size_list, add_more = make_layer_dict(layer, input_size, layer_n, linear_shrink=1)
        full_size_list += size_list
        input_size = size_list[-1]

    out_size_list = list(reversed(full_size_list))[1:]

    conv = layer[0] 

    kernel = conv.kernel_size
    padding = conv.padding
    dilation = conv.dilation
    stride = conv.stride

    out_dict = OrderedDict()

    for i in range(n_layers):

        calc_out_size = get_h_w_tran_out(input_size, stride=stride, kernel=kernel, padding=padding, dilation=dilation)
        output_size = out_size_list[i]
        output_padding = tuple(np.array(output_size) - np.array(calc_out_size))
        print(output_padding)
        out_dict[nn.ConvTranspose2d.__name__.lower() + str(i)] = nn.ConvTranspose2d(1,1,kernel, stride=stride, padding=padding, dilation= dilation, output_padding=output_padding)
        input_size = output_size


    return (out_dict, out_size_list)


def pretrain(autoencoder, train_loader, val_loader, n_epochs=30, learning_rate=0.065):	
	
        #Print all of the hyperparameters of the training iteration:
	print("===== HYPERPARAMETERS =====")
	print("epochs=", n_epochs)
	print("learning_rate=", learning_rate)
	print("=" * 30)
	
	#Get training data
	n_batches = len(train_loader)
	
	#Create our loss and optimizer functions
	optimizer = make_optimizer(autoencoder, learning_rate)
	loss = nn.MSELoss()	

	#Time for printing
	training_start_time = time.time()
	
	#Loop for n_epochs
	for epoch in range(n_epochs):
		
		running_loss = 0.0
		print_every = n_batches // 10
		start_time = time.time()
		total_train_loss = 0
		
		for i, data in enumerate(train_loader, 0):
			
			#Get inputs
			inputs, _ = data
			
			#Wrap them in a Variable object
			inputs = Variable(inputs)
			
			#Set the parameter gradients to zero
			optimizer.zero_grad()
			
			#Forward pass, backward pass, optimize
			outputs = autoencoder(inputs)
			loss_size = loss(outputs, inputs)
			loss_size.backward()
			optimizer.step()
			
			#Print statistics
			running_loss += loss_size.data.item()
			total_train_loss += loss_size.data.item()
			
			#Print every 10th batch of an epoch
			if (i + 1) % (print_every + 1) == 0:
				print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(

						epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
				#Reset running loss and time
				running_loss = 0.0
				start_time = time.time()
			
		#At the end of the epoch, do a pass on the validation set
		total_val_loss = 0
		for inputs, labels in val_loader:
			
			#Wrap tensors in Variables
			inputs, labels = Variable(inputs), Variable(labels)
			
			#Forward pass
			val_outputs = autoencoder(inputs)
			val_loss_size = loss(val_outputs, labels)
			total_val_loss += val_loss_size.data.item()
			
		print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
	
	train_time = time.time() - training_start_time

	print("Training finished, took {:.2f}s".format(train_time))

	return(total_val_loss, train_time)



if __name__ == "__main__":

    in_size = (256,128)
    out_size = (100,1)


    conv_layer = nn.Conv2d(1,1,(2,10), stride=(2,2))

    reverse_layer = make_reverse_layer(([conv_layer],3), in_size)

    all_layers = [([GaussianLayer(), conv_layer, NonlinRelu()],3),
            reverse_layer
                ]
    full_dict, size_list, finish = make_full_dict(all_layers, in_size, output=out_size, linear_shrink=0.90)
    template = "{0:20} {1:90} {2:10}"
    print("IN SIZE: {0}".format(in_size))
    print(template.format("NAME","LAYER","OUT SIZE"))
    removebad = str.maketrans("","","\n")
    for i, layer in enumerate(full_dict):
        print(template.format(layer, str(full_dict[layer]).translate(removebad), str(size_list[i])))


    databatch = torch.randn(100,1,*in_size)
    
    myarb = ArbitraryNet(full_dict)

    if finish:
        out = myarb.forward(databatch)
    else:
        print("DID NOT FINISH")
    print("in shape", databatch.shape)
    print("out shape", out.shape)

    layer0 = ArbitraryNet(OrderedDict([(entry, full_dict[entry]) for entry in full_dict if "0" in entry]))

    mode = "dep"


    input_path = os.path.join(root_dir, "single_bandpass", "windows", "EEG")

    target_path = os.path.join(root_dir, "single_bandpass","windows","Envelopes")	



    train_dataset, val_dataset, test_dataset = get_data(input_path, target_path, mode=mode)	

    train_loader, val_loader, test_loader = get_loaders(train_dataset, val_dataset, test_dataset, batch_size=100)

    pretrain(layer0, train_loader, val_loader)




    import pdb; pdb.set_trace()

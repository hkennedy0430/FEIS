
def get_height_out(height_in,padding,dilation,kernel_height,stride):
    numerator = height_in + (2*padding) - dilation * (kernel_height - 1) -1
    denominator = stride
    return(numerator/denominator) + 1

#def get_h_w_out(shape_in,kernel=(3,3),padding=(0,0),dilation=(0,0),stride=(1,1)):
#    height = get_height_out(shape_in[0],padding[0],dilation[0],kernel[0],stride[0])
#    width = get_height_out(shape_in[1],padding[1],dilation[1],kernel[1],stride[1])
#    return(int(height),int(width))

def get_height_out_tran(height_in,padding,dilation,kernel_height,stride):
    return (height_in -1) * stride - (2*padding) + dilation * (kernel_height - 1) +1



def get_h_w_out(shape_in, **kwargs):
    kernel = kwargs.get("kernel",(3,3))
    padding = kwargs.get("padding",(0,0))
    dilation = kwargs.get("dilation",(0,0))
    stride = kwargs.get("stride",(1,1))
    #for var in [kernel, padding, dilation, stride]:
    if type(kernel) == int:
        kernel = (kernel,kernel)
    if type(padding) == int:
        padding = (padding, padding)
    if type(dilation) == int:
        dilation = (dilation, dilation)
    if type(stride) == int:
        stride = (stride, stride)
    #print(shape_in, padding, dilation, kernel, stride)
    height = get_height_out(shape_in[0],padding[0],dilation[0],kernel[0],stride[0])
    width = get_height_out(shape_in[1],padding[1],dilation[1],kernel[1],stride[1])
    return(int(height),int(width))



def get_h_w_tran_out(shape_in, **kwargs):
    kernel = kwargs.get("kernel",(3,3))
    padding = kwargs.get("padding",(0,0))
    dilation = kwargs.get("dilation",(0,0))
    stride = kwargs.get("stride",(1,1))
    #for var in [kernel, padding, dilation, stride]:
    if type(kernel) == int:
        kernel = (kernel,kernel)
    if type(padding) == int:
        padding = (padding, padding)
    if type(dilation) == int:
        dilation = (dilation, dilation)
    if type(stride) == int:
        stride = (stride, stride)
    #print(shape_in, padding, dilation, kernel, stride)
    height = get_height_out_tran(shape_in[0],padding[0],dilation[0],kernel[0],stride[0])
    width = get_height_out_tran(shape_in[1],padding[1],dilation[1],kernel[1],stride[1])
    return(int(height),int(width))




#print(get_h_w_out((25,513),kernel=(3,3),padding=(0,0),dilation=(1,3),stride=(1,3)))


#print(get_h_w_out((25,513),kernel=(3,3),padding=(0,0),dilation=(1,3),stride=(1,3),youwhat=(9,9),hello="2324"))

#print(get_h_w_out((25,513),(0,0),(0,0),(1,3),(1,3)))


#print(get_h_w_out((50,162),(0,0),(0,0),(1,3),(1,3)))


#print(get_h_w_out((100,45),(0,0),(0,0),(1,3),(1,3)))


#print(get_h_w_out((200,6),(0,0),(0,0),(1,3),(1,3)))

#Same for convolutions except dilation is (1,1)
#print(get_h_w_out((1,522),(0,0),(1,1),(1,10),(1,1)))

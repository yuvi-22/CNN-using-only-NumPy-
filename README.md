# CNN-using-only-NumPy-

this is my attempt to create a Convolutional Neural Network only using NumPy. 
I tried to create all the functions , i.e  , convoluting over an image , Pooling ( in this case  , MaxPool)  , Activation function ( in this case , I used  ReLU for the hidden layer and the Softmax for the output layer) and a fully connected layer. 

and then used this on the MNIST digit dataset 
well the forward pass works well with any number of filters .

but obviously , it predicted the images waayy too wrong 

then tried to implement back propogation , which took me an entire day to code and understand.
and when i tried to train it (forward + back prop) , it took an exceptionally large amount of time , no output for an hour . 

then i reduced the number of filters ( used 4 before and then reduced to 1 ) , kept the fully connected layer as it is ( 10 neuron in each layer with their biases ) 
and finally got an output
but it took 13 minutes to complete 1 epoch 

so i think , filters and computation time are exponentially proportional 
anyway here is the detailed report of what is happening : 

a typical CNN consists of a convoluting block , 2) a ReLU block 3) a Pooling layer  4) and a flatten layer  5) lastly a fully connected layer 

# convolution block 
firstly , we pick a region , i.e , the area of the image to be convoluted , dot product it wth a filter and store it in feature map 
the convolution block captures small-small details first, such as , curves, shadows, and then uses these small details to capture the bigger picture /details .

since the images as grey scale, we only need 1 channel so a filter of size ( 4, 1, 3, 3) is ideal ( basically 4 , 3x3 filters stacked on top of one another) 


# ReLU block 
returns 0 if less than 0 or x if greater than or eqal to 0 , obviously 


# MaxPool block
same concept as convolution block , except instead of mulptiplying it with a filter , we choose the maximum value from that feature and store it in a feature map 


# flattening 
converting the 2D array to a 1D by .ravel() function

# fully connected layer
the fully connected layer is just the dot product between the input from the flatten layer and the weights of the FC layer , which were randomly initialized ( He Initialization would be much better) 

# softmax layer 
it returns the probabality distribution of all the scenarios and gives the index with the hightest probability and the value


this concludes the forward pass
now the main headache , the backward pass

firstly we find the loss function 
cross entropy is the perfect loss fucntion in this scenario


then 

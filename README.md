# DeepLearningLenet5

We ran all of the models implemented separately and used a learning rate of 0.001 and 20 epochs for all
models. The prints the train and test accuracy every 5 epochs. We did not succeed in getting cuda to work and therefore run the models locally on CPU (This took about half an hour :( ). 
The code in the repository entitled 'hw1_secondtry' is designed to be self-contained, in terms of training and returning the required graphs. The graphs were printed using the 'matplotlib' library.
one can run the code via the google colab repo by creating a new code cell and running the following lines:

%cd DeepLearningLenet5
!python EX1_318852738_230336612_345813455.py

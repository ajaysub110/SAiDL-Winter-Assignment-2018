# SAiDL-Winter-Assignment-2018

- [x] Numpy neural network (4.1)
- [x] Word Embeddings (4.2)
- [x] Image Generation (4.3)
- [x] Imitation Learning (4.4)

## Question 4.1: Numpy neural network (File: nn-xor-xnor.ipynb)
- **Data:** I have constructed a truth table for the given function for only one bit. We could do it for 2 bits as well but it is not necessary since both XOR and XNOR are bitwise operations.
- **Model:** I have constructed a neural network (with separate functions for initializing weights, loss calculation, forward propagation, back propagation, train and predict functions) using only the numpy library.
- **Training:** Training has been done on a large number of epochs (100000) since we desire overfitting in this scenario.


## Question 4.2: Word Embeddings (File: word-embeddings.ipynb)
- The first few cells are to integrate the notebook with Google Colab, download Pytorch, and obtain the Large Movie Review Dataset from Kaggle. Please ignore these cells.
- Then, I have defined methods to load a single document (load_doc), clean and extract tokens (clean_doc), Load all files (load_all) and separate tokenization (tokenize_corpus)
- This is an implementation of the word2vec embedding model using a simple 2 layer neural network, in pytorch.
- **Hyperparameters and training:**
  - embedding dimensions = 5
  - epochs = 1000
  - learning rate = 0.001
- Finally, I have used sklearn's TSNE method for dimensionality reduction (to visualize the embeddings on a 2D plane)
- **Note:** Training the model on the aclImdb dataset required high computational resources and hence in the notebook, I have used a smaller corpus of text.

## Question 4.3: Image Generation (File: dcgan.ipynb)
- Again, please ignore the first few cells
- This is the implementation of a standard DCGAN.
- The hyperparameters are obtained from a command line argument parser
- The GAN has been trained on the CIFAR 10 dataset for 10 epochs with a batch size of 64
- Note that the graph for the generator and discriminator losses is not the ideal graph since the model has been trained for 5 epochs twice.
- **Optimizations:**
  - Generated image before optimization:
  
    ![Before optimization](https://raw.githubusercontent.com/ajaysub110/SAiDL-Winter-Assignment-2018/master/gan_images/gan_1.png)
    
  - Optimizations made to the model (Made based on suggestions from ganhacks):
    - Changed some ReLU functions to Leaky ReLU 
    - Added Tanh activation at the end of the Generator network
    - Used Adam optimizer instead of SGD
    - Changed objective function to max(log(D(G(z))) instead of min(log(1 - D(G(z)))
    
  - Generated image after optimization:
  
    ![After optimization](https://raw.githubusercontent.com/ajaysub110/SAiDL-Winter-Assignment-2018/master/gan_images/gan_3.png)
  - I have added another intermediate generated image (gan_2.png) after 5 epochs post optimization, in the gan_images directory
  
 ## Question 4.4 Imitation Learning (File: imitation-learning)
 - This is an implementation of the UCB DRL course in pytorch.
 - I have implemented methods for both the Behavioral Cloning and DAgger methods of imitation learning
 - Again, the hyperparameters have been taken from a command line argument parser
 - **File descriptions:** Below are the description of each file
   - agent.py : Code for the neural network that generates actions from observations
   - main.py : Declares all hyperparameters used in the code
   - run_agent.py : Runs the Mujoco simulation of a humanoid agent taking the trained model weights
   - run_expert.py : Runs the expert simulation that was provided in the repository
   - train.py : Definition for important methods namely, DAgger, Behavioral cloning, fit_dataset, save and load model
   - tf_utils : Redundant file, but scared to delete
   
   ------------------------------------------------------------------------------------------------------------------------------

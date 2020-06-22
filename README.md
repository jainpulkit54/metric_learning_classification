# metric_learning_classification

MNIST datasets are used for training to get their respective embeddings. <br>
I have here used 2-dimensional embeddings which is not the best choice for other complicated datasets but works fine for easy datasets like "MNIST". <br>

The architecture details of the embedding network used is: <br>
### (32 conv 5x5 --> PReLU --> MaxPool 2x2 --> 64 Conv 5x5 --> PReLU --> MaxPool 2x2 --> Dense 256 --> PReLU --> Dense 256 --> PReLU --> Dense 2) <br>

The architecture details of the classification network used is: <br>
### (PReLU --> Dense 10) <br>

The loss function used is the "cross entropy loss".
The embeddings obtained for the datasets are in the "outputs" folders respectively.

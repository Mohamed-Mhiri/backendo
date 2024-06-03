
import tensorflow as tf




N_EPOCHS = 200  # number of training cycles that will be formed on the whole dataset
N_NEURONS =300
N_LAYERS = 3
N_STACKS = 10  # total number of stacks of n-beats block
WINDOW_SIZE = 24  # our windows size
HORIZON = 1  # one horizon (forecast one hour into the future)

INPUT_SIZE = WINDOW_SIZE * HORIZON  # input size used to get the backcast
THETA_SIZE = INPUT_SIZE + HORIZON

INPUT_SIZE = WINDOW_SIZE * HORIZON  # input size used to get the backcast
THETA_SIZE = INPUT_SIZE + HORIZON 
# Create NBeatsBlock custom layer

class NBeatsBlock(tf.keras.layers.Layer):
    def __init__(self, # the constructor takes all the hyperparameters for the layer
               input_size: int,
               theta_size: int,
               horizon: int,
               n_neurons: int,
               n_layers: int,
               **kwargs): # the **kwargs argument takes care of all of the arguments for the parent class (input_shape, trainable, name)
        super().__init__(**kwargs)
        self.input_size = input_size
        self.theta_size = theta_size
        self.horizon = horizon
        self.n_neurons = n_neurons
        self.n_layers = n_layers

        # Block contains stack of 4 fully connected layers each has ReLU activation
        self.hidden = [tf.keras.layers.Dense(n_neurons, activation="relu") for _ in range(n_layers)]
        # Output of block is a theta layer with linear activation
        self.theta_layer = tf.keras.layers.Dense(theta_size, activation="linear", name="theta")

    def call(self, inputs): # the call method is what runs when the layer is called
        x = inputs
        for layer in self.hidden: # pass inputs through each hidden layer
            x = layer(x)
        theta = self.theta_layer(x)
        # Output the backcast and forecast from theta
        backcast, forecast = theta[:, :self.input_size], theta[:, -self.horizon:]

        return backcast, forecast


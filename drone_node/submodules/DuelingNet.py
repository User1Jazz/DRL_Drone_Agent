from tensorflow import GradientTape as tape
from tensorflow import reduce_mean, convert_to_tensor
import copy
import inspect

from keras.layers import Dense
from keras.layers import InputLayer
from keras.layers import Layer
from keras.models import Model
import keras.backend as K
import numpy as np

class DuelingNet(Model):
    def __init__(self, layers=None, num_actions=0, input_shape=None, trainable=True, name=None):
        if num_actions <= 0:
            raise Exception("Num actions is not set properly. num_actions=", num_actions, " Must be greater than 0")
        if input_shape == None:
            raise Exception("Input shape not defined. input_shape=", num_actions)
        super(DuelingNet, self).__init__(trainable=trainable, name=name)
        self._layers = []
        self.state_value = Dense(1, name='state_value')
        self.advantage_value = Dense(num_actions, name='advantage_value')
        self.num_actions = num_actions
        self._layers.append(InputLayer(input_shape=input_shape))
        print("Input shape in list: ", self._layers[0]._batch_input_shape)
        print("Defined input shape: ", input_shape)
        if layers:
            for layer in layers:
                self.add(layer, rebuild=False)
            self.maybe_rebuild()
        return
    
    def add(self, layer, rebuild=True):
        """Adds a layer instance on top of the layer stack.

        Args:
            layer: layer instance.
        """
        print("Adding layer")

        # If we are passed a Keras tensor created by keras.Input(), we
        # extract the input layer from its keras history and use that.
        if hasattr(layer, "_keras_history"):
            origin_layer = layer._keras_history[0]
            if isinstance(origin_layer, InputLayer):
                layer = origin_layer
        if not isinstance(layer, Layer):
            raise ValueError(
                "Only instances of `keras.Layer` can be "
                f"added to a Dueling model. Received: {layer} "
                f"(of type {type(layer)})"
            )
        if not self._is_layer_name_unique(layer):
            raise ValueError(
                "All layers added to a Dueling model "
                f"should have unique names. Name '{layer.name}' is already "
                "the name of a layer in this model. Update the `name` argument "
                "to pass a unique name."
            )
        if (
            isinstance(layer, InputLayer)
            and self._layers
            and isinstance(self._layers[0], InputLayer)
        ):
            raise ValueError(
                f"Dueling model '{self.name}' has already been configured "
                f"to use input shape {self._layers[0].batch_shape}. You cannot "
                f"add a different Input layer to it."
            )

        self._layers.append(layer)
        if rebuild:
            self._maybe_rebuild()
        else:
            self.built = False
            self._functional = None

    def _maybe_rebuild(self):
        self.built = False
        self._functional = None
        if isinstance(self._layers[0], InputLayer) and len(self._layers) > 1:
            input_shape = self._layers[0]._batch_input_shape
            self.build(input_shape)

    def build(self, input_shape=None):
        if not isinstance(input_shape, (tuple, list)):
            # Do not attempt to build if the model does not have a single
            # input tensor.
            return
        if input_shape and not (
            isinstance(input_shape[0], int) or input_shape[0] is None
        ):
            # Do not attempt to build if the model does not have a single
            # input tensor.
            return
        if not self._layers:
            raise ValueError(
                f"Dueling model {self.name} cannot be built because it has "
                "no layers. Call `model.add(layer)`."
            )
        if isinstance(self._layers[0], InputLayer):
            if self._layers[0]._batch_input_shape != input_shape:
                raise ValueError(
                    f"Dueling model '{self.name}' has already been "
                    "configured to use input shape "
                    f"{self._layers[0]._batch_input_shape}. You cannot build it "
                    f"with input_shape {input_shape}"
                )
        else:
            dtype = self._layers[0].compute_dtype
            self._layers = [
                InputLayer(batch_shape=input_shape, dtype=dtype)
            ] + self._layers

        # Build functional model
        inputs = self._layers[0].output
        x = inputs
        for layer in self._layers[1:]:
            try:
                x = layer(x)
            except NotImplementedError:
                # Can happen if shape inference is not implemented.
                # TODO: consider reverting inbound nodes on layers processed.
                return
            except TypeError as e:
                signature = inspect.signature(layer.call)
                positional_args = [
                    param
                    for param in signature.parameters.values()
                    if param.default == inspect.Parameter.empty
                ]
                if len(positional_args) != 1:
                    raise ValueError(
                        "Layers added to a Dueling model "
                        "can only have a single positional argument, "
                        f"the input tensor. Layer {layer.__class__.__name__} "
                        f"has multiple positional arguments: {positional_args}"
                    )
                raise e
        self.state_value = Dense(1, name='state_value', input_shape=x.shape)
        self.advantage_value = Dense(self.num_actions, name='advantage_value', input_shape=x.shape)
        V = self.state_value(x)
        A = self.advantage_value(x)
        outputs = V + (A - reduce_mean(A, axis=1, keepdims=True))
        #outputs = x
        self.built = True

    def call(self, inputs):
        #inputs = self._layers[0].output
        x = np.array([inputs])
        print(x.shape)
        for layer in self._layers:
            x = layer(x)
        state_value = self.state_value(x)
        advantage_value = self.advantage_value(x)
        q_values = state_value + (advantage_value - reduce_mean(advantage_value, axis=1, keepdims=True))
        print(q_values)
        return q_values
    
    def _is_layer_name_unique(self, layer):
        for ref_layer in self._layers:
            if layer.name == ref_layer and ref_layer is not layer:
                return False
        return True
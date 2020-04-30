import torch.nn as nn


class Feedforward(nn.Module):
    def __init__(self, input_size, hidden_dimensions, output_size):
        super(Feedforward, self).__init__()
        assert len(hidden_dimensions) > 0
        self.input_layer = nn.Linear(input_size, hidden_dimensions[0])
        self.input_activation = nn.ReLU()
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dimensions[i], h_size)
                                            for i, h_size in enumerate(hidden_dimensions[1:])])
        self.hidden_activations = nn.ModuleList([nn.ReLU() for _ in hidden_dimensions[1:]])
        self.output_layer = nn.Linear(hidden_dimensions[-1], output_size)

    def forward(self, x):
        x = self.input_activation(self.input_layer(x))
        for i, layer in enumerate(self.hidden_layers):
            x = self.hidden_activations[i](layer(x))
        x = self.output_layer(x)
        return x


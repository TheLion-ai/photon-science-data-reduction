from torch.nn import Linear, Module, Sequential, Conv2d, Tanh, AvgPool2d, Flatten, AdaptiveAvgPool2d

class LeNet(Module):
    def __init__(self, num_classes, img_size=224):
        """A class creating LeNet architecture.

        Args:
            num_classes (int): the number of neurons in the output layer.
            img_size (int, optional): Input image size required by the model. Defaults to 224.
        """
        super(LeNet, self).__init__()
        
        if img_size == 32:
            self.features = Sequential(            
                Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
                Tanh(),
                AvgPool2d(kernel_size=2),
                Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
                Tanh(),
                AvgPool2d(kernel_size=2),
                Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
                Tanh(),
                Flatten()
            )
        else:
            # For img_size==224, avg pooling is required before flattening
            self.features = Sequential(            
                Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
                Tanh(),
                AvgPool2d(kernel_size=2),
                Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
                Tanh(),
                AvgPool2d(kernel_size=2),
                Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
                Tanh(),
                AvgPool2d(kernel_size=49),
                Flatten()
            )

        self.classifier = Sequential(
            Linear(in_features=120, out_features=84),
            Tanh(),
            Linear(in_features=84, out_features=num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        logits = self.classifier(x)
        return logits

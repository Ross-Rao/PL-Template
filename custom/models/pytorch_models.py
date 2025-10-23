import torch
import torch.nn as nn
import torch.nn.functional as F
import monai.networks.nets as monai_nets
import torchvision.models as models
from torch_geometric.utils import dropout_edge
from torch_geometric.nn import GATConv, LayerNorm


class MonaiResNetMLP(nn.Sequential):
    def __init__(self, model_name, pretrained, progress, out_features, **kwargs):
        resnet = getattr(monai_nets, model_name)(pretrained=pretrained, progress=progress, **kwargs)
        features = nn.Sequential(*list(resnet.children())[:-1])
        super().__init__(
            features,
            nn.Flatten(),
            nn.Linear(resnet.in_planes, resnet.in_planes),
            nn.Linear(resnet.in_planes, out_features)
        )

class PredictHead(nn.Sequential):
    def __init__(self, in_features, out_features, dropout=0.5):
        super().__init__(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features, out_features)
        )

class GATNet(nn.Module):
    def __init__(self, in_features, out_features, num_heads=8, drop_edge=0.2, layer_dims=(448, 384, 256)):
        super().__init__()
        self.drop_edge = drop_edge

        self.conv_list = nn.ModuleList([GATConv(in_features, layer_dims[0], num_heads)] +
                                       [GATConv(in_c * num_heads, out_c, num_heads) for in_c, out_c in
                                        zip(layer_dims[:-1], layer_dims[1:])])
        self.norm_list = nn.ModuleList([LayerNorm(out_c * num_heads) for out_c in layer_dims])
        self.final_conv = GATConv(layer_dims[-1] * num_heads, out_features)

    def forward(self, x, edge_index):
        edge_index, _ = dropout_edge(edge_index, p=self.drop_edge, training=self.training)

        for conv, norm in zip(self.conv_list, self.norm_list):
            x = F.relu(conv(x, edge_index))
            x = norm(x)

        x = self.final_conv(x, edge_index)
        return x

class ResnetMLP(nn.Module):
    def __init__(self, path=None, train=True):
        super(ResnetMLP, self).__init__()
        # Load the ResNet model
        resnet = models.resnet18(weights=None, norm_layer=nn.InstanceNorm2d)

        # Remove the last fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # Add a projection MLP
        num_ftrs = resnet.fc.in_features
        self.l1 = nn.Linear(num_ftrs, num_ftrs)

        # This is hardcoded to 256, because the pretrained model was trained with this size
        self.l2 = nn.Linear(num_ftrs, 256)

        if path:
            weight = torch.load(path)
            if path.endswith('.ckpt'):
                weight = {k.replace('model.0.', ''): v for k, v in weight['state_dict'].items() if k.startswith('model.0.')}
                self.load_state_dict(weight)
            else:
                weight = {k.replace('module.', ''): v for k, v in weight.items() if k.startswith('module.')}
                self.load_state_dict(weight)
        if not train:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return x


if __name__ == "__main__":
    model = ResnetMLP()
    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    print(y.shape)

    model = MonaiResNetMLP("resnet18", False, False, 256, spatial_dims=2,
                           norm=('instance', {'affine': True}))
    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    print(y.shape)

    model = GATNet(256, 1000)
    x = torch.randn(10, 256)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 2, 3],
                               [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 2, 0, 4]], dtype=torch.long)
    y = model(x, edge_index)
    print(y.shape)

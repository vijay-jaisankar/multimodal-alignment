"""
    Linear mapping between Pointnet outputs and the CLIP-VIT-LAION model
"""
import torch
import torch.nn

# Set the device the model will be loaded on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model definition
class LinearProjectionHead(nn.Module):
    def __init__(self, input_emb_size = 1024, output_emb_size = 1024, inter_size_1 = 2048, inter_size_2 = 4096, bottle_size = 8192, dropout_rate = 0.15, device = device):
        super().__init__()
        # Initialise parameters
        self.input_emb_size = input_emb_size
        self.output_emb_size = output_emb_size
        self.inter_size_1 = inter_size_1
        self.inter_size_2 = inter_size_2
        self.bottle_size = bottle_size
        self.dropout_rate = dropout_rate
        self.device = device
        
        # Up projection -1
        self.up1 = nn.Sequential(
            nn.Linear(self.input_emb_size, self.inter_size_1),
            nn.PReLU(),
            nn.Dropout(p = self.dropout_rate)
        )
        
        # Up projection -2
        self.up2 = nn.Sequential(
            nn.Linear(self.inter_size_1, self.inter_size_2),
            nn.PReLU(),
            nn.Dropout(p = self.dropout_rate)
        )
        
        # Bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Linear(self.inter_size_2, self.bottle_size),
            nn.Tanh(),
            nn.Dropout(p = self.dropout_rate)
        )
        
        # Down projection-1
        self.down1 = nn.Sequential(
            nn.Linear(self.bottle_size, self.inter_size_2),
            nn.Tanh(),
            nn.Dropout(p = self.dropout_rate)
        )
        
        # Down projection-2
        self.down2 = nn.Sequential(
            nn.Linear(self.inter_size_2, self.inter_size_1),
            nn.Tanh(),
            nn.Dropout(p = self.dropout_rate)
        )
        
        # Final projection to output space
        self.fc = nn.Sequential(
            nn.Linear(self.inter_size_1, self.output_emb_size)
        )
        
    def forward(self, x):
        x = x.to(device)
        x = self.up1(x)
        x = self.up2(x)
        x = self.bottleneck(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.fc(x)
        return x

# Instantiate the model
net = LinearProjectionHead(input_emb_size = 1024, output_emb_size = 1024, inter_size_1 = 2048, inter_size_2 = 4096, bottle_size = 8192, dropout_rate = 0.15, device = device).to(device)
net.train()

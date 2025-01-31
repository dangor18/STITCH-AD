import torch
from torch.amp import autocast, GradScaler
from model_utils.train_utils import get_optimizer, create_model

class RD(torch.nn.Module):
    def __init__(self, architecture: str, bn_attention: bool, channels: int, device, params: dict):
        super(RD, self).__init__()
        self.encoder, self.bn, self.decoder = create_model(architecture=architecture, bn_attention=bn_attention, in_channels=channels)
        self.encoder = self.encoder.to(device).eval()
        self.bn = self.bn.to(device)
        self.decoder = self.decoder.to(device)
        
        self.optimizer = get_optimizer(params, list(self.decoder.parameters()) + list(self.bn.parameters()))
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                                    mode='min', 
                                                                    factor=params["lr_factor"], 
                                                                    patience=params.get("patience", 3))
        self.scaler = GradScaler("cuda")
        self.params = params
        self.device = device
    
    def forward(self, images):
        with autocast(device_type="cuda"):
            inputs = self.encoder(images)
            outputs = self.decoder(self.bn(inputs))
        return inputs, outputs
    
    def save_model(self, path):
        torch.save({'bn': self.bn.state_dict(), 'decoder': self.decoder.state_dict()}, path)
    
    def load_model(self, path):
        ckp = torch.load(path, weights_only=True)
        for k, v in list(ckp['bn'].items()):
            if 'memory' in k:
                ckp['bn'].pop(k)
        self.decoder.load_state_dict(ckp['decoder'])
        self.bn.load_state_dict(ckp['bn'])

    def train(self):
        """
        SET THE MODEL TO TRAIN MODE
        """
        self.encoder.eval()
        self.decoder.train()
        self.bn.train()

    def eval(self):
        """
        SET THE MODEL TO EVAL MODE
        """
        self.encoder.eval()
        self.decoder.eval()
        self.bn.eval()
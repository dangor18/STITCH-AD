import torch
from model_utils.train_utils import create_model_proj

class RevistingRD(torch.nn.Module):
    def __init__(self, architecture: str, bn_attention: bool, channels: int, device, params: dict):
        super(RevistingRD, self).__init__()
        self.encoder, self.bn, self.decoder, self.proj_layer = create_model_proj(architecture=architecture, bn_attention=bn_attention, in_channels=channels)
        self.encoder = self.encoder.to(device).eval()
        self.bn = self.bn.to(device)
        self.decoder = self.decoder.to(device)
        self.proj_layer = self.proj_layer.to(device)
        
        self.optimizer_proj = torch.optim.Adam(list(self.proj_layer.parameters()), lr=params.get("proj_lr", 0.001), betas=(params.get("beta1_proj", 0.5),params.get("beta2_proj", 0.999)))
        self.optimizer_distill = torch.optim.Adam(list(self.decoder.parameters())+list(self.bn.parameters()), lr=params.get("distill_lr", 0.005), betas=(params.get("beta1_distill", 0.5),params.get("beta2_distill", 0.999)))

        # lr schedulers
        self.distill_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_distill, 
            mode='min',
            patience=params.get("distill_patience", 5),
            factor=params.get("distill_lr_factor", 0.5),
            min_lr=1e-6
        )
        self.proj_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_proj, 
            mode='min',
            patience=params.get("proj_patience", 5),
            factor=params.get("proj_lr_factor", 0.5),
            min_lr=1e-6
        )

        self.accumulation_steps = 2
        self.params = params
        self.device = device
    
    def forward(self, img, img_noise = False):
        inputs = self.encoder(img)
        if img_noise is False:
            feature_space = self.proj_layer(inputs)
            outputs = self.decoder(self.bn(feature_space))
            return (inputs, outputs)
        else:
            inputs_noise = self.encoder(img_noise)
            (feature_space_noise, feature_space) = self.proj_layer(inputs, features_noise = inputs_noise)
            outputs = self.decoder(self.bn(feature_space))
            return (feature_space_noise, feature_space, inputs, inputs_noise, outputs)
    
    def save_model(self, path):
        torch.save({'proj': self.proj_layer.state_dict(),
                       'decoder': self.decoder.state_dict(),
                        'bn':self.bn.state_dict()}, path)
    
    def load_model(self, path):
        ckp = torch.load(path, weights_only=True)
        for k, v in list(ckp['bn'].items()):
            if 'memory' in k:
                ckp['bn'].pop(k)
        self.decoder.load_state_dict(ckp['decoder'])
        self.proj_layer.load_state_dict(ckp['proj'])
        self.bn.load_state_dict(ckp['bn'])

    def train(self):
        """
        SET THE MODEL TO TRAIN MODE
        """
        self.encoder.eval()
        self.decoder.train()
        self.bn.train()
        self.proj_layer.train()

    def eval(self):
        """
        SET THE MODEL TO EVAL MODE
        """
        self.encoder.eval()
        self.decoder.eval()
        self.bn.eval()
        self.proj_layer.eval()
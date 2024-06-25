import torch
import torch.nn.functional as F
import torch.optim as optim
import lightning as lt
from utils import Parser

class VAE(lt.LightningModule):
    def __init__(self,
                 config_file: str = 'vae_config.yaml',
                 detached: bool = False,
                 **kwargs) -> None:
        super(VAE, self).__init__()
        self.save_hyperparameters()
        p=Parser(config_file=config_file)
        self.detached = detached

        # Read model file and build networks
        self.config = p.config

        self.kld_weight = self.config['loss_weight']['kld']
        self.mse_weight = self.config['loss_weight']['mse']
        self.class_weight = self.config['loss_weight']['class']

        self.encoder = p.network(self.config['encoder'])
        self.fc_mu = p.network(self.config['latent_space'])
        self.classifier = p.network(self.config['classifier'])
        self.fc_var = p.network(self.config['latent_space'])
        self.decoder = p.network(self.config['decoder'])

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        pred = self.classifier(z) if self.detached == False else self.classifier(z.detach())
        return self.decode(z), pred, mu, log_var

    def encode(self, x):
        flattened = self.encoder(x)
        mu = self.fc_mu(flattened)
        log_var = self.fc_var(flattened)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def loss_function(self, recon_x, x, mu, log_var):
        BCE = F.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE, KLD

    def sample(self, num_samples):
        z = torch.randn(num_samples, self.latent_dim)
        return self.decode(z)

    def generate(self, x):
        return self.forward(x)[0]

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.005)

    def training_step(self, batch, batch_idx):
        x, y = batch
        recon_batch, pred, mu, log_var = self(x)
        train_mse, train_kld = self.loss_function(recon_batch, x, mu, log_var)
        train_class = F.cross_entropy(pred, y)
        train_mse, train_kld, train_class = train_mse * self.mse_weight, train_kld * self.kld_weight, train_class * self.class_weight
        values = {"train_mse": train_mse, "train_kld": train_kld, "train_class": train_class}
        self.log_dict(values, prog_bar=True)
        return train_mse + train_kld + train_class

    def validation_step(self, batch, batch_idx):
        x, y = batch
        recon_batch, pred, mu, log_var = self(x)
        val_mse, val_kld = self.loss_function(recon_batch, x, mu, log_var)
        val_class = F.cross_entropy(pred, y)
        val_mse, val_kld, val_class = val_mse * self.mse_weight, val_kld * self.kld_weight, val_class * self.class_weight
        values = {"val_bce": val_mse, "val_kld": val_kld, "val_class": val_class,
                  "val_loss": val_mse + val_kld + val_class}
        self.log_dict(values, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, _ = batch
        recon_batch, _, mu, log_var = self(x)
        test_bce, test_kld = self.loss_function(recon_batch, x, mu, log_var)

        values = {"test_bce": test_bce, "test_kld": test_kld}
        self.log_dict(values, prog_bar=True)

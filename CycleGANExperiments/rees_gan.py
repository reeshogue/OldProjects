import torch
import itertools
import torch.nn.functional as F
from models import Generator, Discriminator, AEDiscriminator, Real, Fake, OriginalGenerator
from generate_show import generate_show
    
    
class CycleGAN:
    def __init__(self):
        self.generator1 = OriginalGenerator()
        self.discriminator1 = Discriminator()
        self.meta_generator1 = OriginalGenerator()
        self.meta_discriminator1 = Discriminator()

        self.real = Real()
        self.fake = Fake()

        self.generator1_optim = torch.optim.AdamW(self.generator1.parameters(), amsgrad=True, lr=0.009)
        self.discriminator1_optim = torch.optim.AdamW(self.discriminator1.parameters(), amsgrad=True, lr=0.009)
        self.meta_generator1_optim = torch.optim.AdamW(self.generator1.parameters(), amsgrad=True, lr=0.009)
        self.meta_discriminator1_optim = torch.optim.AdamW(self.discriminator1.parameters(), amsgrad=True, lr=0.009)
        self.decay_time = 1.0
        self.growth_time = 0.005

    def train_meta_gan(self, data2):
        data2 = torch.zeros(1, 3, 256, 256) + data2
        noise = torch.randn(1, 1, 16, 16)

        generation = self.meta_generator1(noise)
        loss = torch.max(self.meta_discriminator1(generation))
        
        self.meta_generator1_optim.zero_grad()
        loss.backward()
        self.meta_generator1_optim.step()

        generation = self.meta_generator1(noise)
        
        truth_data2 = self.meta_discriminator1(data2)
        false_data2 = self.meta_discriminator1(generation)
        
        loss = F.mse_loss(truth_data2, self.real()) + F.mse_loss(false_data2, self.fake())
        
        self.meta_discriminator1_optim.zero_grad()
        loss.backward()
        self.meta_discriminator1_optim.step()

    def train_discriminator_one_step(self, data2):
        noise = torch.randn(1, 1, 16, 16)
        
        truth_data2 = self.discriminator1(data2)
        mse_truth_data2 = F.mse_loss(truth_data2, self.real())
 
        lies_data2 = self.discriminator1(self.generator1(noise))
        mse_lies_data2 = F.mse_loss(lies_data2, self.fake())
        
        data2_mse = mse_truth_data2 + mse_lies_data2
        print("Loss (D):", data2_mse.item())
        
        self.discriminator1_optim.zero_grad()
        data2_mse.backward()
        self.discriminator1_optim.step()

    def train_generator_one_step(self, data2):
        noise = torch.randn(1, 1, 16, 16)

        fake_data2 = self.generator1(noise)
        
        loss_generator1_gan = torch.max(self.discriminator1(fake_data2))
        loss_gen1 = (self.growth_time * loss_generator1_gan) + (self.decay_time * F.mse_loss(fake_data2, data2))
        loss_gen2 = loss_gen1 #+ torch.mean(self.meta_generator1(noise))
        self.generator1_optim.zero_grad()
        loss_gen2.backward()
        self.generator1_optim.step()
        self.decay_time *= 0.9995
        self.growth_time *= 1.0005
        print("Loss (G):", loss_gen1.item())
        return loss_gen1.detach()

    def train_dataloader(self, nepochs, dataloader1, dataloader2):
        
        for epoch in range(nepochs):
            for i, data in enumerate(dataloader2):
                data2 = data
                data2 = data2[0]
                self.train_discriminator_one_step(data2)
                self.train_discriminator_one_step(data2)
                loss_gen1 = self.train_generator_one_step(data2)
#                self.train_meta_gan(loss_gen1)
            print("Epoch", epoch, 'done.')
    def generate_and_show(self, data):
        noise = torch.randn(1,1,16,16)
        generate_show(self.generator1, noise, show_data_plain=False)

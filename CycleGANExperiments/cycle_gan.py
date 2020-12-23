import torch
import random
import torch.nn.functional as F
from models import Generator, Discriminator, AEDiscriminator, Real, Fake
from generate_show import generate_show

def greyscale(data1):
    data_avg1 = torch.mean(data1, dim=1, keepdim=True)
    data_avg1 = torch.cat([data_avg1, data_avg1, data_avg1], 1)
    return data_avg1

class CycleGAN:
    def __init__(self):
        self.generator1 = Generator()
        self.generator2 = Generator()
        self.discriminator1 = AEDiscriminator()
        self.discriminator2 = AEDiscriminator()
        self.discriminator3 = AEDiscriminator()
        self.discriminator4 = AEDiscriminator()
        
        self.random_network = Generator()

        self.real = Real()
        self.fake = Fake()

        self.generator1_optim = torch.optim.AdamW(self.generator1.parameters(), amsgrad=True, lr=0.009)
        self.generator2_optim = torch.optim.AdamW(self.generator2.parameters(), amsgrad=True, lr=0.009)
        

        self.discriminator1_optim = torch.optim.AdamW(self.discriminator1.parameters(), amsgrad=True, lr=0.009)
        self.discriminator2_optim = torch.optim.AdamW(self.discriminator2.parameters(), amsgrad=True, lr=0.009)

        self.discriminator3_optim = torch.optim.AdamW(self.discriminator3.parameters(), amsgrad=True, lr=0.009)
        self.discriminator4_optim = torch.optim.AdamW(self.discriminator4.parameters(), amsgrad=True, lr=0.009)

        self.time_growth = 0.0
        self.time_growth_two = 0
    def train_ae_discriminator_one_step(self, data1, data2):
        self.discriminator1_optim.zero_grad()
        false_data1 = self.discriminator1(data1)
        false_data1.backward()
        self.discriminator1_optim.step()

        self.discriminator2_optim.zero_grad()
        false_data2 = self.discriminator2(data2)
        false_data2.backward()
        self.discriminator2_optim.step()

    def train_ae_greyscale_discriminator_one_step(self, data1, data2):
        self.discriminator3_optim.zero_grad()
        false_data1 = self.discriminator3(greyscale(data1))
        false_data1.backward()
        self.discriminator3_optim.step()

        self.discriminator4_optim.zero_grad()
        false_data2 = self.discriminator4(greyscale(data2))
        false_data2.backward()
        self.discriminator4_optim.step()

    def train_generator_one_step(self, data1, data2):
        fake_data2 = self.generator2(data1)
        fake_data1 = self.generator1(data2)

        recovery_data2 = fake_data2
        recovery_data1 = fake_data1

        for i in range(self.time_growth_two):
             recovery_data2 = self.generator1(recovery_data1)
             recovery_data1 = self.generator2(recovery_data2)

        loss_generator1_rdisc = self.discriminator1(recovery_data1)
        loss_generator2_rdisc = self.discriminator2(recovery_data2)
        
        loss_generator1_rdisc_g = self.discriminator3(greyscale(recovery_data1))
        loss_generator2_rdisc_g = self.discriminator4(greyscale(recovery_data2))
        

        loss_gen1 = (self.time_growth * (loss_generator1_rdisc + loss_generator1_rdisc_g)) + F.mse_loss(recovery_data1, data1)
        loss_gen2 = (self.time_growth * (loss_generator1_rdisc + loss_generator2_rdisc_g)) + F.mse_loss(recovery_data2, data2)
        self.generator1_optim.zero_grad()
        loss_gen1.backward(retain_graph=True)
        self.generator1_optim.step()

        self.generator2_optim.zero_grad()
        loss_gen2.backward()
        self.generator2_optim.step()

        print("Loss generator 1:", loss_gen1.item())
        print("Loss generator 2:", loss_gen2.item())

    def train_dataloader(self, nepochs, dataloader1, dataloader2):
        
        for epoch in range(nepochs):
            for i, data in enumerate(zip(dataloader1, dataloader2)):
                data1, data2 = data

                data1 = data1[0]
                data2 = data2[0]
                self.train_ae_discriminator_one_step(data1, data2)
                self.train_ae_greyscale_discriminator_one_step(data1, data2)
                self.train_generator_one_step(data1, data2)
            print("Epoch", epoch, 'done.')
    def generate_and_show(self, data):
        generate_show(self.generator2, data, show_data_plain=False)

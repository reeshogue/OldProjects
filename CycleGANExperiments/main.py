from rees_gan import CycleGAN
from load_data import load_data
from generate_show import generate_show

if __name__ == '__main__':
    gan = CycleGAN()
    dataloader1, dataloader2 = load_data()
    gan.train_dataloader(int(input("How many epochs do you want to train for?:")), dataloader1, dataloader2)

    for i, data in enumerate(dataloader1):
        print("Showing data...")
        gan.generate_and_show(data[0])
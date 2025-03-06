import sys
sys.path.append('/shared_hdd/annasdfghjkl13/APT/stylegan3')
from training.networks_stylegan3 import Generator
from training.networks_stylegan2 import Discriminator

#dataset

#dataset loader

#model
G_z_dims = 512
G_c_dims = 0
G_w_dims = 512
G_img_resol = 512
G_img_chan = 3

D_c_dim = 0
D_img_resol = 512
D_img_chan = 3

G = Generator(z_dim=G_z_dims, c_dim=G_c_dims, w_dim=G_w_dims, img_resolution=G_img_resol, img_channels=G_img_chan) #Generator
D = Discriminator(c_dim=D_c_dim, img_resolution=D_img_resol, img_channels=D_img_chan)#Discriminator

#train
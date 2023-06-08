# Implements a change from a sad face to a cheerful one using an autoencoder

import numpy as np
import pandas as pd
from torch.autograd import Variable
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torch
import matplotlib.pyplot as plt
import os
import skimage.io
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from IPython.display import clear_output
from tqdm import tqdm_notebook
import torchvision
from torchvision import transforms
from copy import deepcopy

def fetch_dataset(attrs_name = "lfw_attributes.txt",
                      images_name = "lfw-deepfunneled",
                      dx=80,dy=80,
                      dimx=64,dimy=64
    ):

    #download if not exists
    if not os.path.exists(images_name):
        print("images not found, donwloading...")
        os.system("wget http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz -O tmp.tgz")
        print("extracting...")
        os.system("tar xvzf tmp.tgz && rm tmp.tgz")
        print("done")
        assert os.path.exists(images_name)

    if not os.path.exists(attrs_name):
        print("attributes not found, downloading...")
        os.system("wget http://www.cs.columbia.edu/CAVE/databases/pubfig/download/%s" % attrs_name)
        print("done")

    #read attrs
    df_attrs = pd.read_csv("lfw_attributes.txt",sep='\t',skiprows=1,) 
    df_attrs = pd.DataFrame(df_attrs.iloc[:,:-1].values, columns = df_attrs.columns[1:])


    #read photos
    photo_ids = []
    for dirpath, dirnames, filenames in os.walk(images_name):
        for fname in filenames:
            if fname.endswith(".jpg"):
                fpath = os.path.join(dirpath,fname)
                photo_id = fname[:-4].replace('_',' ').split()
                person_id = ' '.join(photo_id[:-1])
                photo_number = int(photo_id[-1])
                photo_ids.append({'person':person_id,'imagenum':photo_number,'photo_path':fpath})

    photo_ids = pd.DataFrame(photo_ids)
    # print(photo_ids)
    #mass-merge
    #(photos now have same order as attributes)
    df = pd.merge(df_attrs,photo_ids,on=('person','imagenum'))

    assert len(df)==len(df_attrs),"lost some data when merging dataframes"

    # print(df.shape)
    #image preprocessing
    all_photos =df['photo_path'].apply(skimage.io.imread)\
                                .apply(lambda img:img[dy:-dy,dx:-dx])\
                                .apply(lambda img: resize(img,[dimx,dimy]))

    all_photos = np.stack(all_photos.values)#.astype('uint8')
    all_attrs = df.drop(["photo_path","person","imagenum"],axis=1)
    
    return all_photos, all_attrs
    
# Fetch dataset
data, attrs = fetch_dataset()

# Split into train, validation and test
train_photos, val_photos, train_attrs, val_attrs = train_test_split(data, attrs,
                                                                    train_size=0.8, shuffle=True, random_state=42)
train_loader = torch.utils.data.DataLoader(train_photos, batch_size=16, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_photos, batch_size=16, shuffle=False)

dim_code = 2048 # dimension of latent space

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding="same"), # output = 16 x 64 x 64 (=channels_out x height x width)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # output = 16 x 32 x 32
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding="same"), # 32 x 32 x 32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 32 x 16 x 16
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding="same"), # 64 x 16 x 16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 32 x 8 x 8
            nn.Flatten(),
            nn.Linear(in_features=64*8*8, out_features=dim_code),
            nn.ReLU(),
        )

        self.Flatten = nn.Flatten()

        self.decoder = nn.Sequential(
            nn.Linear(in_features=dim_code, out_features=64*8*8),
            nn.ReLU(),
            nn.Unflatten(1, (64, 8, 8)),
            nn.Upsample(16,mode="nearest"), # 64 x 16 x 16
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding="same"), # 32 x 16 x 16
            nn.ReLU(),
            nn.Upsample(32,mode="nearest"), # 32 x 32 x 32
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding="same"), # 16 x 32 x 32
            nn.ReLU(),
            nn.Upsample(64,mode="nearest"), # 16 x 64 x 64
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding="same"), # 3 x 64 x 64
            nn.ReLU(),
        )
        
    def forward(self, x):
        # returns reconstruction of image (reconstruction) and image of a picture in latent space (latent_code)

        latent_code = self.encoder(x)
        reconstruction = self.decoder(latent_code)
        latent_code = latent_code.reshape(x.shape[0],-1)

        return reconstruction, latent_code
        
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
criterion = nn.MSELoss()
autoencoder = Autoencoder().to(device)
optimizer = torch.optim.Adam(autoencoder.parameters())

# Arguments:
# *. epochs = 10 - epochs number
# *. model = autoencoder - training model (neural netrowk)
# *. train_loader, val_loader - loaders (return only X_batch without Y_batch
# *. optimizer - optimizer (Adam selected)
# *. criterion - loss-function
epochs = 60
model = autoencoder
train_loss_history = []
test_loss_history = []

for epoch in range(epochs):
    print('Epoch %d/%d' % (epoch+1, epochs))
    avg_loss = 0
    model.train()  # train mode
    for X_batch in tqdm_notebook(train_loader):
        inputs = X_batch.permute(0,3,1,2).float().to(device)
        
        # reset the gradient
        optimizer.zero_grad()

        # forward
        pred,_ = model(inputs)
        loss = criterion(inputs, pred)
        #backward
        loss.backward()
        optimizer.step()
        
        # calculate avg loss by epoch
        avg_loss += loss.item()
    
    avg_loss = avg_loss / len(train_loader)
    train_loss_history.append(avg_loss)
    
    # Cleaning GPU
    del X_batch
    del inputs
    del loss
    torch.cuda.empty_cache()

    model.eval()  # testing mode

    avg_loss = 0
    first_batch = None
    for i,X_batch in enumerate(val_loader):
      inputs = X_batch.permute(0,3,1,2).float().to(device)
      pred,_ = model(inputs)
      loss = criterion(inputs, pred)
      avg_loss += loss.item()
      if i==0:
        first_batch = X_batch.detach().numpy()
        first_pred = pred.permute(0,2,3,1).cpu().detach().numpy()
    avg_loss = avg_loss / len(val_loader)
    test_loss_history.append(avg_loss)

    # Cleaning GPU
    del X_batch
    del loss
    del pred
    del inputs
    torch.cuda.empty_cache()

    print('loss: %f' % avg_loss)

    clear_output(wait=True)
    #plt.figure(figsize=(26, 6))
    #for k in range(3):
    #    plt.subplot(2, 4, k+1)
    #    plt.imshow(first_batch[k])
    #    plt.title('Real')
    #    plt.axis('off')
    #
    #    plt.subplot(2, 4, k+5)
    #    plt.imshow(first_pred[k])
    #    plt.title('Output')
    #    plt.axis('off')
    
    #plt.subplot(1,4,4)
    #plt.plot(train_loss_history, label="Train loss")
    #plt.plot(test_loss_history, label="Validation loss")
    #plt.legend()
    
    #plt.suptitle('%d / %d - loss: %f' % (epoch+1, epochs, avg_loss))
    #plt.show()


#Deriving more vectors from the sample to see the indexes of smiling and not smiling people (the selection will be made manually)
#X_imgs = 100 # how many images should be shown
#val_loader_uno = torch.utils.data.DataLoader(val_photos, batch_size=1)
#
#plt.figure(figsize=(15, 5*X_imgs))
#for k, X_batch in enumerate(val_loader_uno):
#    inputs = X_batch.permute(0,3,1,2).float().to(device)
#    pred,_ = model(inputs)
#    input = X_batch.detach().numpy()[0]
#    output = pred.permute(0,2,3,1).cpu().detach().numpy()[0]
#
#    plt.subplot(X_imgs, 2, 2*k+1)
#    plt.imshow(input)
#    plt.title('Real, number {}'.format(k))
#    plt.axis('off')
#
#    plt.subplot(X_imgs, 2, 2*k+2)
#    plt.imshow(output)
#    plt.title('Output')
#    plt.axis('off')
#
#    if k==X_imgs-1:
#      break
#
#plt.show()

# Compute vector from sad face to smile face and show the result
happy_smiles = {0, 2, 3, 5, 6, 13, 17, 18, 19, 32, 36, 37, 39, 41, 42, 43, 46, 48}
sad_smiles = {4, 7, 8, 9, 10, 12, 14, 16, 21, 22, 23, 24, 25, 27, 30, 45, 49}

test_POIs = {62, 65, 68, 74} # with sad smiles

happy_embeddings = np.zeros(shape=(len(happy_smiles),dim_code))
sad_embeddings = np.zeros(shape=(len(sad_smiles),dim_code))

for k, ind in enumerate(happy_smiles):
    inputs = val_loader_uno.dataset[ind]
    inputs = torch.FloatTensor([inputs]).permute(0,3,1,2).to(device)
    pred,latent = model(inputs)
    happy_embeddings[k] = latent[0].cpu().detach().numpy()

for k, ind in enumerate(sad_smiles):
    inputs = val_loader_uno.dataset[ind]
    inputs = torch.FloatTensor([inputs]).permute(0,3,1,2).to(device)
    pred,latent = model(inputs)
    sad_embeddings[k] = latent[0].cpu().detach().numpy()

happy_emb = happy_embeddings.mean(axis=0)
sad_emb = sad_embeddings.mean(axis=0)
sad_to_happy_vec = happy_emb - sad_emb

plt.figure(figsize=(15, 5*len(test_POIs)))
for k, ind in enumerate(test_POIs):
    inputs = val_loader_uno.dataset[ind]
    inputs = torch.FloatTensor([inputs]).permute(0,3,1,2).to(device)
    pred,latent = model(inputs)
    latent = latent[0].cpu().detach().numpy()
    latent+=sad_to_happy_vec
    latent = [latent]
    latent = torch.FloatTensor(latent).to(device)
    output = model.decoder(latent)

    plt.subplot(len(test_POIs), 2, 2*k+1)
    plt.imshow(inputs.permute(0,2,3,1).cpu().detach().numpy()[0])
    plt.title('Real, number {}'.format(k))
    plt.axis('off')

    plt.subplot(len(test_POIs), 2, 2*k+2)

    plt.imshow(output.permute(0,2,3,1)[0].cpu().detach().numpy())
    plt.title('Output')
    plt.axis('off')

plt.show()
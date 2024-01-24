import torch
import torch.nn as nn
import torch.nn.functional as F


def accuracy(outputs, labels): #output et label sont des liste de tenseur et de label(int)
    _, preds = torch.max(outputs, dim=1) #_ pour une variabel dont on se sert pas et pred l'indice du max du tenseur
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))#torch sum renvoie un tenseur a un element et on veur un entier ==> .item()


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        with torch.no_grad():  #permet d'avoir des tenseurs independant des tenseurs de base pour ne pas calculer le gradient
            images, labels = batch 
            out = self(images)                    # Generate predictions
            loss = F.cross_entropy(out, labels)   # Calculate loss
            acc = accuracy(out, labels)           # Calculate accuracy
            return {'val_loss': loss, 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

def conv_block(in_channels, out_channels,kernel_size=3, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1), #on applique le kernel(cf pense bete kernel) et le padding (stride de 1 par defaut)
              nn.BatchNorm2d(out_channels), #on normalize, comme tout le monde fait(le pourquoi du comment on s'en fout)
              nn.ReLU(inplace=True)]#on applique relu, classique
    if pool: layers.append(nn.MaxPool2d(2)) #maxpooling=> reduire les dimension pour eviter la surcharge de calcul et l'overfitting ( a voir quand on s'en sert)
    return nn.Sequential(*layers)


class ResNet20(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
            super().__init__()

            self.conv0 = conv_block(in_channels, 64,7,pool = True)

            self.res1 = nn.Sequential(conv_block(64, 64), conv_block(64, 64))
            self.conv1= conv_block(64,64)

            self.res15 = nn.Sequential(conv_block(64, 128), conv_block(128, 128))
            self.conv15 = conv_block(64,128)

            self.res2 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
            self.conv2 = conv_block(128,128)

            self.res25 = nn.Sequential(conv_block(128, 256), conv_block(256, 256))
            self.conv25 = conv_block(128,256)

            self.res3 = nn.Sequential(conv_block(256,256), conv_block(256, 256))
            self.conv3 = conv_block(256,256)

            self.res35 = nn.Sequential(conv_block(256, 512), conv_block(512, 512))
            self.conv35 = conv_block( 256,512)

            self.res4 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
            self.conv4 = conv_block(512, 512)

            self.classifier = nn.Sequential(nn.AvgPool2d(8), 
                                            nn.Flatten(), 
                                            nn.Linear(512, num_classes))
            
    def forward(self, xb):
        out = self.conv0(xb)
        out = self.res1(out) + self.conv1(out)
        out = self.res1(out) + self.conv1(out)
        out = self.res1(out) + self.conv1(out)
        out = self.res15(out) + self.conv15(out)
        out = self.res2(out) + self.conv2(out)
        out = self.res2(out) + self.conv2(out)
        out = self.res2(out) + self.conv2(out)
        out = self.res2(out) + self.conv2(out)
        out = self.res25(out) + self.conv25(out)
        out = self.res3(out) + self.conv3(out)
        out = self.res3(out) + self.conv3(out)
        out = self.res3(out) + self.conv3(out)
        out = self.res3(out) + self.conv3(out)
        out = self.res3(out) + self.conv3(out)
        out = self.res35(out) + self.conv35(out)
        out = self.res4(out) + self.conv4(out)
        out = self.classifier(out)
        return out


def evaluate(model, val_loader):
    model.eval()#pas oublier le evaal() qui desactive le dropout et le calcul de gradient
    outputs = [model.validation_step(batch) for batch in val_loader]
    model.train() #on oublie pas de la repasser en mode train
    return model.validation_epoch_end(outputs)



# def function_modele(nclass=13,lr=0.01):

#     inputsize = 256*256
#     nunit = 10

#     #mlp
#     model=nn.Sequential(
#         nn.Linear(inputsize,nunit),
#         nn.ReLU(),
#         nn.Linear(nunit,nclass))

#     optimizer = torch.optim.SGD(model.parameters(), lr=lr)

#     return model,optimizer

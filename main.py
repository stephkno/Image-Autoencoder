import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import time
import cv2
import keyboard
import random
from PIL import Image
import numpy
import atexit

trainset = []
testset = []
categories = []
n_categories = 1
batch_size = 6
workers = 1
debug = False

eps = numpy.finfo(numpy.float32).eps.item()

#score_graph = pg.plot(title="stats")

def gaussian(x):
    noise = torch.autograd.Variable(x.data.new(x.size()).normal_(1.0, 0.02))
    return x + noise
class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()

        # encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=8, stride=2, bias=True),
            torch.nn.LeakyReLU(0.2),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 64, kernel_size=6, stride=4, bias=True),
            torch.nn.LeakyReLU(0.2),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 128, kernel_size=8, stride=6, bias=True),
            torch.nn.Tanh()
        )
        # decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, bias=True),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(64),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2, bias=True),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(32),
            torch.nn.ConvTranspose2d(32, 16, kernel_size=6, stride=3, bias=True),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(16),
            torch.nn.ConvTranspose2d(16, 3, kernel_size=8, stride=4, bias=True),
            torch.nn.Tanh()
        )

    def forward(self, x, eval):

        x = gaussian(x)
        if not eval:
            if debug:
                print("In", x.view(-1).shape)
            x = self.encoder(x)

        ret_vec = x

        x = torch.sub(x, x.mean())
        x = torch.div(x, x.std())
        x = torch.add(x, eps)

        if debug:
            print("Hidden", x.view(-1).shape)
        x = self.decoder(x.view(batch_size, 128, 4, 4))
        if debug:
            print("Out", x.view(-1).shape)
        return x, ret_vec

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        #torch.nn.init.xavier_uniform(m.weight)
        #m.weight.data.fill_(0.0)
    if classname.find('ConvTranspose2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        #torch.nn.init.xavier_uniform(m.weight)
        #m.weight.data.fill_(0.0)
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        #torch.nn.init.xavier_uniform(m.weight)
        #m.weight.data.fill_(0.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

print("CUDA: {}".format(torch.cuda.is_available()))
print("Loading images")

#dataset = torchvision.datasets.cifar.CIFAR10(root='./data', download=True, transform=transforms.ToTensor())
dataset = torchvision.datasets.ImageFolder('/Users/stephen/Documents/code/pytorch/data/outdoors', transform=transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

classes = []
losses = []
avges = []
times = []
x_domain = []

encoder = EncoderDecoder().cuda()
# weights_init(encoder)

avg_sum = 0.0
count = 1

loss_func = torch.nn.modules.loss.MSELoss()
optimizer = torch.optim.Adam(lr=0.01, params=encoder.parameters(), weight_decay=1e-15)

# functions to show an image
def imshow(img):
    img = cv2.resize(img, (128, 128*batch_size), interpolation=cv2.INTER_AREA)
    img = img[..., ::-1]
    cv2.imshow('image', img)
    cv2.waitKey(1)

# Show the tensor.
def showTensor(t):
    plt.figure()
    plt.imshow(t.numpy())
    plt.colorbar()
    plt.show()

epochs = 100
epoch = 0
#dataset = [i for i in dataset if i[1] == 0]
dataset_size = len(dataset)
img_count = 0


def update_graph():
    #score_graph.plot(x_domain, avges, pen=(1, 1))
    #score_graph.plot(x_domain, losses, pen=(1, 2))
    #score_graph.plot(x_domain, times, pen=(1, 3))
    pass

def display_outputs(img):
    #print("\r{}/{}".format(count, dataset_size), end="")
    img = img.detach().cpu()
    img = torch.cat(img.detach().cpu().unbind(),1)
    imshow(img.permute(1, 2, 0).detach().numpy())

def save():
    import datetime
    print("Saving encoder...")
    path = "./encoder-{}".format(datetime.datetime.now())

    torch.save(encoder.state_dict(), path)

atexit.register(save)

for i in range(100):
    epoch += 1
    images_list = list(range(dataset_size))

    for _ in range(int(dataset_size/batch_size)):
        start = time.time()
        inp = []

        for i in range(batch_size):

            z = random.choice(images_list)
            images_list.remove(z)
            img_app = dataset[z][0]

            inp.append(img_app.unsqueeze(0))
            img_count += 1
        count += 1

        inp = torch.cat(inp).cuda()

        #get data and target
        target = inp.cuda()

        #forward pass
        out, code = encoder(inp.cuda(), False)

        #backward pass
        loss = loss_func(out, target)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)

        if count % 10 == 0:
            print("\rEpoch {} - Step {}/{} - Loss {}".format(epoch, count, (dataset_size/batch_size), avg_sum/count),end="")

        torch.nn.utils.clip_grad_value_(encoder.parameters(), 1.0)
        optimizer.step()

        #update statistics
        avg_sum += loss.item()
        avges.append(avg_sum/count)
        losses.append(loss.item())
        x_domain.append(count)

        #cleanup
        end = time.time()

        times.append(end-start)

        # display statistics
        if not keyboard.is_pressed(' '):
            if count % 10 == 0:
                display_outputs(out)
        elif keyboard.is_pressed(' '):
            update_graph()
            #i = torch.zeros(4,128,26,26).cuda()
            #i[0][0][0][0] = 1.0
            #o, _ = encoder.forward(i, True)
            #display_outputs(o)

        elif keyboard.is_pressed('X'):
            torch.cuda.empty_cache()
            quit()

        del out, inp
        torch.cuda.empty_cache()

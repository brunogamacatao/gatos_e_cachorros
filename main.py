from datetime import datetime
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from util import dateBack
from PIL import Image

ARQUIVO_REDE = 'gatos_cachorros.pth'
NOMES_LABELS = ['gato', 'cachorro']
data_path    = 'treinamento_noob'

transformacoes = transforms.Compose([
  transforms.Resize([80, 80]),
  transforms.ToTensor()
])

train_dataset = torchvision.datasets.ImageFolder(
  root=data_path,
  transform=transformacoes
)

train_loader = torch.utils.data.DataLoader(
  train_dataset,
  batch_size=5,
  num_workers=4,
  shuffle=True
)

class GatosCachorrosModel(nn.Module):
  def __init__(self):
    super(GatosCachorrosModel, self).__init__()
    self.conv1 = nn.Conv2d(3, 6, 5) # canais, qtd filtros, kernel
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.linear1 = nn.Linear(4624, 120)
    self.linear2 = nn.Linear(120, 84)
    self.linear3 = nn.Linear(84, 2)
    self.pool = nn.MaxPool2d(2, 2)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 4624)
    x = F.relu(self.linear1(x))
    x = F.relu(self.linear2(x))
    x = self.linear3(x)
    return x

rede = GatosCachorrosModel()

def total_certo(labels, saida):
  total = 0
  for i, val in enumerate(saida):
    val = val.tolist()
    max_idx = val.index(max(val))
    if labels[i] == max_idx:
      total += 1
  return total

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(rede.parameters(), lr=0.001, momentum=0.9)

def treina(epochs = 100):
  for epoch in range(epochs):
    erro_total = 0
    acertou = 0
    total = 0
    for batch, (entrada, label) in enumerate(train_loader):
      optimizer.zero_grad()
      saida = rede(entrada)
      acertou += total_certo(label, saida)
      total += len(label)
      erro = criterion(saida, label)
      erro.backward()
      optimizer.step()
      erro_total += erro.item()
    print('Epoch: {} - Erro: {:.4f} - Acuracia: {:.2f}%'.
        format(epoch + 1, erro_total, 100.0 * acertou / total))

def load_image(path):
  image = Image.open(path)
  image = transformacoes(image).float()
  image = Variable(image, requires_grad=True)
  image = image.unsqueeze(0)
  return image

def carrega_rede():
  rede.load_state_dict(torch.load(ARQUIVO_REDE))
  rede.eval()

def testa_imagem(path):
  imagem = load_image(path)
  saida = rede(imagem)
  val = saida.squeeze().tolist()  
  max_idx = val.index(max(val))
  return NOMES_LABELS[max_idx]

def exibe_menu():
  while True:
    print('1. Treinar')
    print('2. Testar a rede')
    print('3. Sair')
    opcao = input('Digite sua opcao: ')
    if opcao == '1':
      inicio = datetime.now()
      treina(100)
      fim = datetime.now()
      print('Rede treinada em {}'.format(dateBack(inicio, fromDate=fim)))
      print('Salvando a rede ...')
      torch.save(rede.state_dict(), ARQUIVO_REDE)
      print('Rede salva com sucesso')
    elif opcao == '2':
      print('Carregando a rede ...')
      carrega_rede()
      print(testa_imagem(input('Digite o caminho da imagem: ')))
    elif opcao == '3':
        break
    else:
        print('Digite uma opcao valida')

if __name__ == '__main__':
  exibe_menu()
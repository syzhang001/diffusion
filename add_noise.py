import torch
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from PIL import Image

betas = torch.linspace(0.02,0.1,1000).double()
alphas = 1-betas
alphas_bar = torch.cumprod(alphas,dim=0)
sqrt_alphas_bar = torch.sqrt(alphas_bar)
sqrt_m1_alphas_bar = torch.sqrt(1 - alphas_bar)

img = Image.open('cat.png')
trans = transforms.Compose(
    [transforms.Resize(224),
     transforms.ToTensor()]
)

x_0 = trans(img)
img_list = [x_0]
noise = torch.randn_like(x_0)
for i in range(15):
    x_t = sqrt_alphas_bar[i]*x_0 + sqrt_m1_alphas_bar[i]*noise
    img_list.append(x_t)
all_img = torch.stack(img_list,dim=0)
all_img = make_grid(all_img)
save_image(all_img, 'car_noise.png')
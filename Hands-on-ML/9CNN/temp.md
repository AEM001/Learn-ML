# 风格迁移
# 该工具包中有AlexNet、VGG等多种训练好的CNN网络
from torchvision import models 
import copy

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"使用设备：{device}")
# 定义图像处理方法
transform = transforms.Resize([512, 512]) # 规整图像形状

def loadimg(path):  
    # 加载路径为path的图像，形状为H*W*C
    img = plt.imread(path)
    # 处理图像，注意重排维度使通道维在最前
    img = transform(torch.tensor(img).permute(2, 0, 1))
    # 展示图像
    plt.imshow(img.permute(1, 2, 0).numpy())
    plt.show()
    # 添加batch size维度
    img = img.unsqueeze(0).to(dtype=torch.float32)
    img /= 255 # 将其值从0-255的整数转换为0-1的浮点数
    return img

content_image_path = 'content.jpg'
style_image_path = 'style.jpg'

# 加载内容图像
print('内容图像')
content_img = loadimg(content_image_path)
content_img = content_img.to(device)
# 加载风格图像
print('风格图像') 
style_img = loadimg(style_image_path)
style_img = style_img.to(device)

---
# 内容损失
class ContentLoss(nn.Module):

    def __init__(self, target):
        # target为从目标图像中提取的内容特征
        super().__init__()
        # 我们不对target求梯度，因此将target从梯度的计算图中分离出来
        self.target = target.detach() 
        self.criterion = nn.MSELoss()

    def forward(self, x):
         # 利用MSE计算输入图像与目标内容图像之间的损失
        self.loss = self.criterion(x.clone(), self.target) 
        return x # 只计算损失，不改变输入

    def backward(self): 
        # 由于本模块只包含损失计算，不改变输入，因此要单独定义反向传播
        self.loss.backward(retain_graph=True)
        return self.loss


def gram(x):
    # 计算G矩阵
    batch_size, n, w, h = x.shape # n为卷积核数目，w和h为输出的宽和高
    f = x.view(batch_size * n, w * h) # 变换为二维
    g = f @ f.T / (batch_size * n * w * h) # 除以参数数目，进行归一化
    return g


# 风格损失
class StyleLoss(nn.Module):

    def __init__(self, target):
        # target为从目标图像中提取的风格特征
        # weight为设置的强度系数lambda
        super().__init__()
        
        self.target = target.detach().to(device)
        self.target_gram = gram(target.detach().to(device))
        self.criterion = nn.MSELoss()

    def forward(self, x):
        input_gram = gram(x.clone()) # 输入的Gram矩阵
        self.loss = self.criterion(input_gram, self.target_gram)
        return x

    def backward(self):
        self.loss.backward(retain_graph=True)
        return self.loss

---
vgg16 = models.vgg16(weights=True).features # 导入预训练的VGG16网络

# 选定用于提取特征的卷积层，Conv_13对应着第5块的第3卷积层
content_layer = ['Conv_13']
# 下面这些层分别对应第1至5块的第1卷积层
style_layer = ['Conv_1', 'Conv_3', 'Conv_5', 'Conv_8', 'Conv_11']

content_losses = [] # 内容损失
style_losses = [] # 风格损失
content_loss = ContentLoss(target).to(device)
style_loss = StyleLoss(target).to(device)

model = nn.Sequential() # 储存新模型的层
vgg16 = copy.deepcopy(vgg16)
model = nn.Sequential().to(device)
vgg16 = copy.deepcopy(vgg16).to(device)
index = 1  # 计数卷积层

# 遍历 VGG16 的网络结构，选取需要的层
for layer in list(vgg16):
    if isinstance(layer, nn.Conv2d): # 如果是卷积层
        name = "Conv_" + str(index)
        model.append(layer)
        if name in content_layer:  
            # 如果当前层用于抽取内容特征，则添加内容损失
            target = model(content_img).clone() # 计算内容图像的特征
            content_loss = ContentLoss(target) # 内容损失模块
            model.append(content_loss)
            content_losses.append(content_loss)

        if name in style_layer:  
            # 如果当前层用于抽取风格特征，则添加风格损失
            target = model(style_img).clone()
            style_loss = StyleLoss(target) # 风格损失模块
            model.append(style_loss)  
            style_losses.append(style_loss) 

    if isinstance(layer, nn.ReLU): # 如果激活函数层
        model.append(layer)
        index += 1

    if isinstance(layer, nn.MaxPool2d): # 如果是池化层
        model.append(layer)

# 输出模型结构
print(model)
---

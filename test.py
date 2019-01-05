from torchvision import models

model = models.vgg16(pretrained=True)

#print(model.classifier)
print(model.classifier[0].in_features)
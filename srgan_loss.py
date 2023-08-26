import torch
from torch import nn
from torchvision.models.vgg import vgg16


class SRGANLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(SRGANLoss, self).__init__()
        self.vgg = vgg16(pretrained=True)
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, out_labels, out_images, target_images):
        content_loss = self.content_loss(out_images, target_images)
        adversarial_loss = self.adversarial_loss(out_labels, torch.ones_like(out_labels))
        tv_loss = self.tv_loss(out_images)
        return content_loss + 1e-3 * adversarial_loss + 2e-8 * tv_loss

    def content_loss(self, out_images, target_images):
        out_features = self.vgg(out_images)
        target_features = self.vgg(target_images).detach()
        loss = self.mse_loss(out_features, target_features)
        return loss

    def tv_loss(self, x):
        return self.tv_loss_weight * 0.5 * (
            torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean() +
            torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
        )

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import copy
import pydiffvg
import argparse

def main(args):
    pydiffvg.set_use_gpu(torch.cuda.is_available())

    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(args.content_file)
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)
    render = pydiffvg.RenderFunction.apply
    img = render(canvas_width, # width
                 canvas_height, # height
                 2,   # num_samples_x
                 2,   # num_samples_y
                 0,   # seed
                 None,
                 *scene_args)
    # Transform to gamma space
    pydiffvg.imwrite(img.cpu(), 'results/style_transfer/init.png', gamma=1.0)
    # HWC -> NCHW
    img = img.unsqueeze(0)
    img = img.permute(0, 3, 1, 2) # NHWC -> NCHW

    loader = transforms.Compose([
        transforms.ToTensor()])  # transform it into a torch tensor

    def image_loader(image_name):
        image = Image.open(image_name)
        # fake batch dimension required to fit network's input dimensions
        image = loader(image).unsqueeze(0)
        return image.to(pydiffvg.get_device(), torch.float)

    style_img = image_loader(args.style_img)
    # alpha blend content with a gray background
    content_img = img[:, :3, :, :] * img[:, 3, :, :] + \
                  0.5 * torch.ones([1, 3, img.shape[2], img.shape[3]]) * \
                  (1 - img[:, 3, :, :])

    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

    unloader = transforms.ToPILImage()  # reconvert into PIL image

    class ContentLoss(nn.Module):
        def __init__(self, target,):
            super(ContentLoss, self).__init__()
            # we 'detach' the target content from the tree used
            # to dynamically compute the gradient: this is a stated value,
            # not a variable. Otherwise the forward method of the criterion
            # will throw an error.
            self.target = target.detach()

        def forward(self, input):
            self.loss = F.mse_loss(input, self.target)
            return input

    def gram_matrix(input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

    class StyleLoss(nn.Module):

        def __init__(self, target_feature):
            super(StyleLoss, self).__init__()
            self.target = gram_matrix(target_feature).detach()

        def forward(self, input):
            G = gram_matrix(input)
            self.loss = F.mse_loss(G, self.target)
            return input

    device = pydiffvg.get_device()
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # create a module to normalize input image so we can easily put it in a
    # nn.Sequential
    class Normalization(nn.Module):
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            # .view the mean and std to make them [C x 1 x 1] so that they can
            # directly work with image Tensor of shape [B x C x H x W].
            # B is batch size. C is number of channels. H is height and W is width.
            self.mean = mean.clone().view(-1, 1, 1)
            self.std = std.clone().view(-1, 1, 1)

        def forward(self, img):
            # normalize img
            return (img - self.mean) / self.std

    # desired depth layers to compute style/content losses :
    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                   style_img, content_img,
                                   content_layers=content_layers_default,
                                   style_layers=style_layers_default):
        cnn = copy.deepcopy(cnn)

        # normalization module
        normalization = Normalization(normalization_mean, normalization_std).to(device)

        # just in order to have an iterable access to or list of content/syle
        # losses
        content_losses = []
        style_losses = []

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization)

        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses

    def run_style_transfer(cnn, normalization_mean, normalization_std,
                           content_img, style_img,
                           canvas_width, canvas_height,
                           shapes, shape_groups,
                           num_steps=500, style_weight=5000, content_weight=1):
        """Run the style transfer."""
        print('Building the style transfer model..')
        model, style_losses, content_losses = get_style_model_and_losses(cnn,
            normalization_mean, normalization_std, style_img, content_img)
        point_params = []
        color_params = []
        stroke_width_params = []
        for shape in shapes:
            if isinstance(shape, pydiffvg.Path):
                point_params.append(shape.points.requires_grad_())
                stroke_width_params.append(shape.stroke_width.requires_grad_())
        for shape_group in shape_groups:
            if isinstance(shape_group.fill_color, torch.Tensor):
                color_params.append(shape_group.fill_color.requires_grad_())
            elif isinstance(shape_group.fill_color, pydiffvg.LinearGradient):
                point_params.append(shape_group.fill_color.begin.requires_grad_())
                point_params.append(shape_group.fill_color.end.requires_grad_())
                color_params.append(shape_group.fill_color.stop_colors.requires_grad_())
            if isinstance(shape_group.stroke_color, torch.Tensor):
                color_params.append(shape_group.stroke_color.requires_grad_())
            elif isinstance(shape_group.stroke_color, pydiffvg.LinearGradient):
                point_params.append(shape_group.stroke_color.begin.requires_grad_())
                point_params.append(shape_group.stroke_color.end.requires_grad_())
                color_params.append(shape_group.stroke_color.stop_colors.requires_grad_())

        point_optimizer = optim.Adam(point_params, lr=1.0)
        color_optimizer = optim.Adam(color_params, lr=0.01)
        stroke_width_optimizers = optim.Adam(stroke_width_params, lr=0.1)
        print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:
            point_optimizer.zero_grad()
            color_optimizer.zero_grad()
            stroke_width_optimizers.zero_grad()

            scene_args = pydiffvg.RenderFunction.serialize_scene(\
                canvas_width, canvas_height, shapes, shape_groups)
            render = pydiffvg.RenderFunction.apply
            img = render(canvas_width, # width
                         canvas_height, # height
                         2,   # num_samples_x
                         2,   # num_samples_y
                         0,   # seed
                         None,
                         *scene_args)
            # alpha blend img with a gray background
            img = img[:, :, :3] * img[:, :, 3:4] + \
                  0.5 * torch.ones([img.shape[0], img.shape[1], 3]) * \
                  (1 - img[:, :, 3:4])

            pydiffvg.imwrite(img.cpu(),
                             'results/style_transfer/step_{}.png'.format(run[0]),
                             gamma=1.0)

            # HWC to NCHW
            img = img.permute([2, 0, 1]).unsqueeze(0)
            model(img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 1 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            point_optimizer.step()
            color_optimizer.step()
            stroke_width_optimizers.step()

            for color in color_params:
                color.data.clamp_(0, 1)
            for w in stroke_width_params:
                w.data.clamp_(0.5, 4.0)

        return shapes, shape_groups

    shapes, shape_groups = run_style_transfer(\
        cnn, cnn_normalization_mean, cnn_normalization_std,
        content_img, style_img,
        canvas_width, canvas_height, shapes, shape_groups)

    scene_args = pydiffvg.RenderFunction.serialize_scene(shapes, shape_groups)
    render = pydiffvg.RenderFunction.apply
    img = render(canvas_width, # width
                 canvas_height, # height
                 2,   # num_samples_x
                 2,   # num_samples_y
                 0,   # seed
                 None,
                 *scene_args)
    # Transform to gamma space
    pydiffvg.imwrite(img.cpu(), 'results/style_transfer/output.png', gamma=1.0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("content_file", help="source SVG path")
    parser.add_argument("style_img", help="target image path")
    args = parser.parse_args()
    main(args)

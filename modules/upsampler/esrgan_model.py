import os

import numpy as np
import torch
from PIL import Image
from basicsr.utils.download_util import load_file_from_url

from modules import images, devices
from modules.upsampler.upscaler import Upscaler, UpscalerData
from modules.cmd_opts import opts

# this file is taken from https://github.com/xinntao/ESRGAN

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.utils import modelloader


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out


def fix_model_layers(crt_model, pretrained_net):
    # this code is adapted from https://github.com/xinntao/ESRGAN
    if 'conv_first.weight' in pretrained_net:
        return pretrained_net

    if 'model.0.weight' not in pretrained_net:
        is_realesrgan = "params_ema" in pretrained_net and 'body.0.rdb1.conv1.weight' in pretrained_net["params_ema"]
        if is_realesrgan:
            raise Exception("The file is a RealESRGAN model, it can't be used as a ESRGAN model.")
        else:
            raise Exception("The file is not a ESRGAN model.")

    crt_net = crt_model.state_dict()
    load_net_clean = {}
    for k, v in pretrained_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    pretrained_net = load_net_clean

    tbd = []
    for k, v in crt_net.items():
        tbd.append(k)

    # directly copy
    for k, v in crt_net.items():
        if k in pretrained_net and pretrained_net[k].size() == v.size():
            crt_net[k] = pretrained_net[k]
            tbd.remove(k)

    crt_net['conv_first.weight'] = pretrained_net['model.0.weight']
    crt_net['conv_first.bias'] = pretrained_net['model.0.bias']

    for k in tbd.copy():
        if 'RDB' in k:
            ori_k = k.replace('RRDB_trunk.', 'model.1.sub.')
            if '.weight' in k:
                ori_k = ori_k.replace('.weight', '.0.weight')
            elif '.bias' in k:
                ori_k = ori_k.replace('.bias', '.0.bias')
            crt_net[k] = pretrained_net[ori_k]
            tbd.remove(k)

    crt_net['trunk_conv.weight'] = pretrained_net['model.1.sub.23.weight']
    crt_net['trunk_conv.bias'] = pretrained_net['model.1.sub.23.bias']
    crt_net['upconv1.weight'] = pretrained_net['model.3.weight']
    crt_net['upconv1.bias'] = pretrained_net['model.3.bias']
    crt_net['upconv2.weight'] = pretrained_net['model.6.weight']
    crt_net['upconv2.bias'] = pretrained_net['model.6.bias']
    crt_net['HRconv.weight'] = pretrained_net['model.8.weight']
    crt_net['HRconv.bias'] = pretrained_net['model.8.bias']
    crt_net['conv_last.weight'] = pretrained_net['model.10.weight']
    crt_net['conv_last.bias'] = pretrained_net['model.10.bias']

    return crt_net


class UpscalerESRGAN(Upscaler):
    def __init__(self, dirname):
        self.name = "ESRGAN"
        self.model_url = "https://github.com/cszn/KAIR/releases/download/v1.0/ESRGAN.pth"
        self.model_name = "ESRGAN_4x"
        self.scalers = []
        self.user_path = dirname
        super().__init__()
        model_paths = self.find_models(ext_filter=[".pt", ".pth"])
        scalers = []
        if len(model_paths) == 0:
            scaler_data = UpscalerData(self.model_name, self.model_url, self, 4)
            scalers.append(scaler_data)
        for file in model_paths:
            if "http" in file:
                name = self.model_name
            else:
                name = modelloader.friendly_name(file)

            scaler_data = UpscalerData(name, file, self, 4)
            self.scalers.append(scaler_data)

    def do_upscale(self, img, selected_model):
        model = self.load_model(selected_model)
        if model is None:
            return img
        model.to(devices.device_esrgan)
        img = esrgan_upscale(model, img)
        return img

    def load_model(self, path: str):
        if "http" in path:
            filename = load_file_from_url(url=self.model_url, model_dir=self.model_path,
                                          file_name="%s.pth" % self.model_name,
                                          progress=True)
        else:
            filename = path
        if not os.path.exists(filename) or filename is None:
            print("Unable to load %s from %s" % (self.model_path, filename))
            return None

        pretrained_net = torch.load(filename, map_location='cpu' if devices.device_esrgan.type == 'mps' else None)
        crt_model = RRDBNet(3, 3, 64, 23, gc=32)

        pretrained_net = fix_model_layers(crt_model, pretrained_net)
        crt_model.load_state_dict(pretrained_net)
        crt_model.eval()

        return crt_model


def upscale_without_tiling(model, img):
    img = np.array(img)
    img = img[:, :, ::-1]
    img = np.moveaxis(img, 2, 0) / 255
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0).to(devices.device_esrgan)
    with torch.no_grad():
        output = model(img)
    output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = 255. * np.moveaxis(output, 0, 2)
    output = output.astype(np.uint8)
    output = output[:, :, ::-1]
    return Image.fromarray(output, 'RGB')


def esrgan_upscale(model, img):
    if opts.ESRGAN_tile == 0:
        return upscale_without_tiling(model, img)

    grid = images.split_grid(img, opts.ESRGAN_tile, opts.ESRGAN_tile, opts.ESRGAN_tile_overlap)
    newtiles = []
    scale_factor = 1

    for y, h, row in grid.tiles:
        newrow = []
        for tiledata in row:
            x, w, tile = tiledata

            output = upscale_without_tiling(model, tile)
            scale_factor = output.width // tile.width

            newrow.append([x * scale_factor, w * scale_factor, output])
        newtiles.append([y * scale_factor, h * scale_factor, newrow])

    newgrid = images.Grid(newtiles, grid.tile_w * scale_factor, grid.tile_h * scale_factor, grid.image_w * scale_factor, grid.image_h * scale_factor, grid.overlap * scale_factor)
    output = images.combine_grid(newgrid)
    return output

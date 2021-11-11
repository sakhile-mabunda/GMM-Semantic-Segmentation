
from __future__ import absolute_import, division
from .squeeze_extractor import *
from torch import nn


class _VGG(SqueezeExtractor):
	def __init__(self, model, features, fixed_feature=True):
		super(_VGG, self).__init__(model, features, fixed_feature)

	def get_copy_feature_info(self):

		lst_copy_feature_info = []
		for i in range(len(self.features)):
			if isinstance(self.features[i], nn.MaxPool2d):
				out_channels = self._get_last_conv2d_out_channels(self.features[:i])
				lst_copy_feature_info.append(CopyFeatureInfo(i, out_channels))
		return lst_copy_feature_info


def vgg_16(batch_norm=True, pretrained=False, fixed_feature=True):
	""" VGG 16-layer model from torchvision's vgg model.

	:param batch_norm: train model with batch normalization
	:param pretrained: if true, return a model pretrained on ImageNet
	:param fixed_feature: if true and pretrained is true, model features are fixed while training.
	"""
	if batch_norm:
		from torchvision.models.vgg import vgg16_bn
		model = vgg16_bn(pretrained)
	else:
		from torchvision.models.vgg import vgg16
		model = vgg16(pretrained)

	ff = True if pretrained and fixed_feature else False
	return _VGG(model, model.features, ff)


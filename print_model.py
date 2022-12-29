# Copyright (c) Alibaba, Inc. and its affiliates.

import sys

import torch

from modelscope.models.audio.kws.farfield.fsmn_sele_v2 import FSMNSeleNetV2

model = torch.load(sys.argv[1], map_location='cpu')
model.print_header()
model.print_model()

# Benchmarking ConvNeXt models on FGVC datasets

This repository adds data-loader supports for 3 fine-grained visual classification datasets for training ConvNeXt. You can train ConvNeXt using:
<ul>
  <li>
    CUB dataset
  </li>
    <li>
    FoodX-251 dataset
  </li>
    <li>
    CUB + Stanford Dogs dataset (a custom dataset containing categories from both CUB and Dogs datasets providing 320 classes)
    </li>
  </ul>

This repo was mainly designed to make a strong ConvNeXt teacher model trained on FGVC dataset and use it to distill knowledge into DeiT model. Please refer here to our DeiT Knowledge distillation repository.

--- 
Acknowledgement:
This repo is based on official [ConvNeXt repository](https://github.com/facebookresearch/ConvNeXt)


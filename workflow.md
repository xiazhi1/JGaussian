# workflow

## introduction

这个文档的设计是为了让作者在进行项目修改时，有一个连贯完整的调试思路和编写思路

## train.py

目前train.py主要是直接照搬GaussianSPlatting的代码，具体修改思路是暂时不管tensorboard和NetworkGui部分，先修改核心的GaussianModel与SceneModel
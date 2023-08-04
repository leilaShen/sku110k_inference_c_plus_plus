# sku110k_inference_c_plus_plus
Rewrite sku110k inferencer using c++

sku110k 论文：《Precise Detection in Densely Packed Scenes》 github网址：https://github.com/eg4000/SKU110K_CVPR19 这个仓库重写了EMMerger部分，将python变成了c++代码，整个inference的速度从python版本的1s提升至0.3m左右
为了保证重写后c++的代码和原始代码大体一致 易于阅读理解，没有严格遵循c++的编码规范

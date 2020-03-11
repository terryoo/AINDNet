## Code
### Trained Model
[**Trained Model**](https://drive.google.com/drive/folders/1vajSO5pDaGDvTbL9caPe7uXi5TB-P-U9?usp=sharing)

The quantitative performance of uploaded model would be higher than reported performance at the paper with fixing minor bugs.

| Datasets      | Paper (dB)           | Uploaded (dB)  |
| ------------- |:-------------:| -----:|
| DND      | 39.37 | 39.46 |
| SIDD Validation      | 38.90       | 38.96  |
| SIDD Test | 38.95     | 39.01 |

### Test
Ready for the input data.
python test_images.py --ckpt [trained model] --input_dir [Noisy Image directory] --output_dir [save directory]  --gpu_id [GPU number]

[Options]
```
python test_images.py --ckpt [trained model] --input_dir [Noisy Image directory] --output_dir [save directory]  --gpu_id [GPU number]
--ckpt: Path of trained model [Default: ./checkpoint/SIDD_transfer/]
--input_dir: Directory of input images [Default: ./testset/RNI15/]
--output_dir: Directory for the output images. [Default: ./result/RNI15/]
--gpu: If you have more than one gpu in your computer, the number designates the index of GPU which is going to be used. [Default 0]
```

### An example of test codes
```
python test_images.py --ckpt SIDD_transfer --input_dir ./testset/RNI15 --output_dir ./results/RNI15/ --gpu_id 0
```

## Test Set
The real-noise benchmarks can be downloaded at [**DND**](https://noise.visinf.tu-darmstadt.de/)
and [**SIDD**](https://www.eecs.yorku.ca/~kamel/sidd/benchmark.php).
RNI15 is attached in testsets.

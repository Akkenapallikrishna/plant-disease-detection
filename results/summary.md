| Model | Colour | Noise aug. | Val acc | Test Top-1 | Test Top-5 | Test macro-F1 | Test ROC-AUC | Noisy-test Top-1 |
|---|---|---|---|---|---|---|---|---|
| resnet50 | RGB | no | 91.6% | 93.3% | 100.0% | 0.932 | 0.998 | 88.0% |
| densenet121 | RGB | no | 88.4% | 92.4% | 100.0% | 0.923 | 0.998 | 86.2% |
| densenet121 | HSV | no | 85.8% | 88.0% | 99.1% | 0.876 | 0.995 | 36.9% |
| resnet50 | HSV | no | 85.8% | 85.3% | 99.1% | 0.850 | 0.991 | 57.3% |
| resnet50 | RGB | yes | 83.6% | 84.9% | 99.1% | 0.842 | 0.994 | 90.2% |
| densenet121 | HSV | yes | 77.3% | 73.8% | 97.3% | 0.705 | 0.987 | 61.8% |
| resnet50 | HSV | yes | 75.6% | 72.9% | 97.3% | 0.718 | 0.974 | 61.3% |

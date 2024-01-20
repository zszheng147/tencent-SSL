# Architecture
```
.
├── dual-mic-SSL
├── FRA-RIR
├── librispeech
└── reverbs
```
* librispeech: clean speech
* reverbs: offline rir data
* FRA-RIR: code for generating rir
* dual-mic-SSL: training code


## Dataset Generation
### Speech Source
Use Librispeech dev-clean as training data, and use its test-clean as test data.

### RIR Data
Use [FRA-RIR](https://github.com/tencent-ailab/FRA-RIR) to generate RIR data.  
Specifically, set parameters as follows.
```python
simu_config = {
    "min_max_room": [[3, 3, 2.5], [10, 6, 4]],
    "rt60": [0.1, 0.7],
    "sr": 16000,
    "mic_dist": [0.2, 5.0],
    "num_src": 2,
    "mic_arch": [[-0.05, 0, 0], [0.05, 0, 0]],
    "min_dis_wall": [0.5, 0.5, 0.5]
}
```
Then uniformly generate room_dim, array_pos (center of the microphone array), and src_pos.
Note: we always fix the microphone on the x-axis and there is no specific directions.

In total, generate 3,000 rirs for training, and 200 rirs for eval.   
Please ref to FRA-RIR/test_samples.py in this repo to automatically generate.

## Model Architecture
Use 2D convultion to do downsample, and employ 8-layer conformer as backbone architecture. The speech tokens will be concatenated with two \[CLS\] tokens before feeding into conformer. The two \[CLS\] tokens are designed to predict the two azimuths.

```
SSLModel(
  (stft): STFT(
    (conv_real): Conv1d(1, 257, kernel_size=(512,), stride=(160,), bias=False)
    (conv_imag): Conv1d(1, 257, kernel_size=(512,), stride=(160,), bias=False)
  )
  (inter_conv): Sequential(
    (0): Conv2d(2, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): GELU(approximate='none')
    (3): Conv2d(4, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): GELU(approximate='none')
  )
  (patch_embed): Sequential(
    (0): Conv2d(8, 32, kernel_size=(4, 4), stride=(4, 4))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): GELU(approximate='none')
    (3): Conv2d(32, 128, kernel_size=(4, 4), stride=(4, 4))
    (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): GELU(approximate='none')
    (6): Conv2d(128, 512, kernel_size=(4, 4), stride=(4, 4))
    (7): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): GELU(approximate='none')
  )
  (backbone): Conformer(
    (layers): ModuleList(
      (0-7): 8 x ConformerBlock(
        (ff1): Scale(
          (fn): PreNorm(
            (fn): FeedForward(
              (net): Sequential(
                (0): Linear(in_features=512, out_features=2048, bias=True)
                (1): Swish()
                (2): Dropout(p=0.0, inplace=False)
                (3): Linear(in_features=2048, out_features=512, bias=True)
                (4): Dropout(p=0.0, inplace=False)
              )
            )
            (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          )
        )
        (attn): PreNorm(
          (fn): Attention(
            (to_q): Linear(in_features=512, out_features=512, bias=False)
            (to_kv): Linear(in_features=512, out_features=1024, bias=False)
            (to_out): Linear(in_features=512, out_features=512, bias=True)
            (rel_pos_emb): Embedding(1025, 64)
            (dropout): Dropout(p=0.0, inplace=False)
          )
          (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        )
        (conv): ConformerConvModule(
          (net): Sequential(
            (0): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (1): Rearrange('b n c -> b c n')
            (2): Conv1d(512, 2048, kernel_size=(1,), stride=(1,))
            (3): GLU()
            (4): DepthWiseConv1d(
              (conv): Conv1d(1024, 1024, kernel_size=(31,), stride=(1,), groups=1024)
            )
            (5): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (6): Swish()
            (7): Conv1d(1024, 512, kernel_size=(1,), stride=(1,))
            (8): Rearrange('b c n -> b n c')
            (9): Dropout(p=0.0, inplace=False)
          )
        )
        (ff2): Scale(
          (fn): PreNorm(
            (fn): FeedForward(
              (net): Sequential(
                (0): Linear(in_features=512, out_features=2048, bias=True)
                (1): Swish()
                (2): Dropout(p=0.0, inplace=False)
                (3): Linear(in_features=2048, out_features=512, bias=True)
                (4): Dropout(p=0.0, inplace=False)
              )
            )
            (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          )
        )
        (post_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      )
    )
  )
  (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
  (head): Linear(in_features=512, out_features=36, bias=True)
)
```
## How to train
```bash
cd dual-mic-SSL/
bash train.sh
```

## Perfermance
I train the model for 100 epoch, and use MAE as metrics to evaluate. Result can be viewed in `dual-mic-SSL/result.log`.
```
[2024-01-18 16:22:54,443][__main__][INFO] - Epoch 91: loss=7.8436
[2024-01-18 16:23:26,013][__main__][INFO] - Epoch 92: loss=7.8519
[2024-01-18 16:23:44,365][__main__][INFO] - Epoch 93: loss=7.8575
[2024-01-18 16:24:10,332][__main__][INFO] - Epoch 94: loss=7.8537
[2024-01-18 16:24:32,502][__main__][INFO] - Epoch 95: loss=7.8565
[2024-01-18 16:24:54,124][__main__][INFO] - Epoch 96: loss=7.8184
[2024-01-18 16:25:16,829][__main__][INFO] - Epoch 97: loss=7.8639
[2024-01-18 16:25:41,448][__main__][INFO] - Epoch 98: loss=7.7560
[2024-01-18 16:26:07,286][__main__][INFO] - Epoch 99: loss=7.8561
[2024-01-18 16:26:27,925][__main__][INFO] - Epoch 100: loss=7.7540
[2024-01-18 16:26:46,483][__main__][INFO] - MAE: (tensor(6.0820, device='cuda:0'), tensor(5.6465, device='cuda:0'))
```



from pathlib import Path
import gc
import torch
import ultralytics
from ultralytics import YOLO

if __name__=='__main__':
    root=Path(__file__).resolve().parent
    print('PyTorch:',torch.__version__)
    print('Ultralytics:',ultralytics.__version__)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Configure a CUDA-enabled PyTorch environment.')
    print('GPU:',torch.cuda.get_device_name(0))
    # Training needs backward kernels too; inference-only success does not prove this works.
    layer=torch.nn.Conv2d(3,8,3,padding=1).cuda()
    x=torch.randn(2,3,64,64,device='cuda')
    layer(x).square().mean().backward()
    torch.cuda.synchronize()
    print('CUDA forward + backward: OK')
    del layer,x
    gc.collect(); torch.cuda.empty_cache()
    # Resolve dataset path locally, so moving the lab folder remains supported.
    yaml_path=root/'dataset/data.yaml'
    yaml_path.write_text(f'path: {root / "dataset"}\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 2\nnames:\n  0: mate\n  1: termo\n')
    existing=Path(__file__).resolve().parent/'weights/yolo26n.pt'
    model=YOLO(str(existing) if existing.is_file() else 'yolo26n.pt')
    model.train(data=str(yaml_path),epochs=60,patience=15,imgsz=640,batch=2,
                device=0,workers=0,cache=False,amp=True,seed=42,
                project=str(root/'runs'),name='mate-termo-v1',exist_ok=False)
    print('Training completed. Find best.pt inside the run folder shown above.')
    print('Do not use the test split for training or selecting hyperparameters.')

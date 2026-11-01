# Knee MRI Structure Detection (YOLO11n + Streamlit)

**Research prototype - not for medical use. Not clinically validated.**

A Streamlit app that detects four structures in 2D knee MRI slices:
ACL, PCL, Meniscus and Cartilage, using a YOLO11n model fine-tuned on fastMRI+ annotations.

## Links
- Live app: _add your Streamlit link here_
- Fine-tuned model: https://huggingface.co/hajirabaloch488/knee-mri-yolo11n
- Pretrained base model: https://huggingface.co/Ultralytics/YOLO11 (`yolo11n.pt`)

## Dataset
- Images: fastMRI knee single-coil MRI (https://fastmri.med.nyu.edu/), converted to 320x320 slices.
- Annotations: fastMRI+ (https://github.com/microsoft/fastmri-plus), `knee.csv`.
- 59 MRI files, 534 annotated slices, 953 boxes; split by MRI file (41 train / 9 val / 9 test).
- The MRI data is **not** included in this repository (data use terms).

## Results (test set, 9 unseen MRI files, 138 boxes)
| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| all | 0.288 | 0.312 | 0.212 | 0.064 |
| ACL | 0.286 | 0.357 | 0.203 | 0.043 |
| PCL | 0.461 | 0.435 | 0.393 | 0.144 |
| Meniscus | 0.240 | 0.224 | 0.166 | 0.047 |
| Cartilage | 0.165 | 0.231 | 0.085 | 0.024 |

The dataset is very small, so results are noisy and performance is low in absolute terms.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py` - Streamlit application (downloads the model from Hugging Face)
- `requirements.txt` - Python packages (CPU PyTorch)
- `packages.txt` - system packages needed by OpenCV on Streamlit Community Cloud

## License
Code and model follow the AGPL-3.0 license of Ultralytics YOLO.

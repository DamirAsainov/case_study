# Как обучить модели на удалённом сервере

Эта инструкция описывает полный прогон эксперимента на удалённой машине (Linux + NVIDIA GPU)
и возврат результатов обратно для сборки статьи.

## 1. Что скопировать на сервер

Нужны только код и зависимости. Данные и веса ResNet-18 скачаются сами.

```
case_study/
├── run_experiments.py
├── requirements.txt
└── src/
    ├── __init__.py
    ├── data.py
    ├── models.py
    ├── train.py
    ├── metrics.py
    ├── gradcam.py
    └── plots.py
```

С локальной машины (PowerShell / Git Bash):

```bash
# вариант 1: scp
scp -r run_experiments.py requirements.txt src user@SERVER:~/case_study/

# вариант 2: архив
tar -czf case_study.tgz run_experiments.py requirements.txt src
scp case_study.tgz user@SERVER:~/
ssh user@SERVER "mkdir -p ~/case_study && tar -xzf ~/case_study.tgz -C ~/case_study"
```

> Папки `data/`, `outputs/` и файл `outputs_train_log.txt` копировать **не нужно**.

## 2. Окружение на сервере

```bash
ssh user@SERVER
cd ~/case_study

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Сначала поставьте PyTorch под версию CUDA на сервере. Версию драйвера показывает `nvidia-smi`
в правом верхнем углу. Команду установки берите с <https://pytorch.org/get-started/locally/>, например:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Проверка, что GPU виден:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Должно вывести `True` и имя видеокарты. Если `False`, PyTorch установлен без CUDA или не совпадает
с драйвером. Переустановите его с правильным `--index-url`.

## 3. Быстрая проверка (2–5 минут)

Прогоните одну эпоху, чтобы убедиться, что всё скачивается и работает:

```bash
python -u run_experiments.py --epochs-cnn 1 --epochs-resnet 1 --num-workers 4 --out outputs_smoke
```

В конце должна появиться строка `Saved results to outputs_smoke/results.json ...`.
Потом папку можно удалить: `rm -rf outputs_smoke`.

## 4. Полный прогон

Запускайте внутри `tmux` (или `screen`), чтобы обучение не прервалось при разрыве SSH:

```bash
tmux new -s train
source .venv/bin/activate
cd ~/case_study
python -u run_experiments.py --num-workers 8 2>&1 | tee train_log.txt
```

- Отсоединиться от сессии и оставить обучение работать: `Ctrl+B`, затем `D`.
- Вернуться позже: `tmux attach -t train`.
- `-u` отключает буферизацию, поэтому прогресс по эпохам виден сразу.

Без tmux можно так:

```bash
nohup python -u run_experiments.py --num-workers 8 > train_log.txt 2>&1 &
tail -f train_log.txt
```

### Параметры

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--size` | `64` | Разрешение изображений MedMNIST: 28 / 64 / 128 / 224 |
| `--epochs-cnn` | `30` | Эпохи для Simple CNN |
| `--epochs-resnet` | `15` | Эпохи для ResNet-18 (transfer learning) |
| `--batch-size` | `64` | Размер батча |
| `--num-workers` | `0` | Процессы загрузки данных; на Linux ставьте 4–8 (≈ число ядер CPU) |
| `--seed` | `42` | Seed для воспроизводимости |
| `--out` | `outputs` | Папка с результатами |
| `--data-root` | `data` | Куда скачивать датасеты |

Для статьи используйте **параметры по умолчанию** (только с `--num-workers`), чтобы цифры совпадали
с описанием методологии. Если хотите сделать отдельный эксперимент с разрешением 128 px,
сохраните его в другую папку: `--size 128 --out outputs_128`.

### Что происходит во время прогона

Для каждого из двух датасетов (PneumoniaMNIST, затем DermaMNIST):
1. Скачивание данных в `data/`.
2. PCA + Logistic Regression (baseline), около минуты.
3. Simple CNN, 30 эпох.
4. ResNet-18 (ImageNet-pretrained), 15 эпох. При первом запуске скачиваются веса (~45 МБ).
5. Подсчёт метрик, графики, Grad-CAM.

Ожидаемое время: на современной GPU (T4 / RTX 3060 и выше) с `--num-workers 8`
примерно **15–30 минут** на весь прогон.

Примерно так выглядит строка лога для каждой эпохи:
```
  epoch 12/30  train_loss=0.2101  val_loss=0.1350  val_auc=0.9921  val_acc=0.9580
```

## 5. Результат

После завершения в `outputs/` будет:

```
outputs/
├── results.json                 # все метрики, история обучения, пороги
├── figures/                     # графики для статьи (PNG)
│   ├── class_distribution.png
│   ├── samples_*.png
│   ├── training_*.png
│   ├── roc_*.png
│   ├── confusion_*_*.png
│   ├── calibration_*.png
│   ├── threshold_pneumoniamnist.png
│   ├── gradcam_*.png
│   └── model_comparison.png
├── predictions/*.npz            # вероятности на тесте для каждой модели
└── *.pt                         # веса обученных моделей
```

### Дообучение отдельного датасета

Метрики сохраняются после каждого датасета в `outputs/results_<dataset>.json`. Если прогон прервался,
можно досчитать только недостающее. Остальные результаты в `outputs/` сохранятся, а `results.json`
и общие графики пересоберутся:

```bash
python -u run_experiments.py --num-workers 8 --datasets dermamnist 2>&1 | tee -a train_log.txt
```

## 6. Забрать результаты обратно

> **Скачивайте только после завершения.** Последняя строка `train_log.txt` должна начинаться
> с `Saved results (pneumoniamnist, dermamnist) to outputs/results.json`.
> Проверить можно так: `tail -n 2 train_log.txt` и `ls outputs/*.json`.

С локальной машины:

```bash
scp -r user@SERVER:~/case_study/outputs ./
scp user@SERVER:~/case_study/train_log.txt ./
```

Веса `*.pt` (≈ 45 МБ на ResNet) для статьи не нужны. Если не хотите их тянуть, на сервере:

```bash
tar -czf outputs.tgz --exclude='*.pt' outputs train_log.txt
```
и скачайте `outputs.tgz`.

Положите папку `outputs/` в корень `case_study/`. Потом напишите мне, и я соберу по ней статью
(`build_report.py` → `Case_Study_Report.docx`).

## 7. Частые проблемы

| Проблема | Решение |
|---|---|
| `CUDA out of memory` | Уменьшите `--batch-size 32` |
| `torch.cuda.is_available() == False` | Переустановите torch с нужным `--index-url` под версию CUDA |
| Зависает на загрузке данных / ошибки `DataLoader worker` | Уменьшите `--num-workers` (или `0`) |
| Нет доступа в интернет на сервере | Скачайте локально папку `data/` (файлы `*_64.npz`) и `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`, скопируйте их на сервер в те же пути |
| Обучение медленное, GPU загружена слабо (`nvidia-smi`) | Увеличьте `--num-workers`: узкое место в аугментации на CPU |

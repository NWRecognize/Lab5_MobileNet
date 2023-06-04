# Lab5_MobileNet
Обнаружение лиц на видео с помощью MobileNet  
  
Проверяем наличие доступной для работы видеокарты с поддержкой CUDA и вибираем её.  
Создается экземпляр модели обнаружения лиц MTCNN и устанавливается устройство для модели.  
Считываем входной видео файл:
```python
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
mtcnn = MTCNN(keep_all=True, device=device)
video = mmcv.VideoReader('31.mp4')
frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]

display.Video('31.mp4', width=640)
```
Считываем видеофайл, код обрабатывает каждый кадр, отображает видео и отслеживает лица, рисуя прямоугольники вокруг них. Результат сохраняется в списке frames_tracked.
```python
frames_tracked = []
for i, frame in enumerate(frames):
    print('\rTracking frame: {}'.format(i + 1), end='')
    boxes, _  = mtcnn.detect(frame)
    frame_draw = frame.copy()
    draw = ImageDraw.Draw(frame_draw)
    for box in boxes:
        draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
    frames_tracked.append(frame_draw.resize((640, 360), Image.BILINEAR))
print('\nDone')
```
Результат:
![image](https://github.com/NWRecognize/Lab5_MobileNet/assets/118212881/79af40fb-81a8-48f3-b27f-88e8202ac0f0)

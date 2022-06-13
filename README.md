mask_detection
==============================

Репозиторий для классификации масок (есть маска, нет маски)

**Данные**

Внешние данные были взяты с [kaggle](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)

**Описание пайплайна**

![Untitled Diagram drawio](https://user-images.githubusercontent.com/103132748/173438512-78498ee3-a48f-4bb0-8ec2-6b7c1614c527.png)

Пайплайн состоит из нескольких stages:
1. download_external_dataset - скачивает датасет с kaggle
2. make_dataset_external - парсит xml файлы для каждого изображения и собирает файл annotations.csv со следующими колонками (xmin,ymin,xmax,ymax,name,file,width,height)
3. create_crop_dataset_external - так как мы скачали датасет для детекции лиц, а решаем задачу классификации, данный stage вырезает все лица с изображений и складывает их по папкам в зависимости от класса
4. merge_dataset - мерджит исходный внутренний датасет с нарезанным внешним

Следующие stages выполняются для каждого типа датасета по отдельности (internal: только внутренние данные, которые изначально были, external: только те данные, которые скачали и нарезали с kaggle, both: смешанные internal и external датасеты)

6. train_test_split - делит данные для обучения на train, test и val подвыборки
7. train_model - обучает модель. Возможно обучение трех разные видов моделей - mobilenetv2, resnet50 и vit_base_patch16_224
8. vaidate_model - подсчет качества модели на валидационной подвыборке


**Управление жизненным циклом**

Для управления жизненным циклом моделей был выбран MLFLow

![image](https://user-images.githubusercontent.com/103132748/173439840-652850bb-5348-45f7-a9a0-f5e9c207848b.png)

MLFlow был развернут по сценарию 4

![image](https://user-images.githubusercontent.com/103132748/173440002-7a3bc0cb-962a-48b9-af36-4772574a6196.png)



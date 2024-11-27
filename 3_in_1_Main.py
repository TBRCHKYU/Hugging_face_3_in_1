from transformers import pipeline
from PIL import Image
import cv2
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests
import urllib
import numpy as np
import streamlit as st


# Заголовок приложения
st.title("Классификация изображений с помощью Hugging Face")

# Форма для загрузки изображения
uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])


classifier = pipeline("sentiment-analysis", "blanchefort/rubert-base-cased-sentiment")

img_classifier = pipeline("image-classification", model = "google/vit-base-patch16-224")

translator = pipeline("translation_en_to_ru", "Helsinki-NLP/opus-mt-en-ru")

if uploaded_file is not None:

  st.image(uploaded_file, caption="Загруженное изображение")

  onpicture = ""
  items = ""

  def translate(Eng_word):
    translated = str(translator(Eng_word))
    k = 0
    answer = ""
    for i in translated:
      if k == 3:
        answer += i
      if i == "'":
        k += 1
    return answer[:-1]

  def classify(z):
    z = str(classifier(z))
    result = ""
    number = ""
    answer = ""
    k = 0
    for i in z:
      if k == 3:
        result += i.lower()
      elif k == 6:
        number += i
      if i == "'":
        k += 1
    result = result[:-1]
    number = number[2:]
    z = str(translator(result))
    k = 0
    for i in z:
      if k == 3:
        answer += i
      if i == "'":
        k += 1
    answer = "негативный" if "отрицательный" in answer else answer[:-1]
    st.write("Общий эмоциональный фон изображения", answer, "с точностью", number[:-2])

  def draw_object_bounding_box(image_to_process, box, item):
    x, y, w, h = box
    start = (int(x), int(y))
    end = (int(x + w), int(y + h))
    color = (0, 255, 0)
    width = 2
    final_image = cv2.rectangle(image_to_process, start, end, color, width)

    start = (int(x), int(y - 10))
    font_size = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = 2
    final_image = cv2.putText(final_image, item, start, font,
                              font_size, color, width, cv2.LINE_AA)

    return final_image

  image = Image.open(uploaded_file)

  processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
  model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

  inputs = processor(images=image, return_tensors="pt")
  outputs = model(**inputs)

  # конвертируем выходные данные (ограничивающие рамки и логиты классов)
  # оставим только обнаружения со счетом > 0,9

  st.write("**Результаты классификации:**")

  target_sizes = torch.tensor([image.size[::-1]])
  results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

  for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    onpicture += model.config.id2label[label.item()]
    onpicture += " "
    translated = translator(model.config.id2label[label.item()])
    st.write(
      f"На изображении присутствует {translate(model.config.id2label[label.item()])} с уверенностью в  "
      f"{round(score.item(), 3)} в координатах {box} \n"
    )
  classify(onpicture)


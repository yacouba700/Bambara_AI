from transformers import pipeline

translator = pipeline(
    "translation",
    model="./bambara-model"
)

print(translator("bonjour"))

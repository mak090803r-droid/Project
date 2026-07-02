import time
from transformers import MarianMTModel, MarianTokenizer

# --- Configuration ---
model_name = "Helsinki-NLP/opus-mt-fr-en"  # French -> English

# --- Sample French texts ---
french_texts = [
    "L1 项目名称：本项目名称为“用于实时外语阅读/翻译的智能眼镜”。1.2 项目提案：本项目旨在设计并开发一款智能眼镜，该眼镜能够捕捉外语的文本和数字文本，并实时将其翻译成用户选择的目标语言。该系统将集成一个安装在可穿戴眼镜上的轻型摄像头，用于采集书籍、说明书、标签和技术文档中的文本信息。采集到的文本将首先通过光学字符识别 (OCR) 技术转换为机器可读格式，然后由基于人工智能的翻译算法生成翻译后的内容。翻译后的输出将通过内置扬声器或耳机以音频形式播放，从而实现免提实时阅读和理解。我们将开发一个功能原型，并在实验室条件下进行演示，以验证实时文本采集、翻译准确性和音频输出性能。1.3 项目范围：本项目范围包括设计、实现和测试用于实时外语阅读和翻译的概念验证型智能眼镜系统。该项目将涉及将摄像头模块集成到可穿戴式眼镜框架中，并开发支持多种语言的OCR（光学字符识别）功能。此外，还将集成一个音频输出子系统以提供音频输出。"
]

# --- Load model and tokenizer (timed) ---
print(f"Loading MarianMT model: {model_name}")
load_start = time.time()
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
load_time = time.time() - load_start
print(f"Model loaded in {load_time:.2f}s\n")

# --- Translate each sentence individually (timed) ---
print("=" * 70)
print("INDIVIDUAL TRANSLATIONS")
print("=" * 70)

total_individual_time = 0.0

for i, text in enumerate(french_texts, 1):
    start = time.time()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    result = tokenizer.decode(translated[0], skip_special_tokens=True)
    elapsed = time.time() - start
    total_individual_time += elapsed

    print(f"\n[{i}] French:  {text}")
    print(f"    English: {result}")
    print(f"    Time:    {elapsed:.4f}s")

print(f"\nTotal individual translation time: {total_individual_time:.4f}s")

# --- Batch translate all sentences at once (timed) ---
print("\n" + "=" * 70)
print("BATCH TRANSLATION")
print("=" * 70)

batch_start = time.time()
batch_inputs = tokenizer(french_texts, return_tensors="pt", padding=True, truncation=True)
batch_translated = model.generate(**batch_inputs)
batch_results = [tokenizer.decode(t, skip_special_tokens=True) for t in batch_translated]
batch_time = time.time() - batch_start

for i, (fr, en) in enumerate(zip(french_texts, batch_results), 1):
    print(f"    English: {en}")

print(f"\nBatch translation time: {batch_time:.4f}s")

# --- Summary ---
print("\n" + "=" * 70)
print("TIMING SUMMARY")
print("=" * 70)
print(f"  Model load time:           {load_time:.2f}s")
print(f"  Individual total time:     {total_individual_time:.4f}s")
print(f"  Batch translation time:    {batch_time:.4f}s")
print(f"  Speedup (batch vs indiv):  {total_individual_time / batch_time:.2f}x")
print("=" * 70)

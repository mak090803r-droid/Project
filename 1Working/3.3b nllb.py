import os
import re
import time
import torch
import ctranslate2
import transformers
from huggingface_hub import snapshot_download

# ══════════════════════════════════════════════
# 1. HARDWARE ACCELERATION DETECTION
# ══════════════════════════════════════════════
if torch.cuda.is_available():
    DEVICE = "cuda"
    COMPUTE_TYPE = "float16" # Harness lightning-fast Tensor Cores on your GPU
    print("[INFO] Hardware Target: NVIDIA GPU Active (CUDA Processing Enabled)")
else:
    DEVICE = "cpu"
    COMPUTE_TYPE = "int8"    # Fallback to speed-optimized integer quantization on CPU
    print("[INFO] Hardware Target: CPU ONLY (Fallback Execution Enabled)")

# The community pre-compiled repository containing the absolute 3.3B flagship layers
CT2_PRECOMPILED_MODEL = "michaelfeil/ct2fast-nllb-200-3.3B"

# FLORES-200 Language target vector designators
LANG_MAP = {
    "zh": "zho_Hans",  # Chinese (Simplified)
    "fr": "fra_Latn",  # French
    "en": "eng_Latn"   # English
}

# ══════════════════════════════════════════════
# 2. DOWNLOAD & INITIALIZE PRE-COMPILED 3.3B ENGINE
# ══════════════════════════════════════════════
# ══════════════════════════════════════════════
# 2. DOWNLOAD & INITIALIZE PRE-COMPILED 3.3B ENGINE
# ══════════════════════════════════════════════
print(f"\n[INFO] Loading Pre-Compiled 3.3B Transformer files from Local Storage...")
start_load = time.time()

try:
    # 🌟 local_files_only=True forces the script to skip the internet handshake completely!
    model_dir = snapshot_download(repo_id=CT2_PRECOMPILED_MODEL, local_files_only=True)
except Exception as e:
    # If you ever move to a brand new computer, this catches the missing files error and downloads them once
    print("[INFO] Model not found locally. Connecting to Hub for a one-time setup download...")
    model_dir = snapshot_download(repo_id=CT2_PRECOMPILED_MODEL, local_files_only=False)

print("[INFO] Initializing optimized 3.3B layers into native CTranslate2 execution space...")
TRANSLATOR = ctranslate2.Translator(model_dir, device=DEVICE, compute_type=COMPUTE_TYPE)
TOKENIZER = transformers.AutoTokenizer.from_pretrained(model_dir)

print(f"[INFO] Flagship 3.3B Model Loaded Successfully in {time.time()-start_load:.2f}s")
# 3. PARALLEL BATCH TRANSLATION CORE PIPELINE
# ══════════════════════════════════════════════
def translate_paragraph(paragraph_text, src_lang_code, tgt_lang_code="en"):
    if not paragraph_text.strip():
        return ""
        
    src_token = LANG_MAP[src_lang_code]
    tgt_token = LANG_MAP[tgt_lang_code]
    
    # Update the global tokenizer's source language state tracking prefix
    TOKENIZER.src_lang = src_token
    
    # --- STEP A: SENTENCE SEGMENTATION ---
    if src_lang_code == "zh":
        raw_chunks = re.split(r'(。|？|！)', paragraph_text)
        chunks = []
        for i in range(0, len(raw_chunks)-1, 2):
            chunks.append(raw_chunks[i] + raw_chunks[i+1])
        if len(raw_chunks) % 2 != 0 and raw_chunks[-1].strip():
            chunks.append(raw_chunks[-1])
    else:
        # Splits cleanly across sentences while keeping numeric structures intact (like 1.2, 1.3)
        chunks = re.split(r'(?<=[.!?])\s+', paragraph_text)
        
    chunks = [c.strip() for c in chunks if c.strip()]
    
    # --- STEP B: BATCH TOKENIZATION ---
    tokenized_batch = []
    for chunk in chunks:
        tokens = TOKENIZER.convert_ids_to_tokens(TOKENIZER.encode(chunk))
        tokenized_batch.append(tokens)
        
    if not tokenized_batch:
        return ""

    # --- STEP C: PARALLEL GPU INFERENCE EXECUTOR ---
    # Construct target token constraints matching the exact sizing of the split lines array
    target_prefixes = [[tgt_token]] * len(tokenized_batch)
    
    # Fire off entire batch matrix onto processing hardware inside a single operation
    results = TRANSLATOR.translate_batch(tokenized_batch, target_prefix=target_prefixes)
    
    # --- STEP D: STREAM DECODING & PARAGRAPH RECONSTRUCTION ---
    translated_sentences = []
    for res in results:
        output_tokens = res.hypotheses[0]  # Grab top generation path hypothesis
        raw_decoded = TOKENIZER.decode(TOKENIZER.convert_tokens_to_ids(output_tokens))
        clean_sentence = raw_decoded.replace(tgt_token, "").strip()
        translated_sentences.append(clean_sentence)
        
    return " ".join(translated_sentences)


# ══════════════════════════════════════════════
# 4. USER TESTING PLAYGROUND SECTION
# ══════════════════════════════════════════════
if __name__ == "__main__":
    
    # Test Payload 1: French Raw Context Input
    my_french_paragraph = (
       
"Titre du projet L1 : Ce projet s'intitule « Lunettes intelligentes pour la lecture et la traduction en temps réel de langues étrangères ». 1.2 Description du projet : Ce projet propose la conception et le développement de lunettes intelligentes capables de capturer du texte imprimé et numérique en langues étrangères et de le traduire en temps réel dans la langue choisie par l'utilisateur. Le système intégrera une caméra légère fixée sur les lunettes pour acquérir des informations textuelles provenant de livres, de manuels d'utilisation, d'étiquettes et de documents techniques. Le texte capturé sera traité par reconnaissance optique de caractères (OCR) pour le convertir en un format lisible par machine, puis par des algorithmes de traduction basés sur l'intelligence artificielle pour générer le contenu traduit. La traduction sera ensuite diffusée oralement via des haut-parleurs ou des écouteurs intégrés, permettant une lecture et une compréhension mains libres en temps réel. Un prototype fonctionnel sera développé et testé en laboratoire afin de valider la capture de texte en temps réel, la précision de la traduction et la qualité de la sortie audio. 1.3 Portée du projet : Ce projet comprend la conception, la mise en œuvre et les tests d'un système de lunettes intelligentes de démonstration de faisabilité pour la lecture et la traduction en temps réel de langues étrangères. Le projet prévoit l'intégration d'un module caméra à un dispositif portable, le développement d'un système de reconnaissance optique de caractères (OCR) multilingue et l'intégration d'un sous-système de sortie audio."


    )

    # Test Payload 2: Chinese Raw Context Input
    my_chinese_paragraph = (
        
"L1 项目名称：本项目名称为“用于实时外语阅读/翻译的智能眼镜”。1.2 项目提案：本项目旨在设计并开发一款智能眼镜，该眼镜能够捕捉外语的文本和数字文本，并实时将其翻译成用户选择的目标语言。该系统将集成一个安装在可穿戴眼镜上的轻型摄像头，用于采集书籍、说明书、标签和技术文档中的文本信息。采集到的文本将首先通过光学字符识别 (OCR) 技术转换为机器可读格式，然后由基于人工智能的翻译算法生成翻译后的内容。翻译后的输出将通过内置扬声器或耳机以音频形式播放，从而实现免提实时阅读和理解。我们将开发一个功能原型，并在实验室条件下进行演示，以验证实时文本采集、翻译准确性和音频输出性能。1.3 项目范围：本项目范围包括设计、实现和测试用于实时外语阅读和翻译的概念验证型智能眼镜系统。该项目将涉及将摄像头模块集成到可穿戴式眼镜框架中，并开发支持多种语言的OCR（光学字符识别）功能。此外，还将集成一个音频输出子系统以提供音频输出。"

    )

    # Execute French to English Test Vector
    print("\n" + "="*70 + "\n🔥 RUNNING INFERENCE: FRENCH TO ENGLISH (3.3B BATCH LAYER)\n" + "="*70)
    t_start = time.time()
    french_output = translate_paragraph(my_french_paragraph, src_lang_code="fr")
    print(f"{french_output}\n" + "-"*70)
    print(f"⏱️ Paragraph Total Compute Latency: {(time.time() - t_start)*1000:.2f} ms\n" + "="*70)

    # Execute Chinese to English Test Vector
    print("\n" + "="*70 + "\n🔥 RUNNING INFERENCE: CHINESE TO ENGLISH (3.3B BATCH LAYER)\n" + "="*70)
    t_start = time.time()
    chinese_output = translate_paragraph(my_chinese_paragraph, src_lang_code="zh")
    print(f"{chinese_output}\n" + "-"*70)
    print(f"⏱️ Paragraph Total Compute Latency: {(time.time() - t_start)*1000:.2f} ms\n" + "="*70)
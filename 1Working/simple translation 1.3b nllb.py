import os
import re
import time
import torch
import ctranslate2
import transformers

# ══════════════════════════════════════════════
# 1. HARDWARE ACCELERATION DETECTION
# ══════════════════════════════════════════════
if torch.cuda.is_available():
    DEVICE = "cuda"
    COMPUTE_TYPE = "float16" # Uses lightning-fast tensor math cores on GPU VRAM
    os.environ["ARGOS_DEVICE_TYPE"] = "cuda" # Cover all backend bases
    print("[INFO] Hardware Target: NVIDIA GPU (CUDA Optimization Active)")
else:
    DEVICE = "cpu"
    COMPUTE_TYPE = "int8"    # Fallback to high-speed quantized integer math on CPU
    print("[INFO] Hardware Target: CPU ONLY (Fallback Int8 Execution Active)")

# Model mapping tracking variables
MODEL_DIR = "nllb-200-1.3B-ct2"
BASE_MODEL = "facebook/nllb-200-distilled-1.3B"

# FLORES-200 target language vector designators
LANG_MAP = {
    "zh": "zho_Hans",  # Chinese (Simplified)
    "fr": "fra_Latn",  # French
    "en": "eng_Latn"   # English
}

# ══════════════════════════════════════════════
# 2. AUTOMATIC LOCAL TRANSFORMATION ENGINE
# ══════════════════════════════════════════════
if not os.path.exists(MODEL_DIR):
    print(f"\n[INFO] Compiling raw {BASE_MODEL} matrix into an optimized CTranslate2 binary structure...")
    start_conv = time.time()
    
    conversion_cmd = (
        f"ct2-transformers-converter --model {BASE_MODEL} "
        f"--output_dir {MODEL_DIR} --force --quantization {COMPUTE_TYPE}"
    )
    exit_code = os.system(conversion_cmd)
    if exit_code != 0:
        raise RuntimeError("[ERROR] Model compilation wrapper failed. Verify system dependencies.")
    print(f"[INFO] Serialization complete! Saved to '{MODEL_DIR}' ({time.time()-start_conv:.2f}s)")

# Initialize translation engine instances once into global memory space
print("[INFO] Initializing Transformer attention layer matrices...")
TRANSLATOR = ctranslate2.Translator(MODEL_DIR, device=DEVICE, compute_type=COMPUTE_TYPE)

# ══════════════════════════════════════════════
# 3. PARALLEL BATCH TRANSLATION CORE PIPELINE
# ══════════════════════════════════════════════
def translate_paragraph(paragraph_text, src_lang_code, tgt_lang_code="en"):
    """
    Splits incoming text to prevent token omission, translates sentences 
    in parallel via GPU batches, and returns a reconstructed paragraph.
    """
    if not paragraph_text.strip():
        return ""
        
    src_token = LANG_MAP[src_lang_code]
    tgt_token = LANG_MAP[tgt_lang_code]
    
    # --- STEP A: SENTENCE SEGMENTATION ---
    if src_lang_code == "zh":
        # Split securely on Chinese native punctuation marks
        raw_chunks = re.split(r'(。|？|！)', paragraph_text)
        chunks = []
        for i in range(0, len(raw_chunks)-1, 2):
            chunks.append(raw_chunks[i] + raw_chunks[i+1])
        if len(raw_chunks) % 2 != 0 and raw_chunks[-1].strip():
            chunks.append(raw_chunks[-1])
    else:
        # Split Latin scripts by punctuation boundaries followed by whitespace
        chunks = re.split(r'(?<=[.!?])\s+', paragraph_text)
        
    chunks = [c.strip() for c in chunks if c.strip()]
    
    # --- STEP B: BATCH TOKENIZATION ---
    tokenizer = transformers.AutoTokenizer.from_pretrained(BASE_MODEL, src_lang=src_token)
    tokenized_batch = []
    for chunk in chunks:
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(chunk))
        tokenized_batch.append(tokens)
        
    if not tokenized_batch:
        return ""

    # --- STEP C: PARALLEL INFERENCE EXECUTOR ---
    # Create matching prefix target tokens matching the absolute batch length
    target_prefixes = [[tgt_token]] * len(tokenized_batch)
    
    # Dispatch entire list block directly onto hardware acceleration lanes concurrently
    results = TRANSLATOR.translate_batch(tokenized_batch, target_prefix=target_prefixes)
    
    # --- STEP D: STREAM DECODING & PARAGRAPH ASSEMBLY ---
    translated_sentences = []
    for res in results:
        output_tokens = res.hypotheses[0]  # Grab top predictive target hypothesis line
        raw_decoded = tokenizer.decode(tokenizer.convert_tokens_to_ids(output_tokens))
        clean_sentence = raw_decoded.replace(tgt_token, "").strip()
        translated_sentences.append(clean_sentence)
        
    # Reassemble individual translated nodes back into a unified block layout
    return " ".join(translated_sentences)


# ══════════════════════════════════════════════
# 4. USER TESTING PLAYGROUND SECTION
# ══════════════════════════════════════════════
if __name__ == "__main__":
    
    # 👇 DROP YOUR RAW FRENCH TEXT TEST HERE 👇
    my_french_paragraph = (
        "Titre du projet L1 : Lunettes intelligentes pour la lecture et la traduction en "
        "temps réel de langues étrangères. 1.2 Présentation du projet : Ce projet propose "
        "la conception et le développement de lunettes intelligentes capables de capturer du "
        "texte imprimé et numérique en langues étrangères et de le traduire en temps réel "
        "dans la langue choisie par l'utilisateur. Le système intégrera une caméra légère "
        "montée sur des lunettes pour acquérir des informations textuelles provenant de livres, "
        "de manuels d'utilisation, d'étiquettes et de documents techniques. Le texte capturé "
        "sera traité par reconnaissance optique de caractères (OCR) pour le convertir en un "
        "format lisible par machine, puis par des algorithmes de traduction basés sur "
        "l'intelligence artificielle pour générer le contenu traduit. La traduction sera "
        "ensuite diffusée oralement via des haut-parleurs intégrés ou un téléphone portable, "
        "permettant une lecture et une compréhension mains libres en temps réel. Un prototype "
        "fonctionnel sera développé et testé en laboratoire afin de valider la capture de "
        "texte en temps réel, la précision de la traduction et les performances de la sortie "
        "audio. 1.3 Portée du projet : Ce projet comprend la conception, la mise en œuvre et "
        "les tests d'un prototype de lunettes intelligentes pour la lecture et la traduction "
        "en temps réel de langues étrangères. Le projet prévoit l'intégration d'un module "
        "caméra à un cadre de montre portable, le développement d'un système de communication "
        "ouvert multilingue et l'intégration d'un sous-système de sortie audio."
    )

    # 👇 DROP YOUR RAW CHINESE TEXT TEST HERE 👇
    my_chinese_paragraph = (
        "L1 项目名称：本项目名称为“用于实时外语阅读/翻译的智能眼镜”。1.2 项目陈述：本项目旨在设计"
        "并开发一款智能眼镜，该眼镜能够捕捉外语的文本（包括手写文本和数字文本），并实时将其翻译成用户"
        "选择的目标语言。该系统将集成一个安装在可穿戴眼镜上的轻型摄像头，用于采集书籍、说明书、标签和技术"
        "文档中的文本信息。采集到的文本将首先通过光学字符识别（OCR）技术转换为机器可读格式，然后利用人工"
        "智能（AI）翻译算法生成翻译后的内容。翻译后的输出将通过内置扬声器或车载电话以语音形式播放，从而实现"
        "免提实时阅读和理解。我们将开发一个功能原型，并在实验室条件下进行演示，以验证实时文本采集、翻译"
        "准确性和音频输出性能。 1.3 项目范围 本项目范围包括设计、实现和测试用于实时外语阅读和翻译的概念"
        "验证型智能眼镜系统。项目将涉及将摄像头模块集成到可穿戴眼镜框架中，并开发支持多种语言的操作系统。"
        "此外，还将集成一个音频输出子系统以提供音频输出功能。"
    )

    # ──────────────────────────────────────────
    # RUNNING EXPERIMENT 1: FRENCH TO ENGLISH
    # ──────────────────────────────────────────
    print("\n" + "="*70)
    print("                 BENCHMARK RUN: FRENCH -> ENGLISH")
    print("="*70)
    
    t_start = time.time()
    french_output = translate_paragraph(my_french_paragraph, src_lang_code="fr")
    t_end = time.time()
    
    print(french_output)
    print("-" * 70)
    print(f"⏱️ Paragraph Compute Time: {(t_end - t_start)*1000:.2f} ms")
    print("="*70)

    # ──────────────────────────────────────────
    # RUNNING EXPERIMENT 2: CHINESE TO ENGLISH
    # ──────────────────────────────────────────
    print("\n" + "="*70)
    print("                BENCHMARK RUN: CHINESE -> ENGLISH")
    print("="*70)
    
    t_start = time.time()
    chinese_output = translate_paragraph(my_chinese_paragraph, src_lang_code="zh")
    t_end = time.time()
    
    print(chinese_output)
    print("-" * 70)
    print(f"⏱️ Paragraph Compute Time: {(t_end - t_start)*1000:.2f} ms")
    print("="*70)
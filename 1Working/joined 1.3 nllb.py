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
        "heee y  nen Ca To coe de n Ce Ce De Cres Cee es eehee ☆ D 附件： 吐鲁番市集中教育培训学校学员 子女问答策略 一、我的家人在什么地方” 第他们各质购谈主的婚训学校供 参加系统性的培训 学习耕育他们在账要的学习生通样境都很时，你不用报 心，他们学习期间的学命免费吃作电费，并且标准比较高。 每天的快食會合1元以上要至超过了部分学具在家里的 生活标准，每天自我们的于部陪着绝们一同学习，提供铺 导 帮助，与做们吃同特的很果，待同样的密会，所以你完全不 用很心他们的话，如果你想见一见他们的话，我们可以安 操你和他们进行便频会面. 二、为什么我的家人要去参加学习？ 答：让你家人去学习因为他 们不同程度的受到了宗教极 端和暴力恐怖思想的侵害影响，如果一旦受到“三股势力”、 别有用心的人的煽动，提爱，鑫感，后果是很严重的。如果 他们因为极端思想和“三股势力”的影响，做了不该做的事 情，不 仅会伤害到无辜群众，而且会伤害到他们自己、其他 家人、亲戚朋友，甚至包括你，我想这些绝不是你想看到的。 所以，为了大家的安全、为了你的家庭幸福、为了你能安心 学习，必须要让他们第一时问到学校接参加集中教育学习。 -6-"
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
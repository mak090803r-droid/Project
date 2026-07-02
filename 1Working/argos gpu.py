import os
os.environ["ARGOS_DEVICE_TYPE"] = "cuda"  # 🌟 Forces Argos to use your GPU

import argostranslate.package
import argostranslate.translate
import time

start=time.time()

# ══════════════════════════════════════════════
# CONFIG — change these to your language pair
# ══════════════════════════════════════════════
FROM_LANG = "fr"   # source language code
TO_LANG   = "en"   # target language code

# Language codes reference:
# en = English    fr = French     ar = Arabic
# ur = Urdu       de = German     zh = Chinese
# es = Spanish    it = Italian    ru = Russian
# ja = Japanese   ko = Korean     tr = Turkish

def install_language_package(from_code, to_code):
    """Downloads and installs the translation package if not already installed."""
    print(f"[INFO] Checking language package: {from_code} → {to_code}")

    # Update package index
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()

    # Find the right package
    package = next(
        filter(
            lambda p: p.from_code == from_code and p.to_code == to_code,
            available_packages
        ),
        None
    )

    if package is None:
        print(f"[ERROR] No package found for {from_code} → {to_code}")
        print("[INFO] Available packages:")
        for p in available_packages:
            print(f"       {p.from_code} → {p.to_code}")
        return False

    # Check if already installed
    installed = argostranslate.package.get_installed_packages()
    already_installed = any(
        p.from_code == from_code and p.to_code == to_code
        for p in installed
    )

    if already_installed:
        print(f"[INFO] Package {from_code} → {to_code} already installed ✓")
        return True

    # Download and install
    print(f"[INFO] Downloading {from_code} → {to_code} package...")
    argostranslate.package.install_from_path(package.download())
    print(f"[INFO] Package installed ✓")
    return True

def translate(text, from_code=FROM_LANG, to_code=TO_LANG):
    """Translates text from source to target language."""
    if not text or not text.strip():
        return ""

    translated = argostranslate.translate.translate(text, from_code, to_code)
    return translated

# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
if __name__ == "__main__":

    # Step 1: Install package (only downloads once, cached after)
    success = install_language_package(FROM_LANG, TO_LANG)

    if success:
        # Step 2: Test translation
        test_texts = [
            "Titre du projet L1 : Ce projet s'intitule « Lunettes intelligentes pour la lecture et la traduction en temps réel de langues étrangères ». 1.2 Description du projet : Ce projet propose la conception et le développement de lunettes intelligentes capables de capturer du texte imprimé et numérique en langues étrangères et de le traduire en temps réel dans la langue choisie par l'utilisateur. Le système intégrera une caméra légère fixée sur les lunettes pour acquérir des informations textuelles provenant de livres, de manuels d'utilisation, d'étiquettes et de documents techniques. Le texte capturé sera traité par reconnaissance optique de caractères (OCR) pour le convertir en un format lisible par machine, puis par des algorithmes de traduction basés sur l'intelligence artificielle pour générer le contenu traduit. La traduction sera ensuite diffusée oralement via des haut-parleurs ou des écouteurs intégrés, permettant une lecture et une compréhension mains libres en temps réel. Un prototype fonctionnel sera développé et testé en laboratoire afin de valider la capture de texte en temps réel, la précision de la traduction et la qualité de la sortie audio. 1.3 Portée du projet : Ce projet comprend la conception, la mise en œuvre et les tests d'un système de lunettes intelligentes de démonstration de faisabilité pour la lecture et la traduction en temps réel de langues étrangères. Le projet prévoit l'intégration d'un module caméra à un dispositif portable, le développement d'un système de reconnaissance optique de caractères (OCR) multilingue et l'intégration d'un sous-système de sortie audio."
        ]

        print("\n" + "="*60)
        print(f"  TRANSLATION TEST  ({FROM_LANG.upper()} → {TO_LANG.upper()})")
        print("="*60)

        for text in test_texts:
            result = translate(text)
            
            print(f"  Translated : {result}")

        print("\n" + "="*60)


end = time.time()
print("Execution time:", end - start, "seconds")
import json
import re
from pathlib import Path


class MangaTranslator:
    """
    Translates Japanese text in imagedata.json to English.

    Supported backends:
        - "argos"   : Argos Translate (fully local, free, ideal for Apple Silicon)
                      pip install argostranslate
        - "google"  : Google Cloud Translate v2 (requires billing + service account)
                      pip install google-cloud-translate
        - "deep"    : deep-translator using Google Translate (free, no API key)
                      pip install deep-translator
    """

    SUPPORTED_BACKENDS = ("argos", "google", "deep")

    def __init__(self, json_path: str, backend: str = "argos", credentials_path: str = None):
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(f"Unknown backend '{backend}'. Choose from: {self.SUPPORTED_BACKENDS}")

        self.json_path = Path(json_path)
        self.backend = backend
        self._translator = self._init_backend(backend, credentials_path)

        with open(self.json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def _init_backend(self, backend: str, credentials_path: str | None):
        if backend == "argos":
            return self._init_argos()
        elif backend == "google":
            return self._init_google(credentials_path)
        elif backend == "deep":
            return self._init_deep()

    def _init_argos(self):
        try:
            import argostranslate.package as pkg
            import argostranslate.translate as argt
        except ImportError:
            raise ImportError("Run: pip install argostranslate")

        installed = {f"{p.from_code}-{p.to_code}" for p in argt.get_installed_languages()
                     if hasattr(p, "from_code")}

        if "ja-en" not in installed:
            print("Downloading Argos ja→en language package...")
            pkg.update_package_index()
            available = pkg.get_available_packages()
            ja_en = next((p for p in available if p.from_code == "ja" and p.to_code == "en"), None)
            if ja_en is None:
                raise RuntimeError("ja→en Argos package not found.")
            pkg.install_from_path(ja_en.download())

        def _translate(text: str) -> str:
            langs = argt.get_installed_languages()
            ja = next((l for l in langs if l.code == "ja"), None)
            en = next((l for l in langs if l.code == "en"), None)
            return ja.get_translation(en).translate(text)

        return _translate

    def _init_google(self, credentials_path: str | None):
        from google.cloud import translate_v2 as translate
        client = (translate.Client.from_service_account_json(credentials_path)
                  if credentials_path else translate.Client())
        return lambda text: client.translate(text, target_language="en", source_language="ja")["translatedText"]

    def _init_deep(self):
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source="ja", target="en")
        return lambda text: translator.translate(text)

    def _clean_text(self, text: str) -> str:
        """Removes HTML tags and Japanese-style brackets."""
        # Remove anything between < > (tags like <b>, <br/>, etc.)
        # Also handles cases where OCR might use full-width ＜ ＞ brackets
        clean = re.sub(r'<[^>]+>', '', text)
        clean = re.sub(r'＜[^＞]+＞', '', clean)
        return clean.strip()

    def translate_bubbles(self, overwrite: bool = False) -> int:
        translated_count = 0
        pages = self.data if isinstance(self.data, list) else [self.data]

        for page in pages:
            bubbles = page.get("bubbles", [])
            for bubble in bubbles:
                # If a previous run left tags in translated_text,
                # overwrite=False will skip it.
                if not overwrite and bubble.get("translated_text"):
                    continue

                original_raw = bubble.get("original_text", "")
                original_cleaned = self._clean_text(original_raw)

                if not original_cleaned:
                    continue

                try:
                    bubble["translated_text"] = self._translator(original_cleaned)
                    translated_count += 1
                    print(f"[{bubble.get('id', '?')}] {original_cleaned!r} → {bubble['translated_text']!r}")
                except Exception as exc:
                    print(f"[{bubble.get('id', '?')}] Translation failed: {exc}")

        return translated_count

    def save_results(self) -> None:
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        print(f"Saved → {self.json_path}")

    def run(self, overwrite: bool = False) -> None:
        count = self.translate_bubbles(overwrite=overwrite)
        self.save_results()
        print(f"Done — {count} bubble(s) translated.")


if __name__ == "__main__":
    JSON_FILE = "imagedata.json"
    if Path(JSON_FILE).exists():
        translator = MangaTranslator(JSON_FILE, backend="argos")
        # Use overwrite=True if you want to re-translate bubbles
        # that already have existing (possibly broken) translations.
        translator.run(overwrite=True)
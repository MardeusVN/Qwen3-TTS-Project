from pathlib import Path
p = Path('data/processed/metadata_clean.jsonl')
text = p.read_text(encoding='utf-8')
text = text.replace('"audio": "data/', '"audio": "wavs_24k/')
text = text.replace('"ref_audio": "data/', '"ref_audio": "wavs_24k/')
p.write_text(text, encoding='utf-8')
print('updated')

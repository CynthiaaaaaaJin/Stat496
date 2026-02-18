import json
from pathlib import Path

ALLOWED = set("ABCDE")

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    in_path = root / "data" / "answers_clean.json"
    out_path = root / "data" / "answers_letters_only.json"

    answers_map = json.loads(in_path.read_text(encoding="utf-8"))

    cleaned = {}
    for k, v in answers_map.items():
        key = str(k).strip().lstrip("Qq")  # "Q6" -> "6" (if ever happens)

        # v should be a list like ["B"] or ["A","C"] etc
        letters = []
        for x in (v or []):
            s = str(x).strip().upper()
            if s in ALLOWED:
                letters.append(s)

        cleaned[key] = letters

    # sort keys numerically for pretty output
    cleaned_sorted = {k: cleaned[k] for k in sorted(cleaned, key=lambda x: int(x))}

    out_path.write_text(
        json.dumps(cleaned_sorted, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # also print to terminal in the same JSON format
    print(json.dumps(cleaned_sorted, ensure_ascii=False, indent=2))
    print(f"\n✅ wrote: {out_path}")

if __name__ == "__main__":
    main()

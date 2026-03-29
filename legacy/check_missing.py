import json

with open('manual_formulas.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

missing = []
for a in range(4):
    for b in range(4):
        for c in range(4):
            k = f"{a},{b},{c}"
            if k not in data or len(data[k]) == 0:
                missing.append(k)

print("Missing or empty classes:")
for m in missing:
    print(m)

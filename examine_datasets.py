import json
import sys
sys.stdout.reconfigure(encoding="utf-8")

# Gita Guidance QA
print('=== Gita Guidance QA (711 pairs) ===')
with open('data/evaluation/external/gita_guidance_qa.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 3: break
        d = json.loads(line)
        msgs = d.get('messages', [])
        print(f'Keys: {list(d.keys())}')
        print(f'Num messages: {len(msgs)}')
        for m in msgs:
            role = m.get('role', '')
            content = str(m.get('content', ''))[:150]
            print(f'  [{role}]: {content}')
        print()

# ISKCON VedaBase
print('=== ISKCON VedaBase (657 entries) ===')
with open('data/evaluation/external/iskcon_vedabase.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 2: break
        d = json.loads(line)
        print(f'Keys: {list(d.keys())}')
        for k, v in d.items():
            if k in ('minilm_embedding', 'bge_embedding'):
                print(f'  {k}: [{len(v)} floats]')
            else:
                print(f'  {k}: {str(v)[:120]}')
        print()

# Edwin Arnold - check for philosophy-related Qs
print('=== Edwin Arnold - Philosophy Questions ===')
with open('data/evaluation/external/edwin_arnold_qa.jsonl', 'r', encoding='utf-8') as f:
    questions = []
    for line in f:
        d = json.loads(line)
        q = d.get('question', '')
        a = d.get('answer', '')
        # Filter for philosophy-related questions
        if any(w in q.lower() for w in ['karma', 'dharma', 'soul', 'self', 'action', 'desire', 'yoga', 'krishna', 'arjuna', 'teach', 'wisdom', 'liberation', 'moksha', 'death', 'birth', 'detachment', 'meditation', 'bhakti']):
            questions.append((q, a))
    print(f'Philosophy-related: {len(questions)} out of 500')
    for q, a in questions[:5]:
        print(f'  Q: {q[:100]}')
        print(f'  A: {a[:100]}')
        print()

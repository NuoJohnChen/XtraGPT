import json

before_results = []
after_results = []

with open("./decisions_before_xtragpt.json", 'r', encoding='utf-8') as f:
    before_results = json.load(f)

with open("./decisions_after_xtragpt.json", 'r', encoding='utf-8') as f:
    after_results = json.load(f)


total = 0
rating_improvement = 0.0
num_accept_before = 0
num_accept_after = 0
for r1, r2 in zip(before_results, after_results):
    if r1['overall'] != 'FAILED' and r2['overall'] != 'FAILED':
        total += 1
        rating_improvement += r2['overall'] - r1['overall']
        if r1['ai_scientist_decision'] == 'Accept':
            num_accept_before += 1
        if r2['ai_scientist_decision'] == 'Accept':
            num_accept_after += 1

print(f"Improvement to Rating: {rating_improvement/total:.2}")
print(f"Before Acceptance Rate: {num_accept_before/total:.2%}")
print(f"After Acceptance Rate: {num_accept_after/total:.2%}")
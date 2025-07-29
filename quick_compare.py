import json

# Load reference and current output for comparison
with open('reference.json', 'r') as f:
    reference = json.load(f)
with open('output/Devops.json', 'r') as f:
    current = json.load(f)

print('Expected: %d headings' % len(reference['outline']))
print('Current:  %d headings' % len(current['outline']))

ref_texts = set(item['text'] for item in reference['outline'])
curr_texts = set(item['text'] for item in current['outline'])
missing = ref_texts - curr_texts
incorrect = curr_texts - ref_texts

print('Missing:  %d headings' % len(missing))
print('Incorrect: %d headings' % len(incorrect))
print('Accuracy: %.1f%%' % (100 * len(ref_texts & curr_texts) / len(ref_texts)))

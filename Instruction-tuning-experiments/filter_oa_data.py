import json

oasst_filenames = ['./data/oasst-fi/oasst1-fi-train.jsonl', './data/oasst-fi/oasst1-fi-valid.jsonl', './data/oasst-fi/oasst1-fi-eval.jsonl']
for oasst_file in oasst_filenames:
    print("Filtering", oasst_file)
    data = [json.loads(line) for line in open(oasst_file)]
    # include only message not marked as deleted
    filter_deleted = [entry for entry in data if entry['deleted'] == False]
    # include only messages with spam < 0.5 and not_appropriate < 0.5 and toxicity < 0.5
    filter_spam_na = []
    for entry in filter_deleted:
        if 'labels' in entry and entry['labels'] is not None:
            if 'name' in entry['labels']:
                if 'spam' in entry['labels']['name']:
                    spam_index = entry['labels']['name'].index('spam')
                    spam_score = entry['labels']['value'][spam_index]
                else:
                    spam_score = 0.0
                if 'not_appropriate' in entry['labels']['name']:
                    na_index = entry['labels']['name'].index('not_appropriate')
                    na_score = entry['labels']['value'][na_index]
                else:
                    na_score = 0.0
                if 'toxicity' in entry['labels']['name']:
                    tox_index = entry['labels']['name'].index('toxicity')
                    tox_score = entry['labels']['value'][tox_index]
                else:
                    tox_score = 0.0
                if spam_score < 0.5 and na_score < 0.5 and tox_score < 0.5:
                    filter_spam_na.append(entry)
    print("Original data:", len(data))
    print("Filtered data:", len(filter_spam_na))
    filtered_filename = oasst_file.replace(".jsonl", "-filter.jsonl")
    with open(filtered_filename, "w") as f:
        for entry in filter_spam_na:
            json.dump(entry, f)
            f.write("\n")
        f.close()
        print("Saved filtered data to:", filtered_filename)   
             

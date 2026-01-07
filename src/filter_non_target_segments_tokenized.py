import os
import json
from tqdm import tqdm
from smart_open import open
from optparse import OptionParser


def find_first_last_occurrences(lst, items):
    first = len(lst)
    last = 0
    for item in items:
        first_index = lst.index(item)
        last_index = len(lst) - 1 - lst[::-1].index(item)
        if first_index < first:
            first = first_index
        if last_index > last:
            last = last_index
    return first, last


def _find_truncation_indices(segment, target_words_found, max_len, length, trunc_segm_count):
    first, last = find_first_last_occurrences(
        segment, target_words_found
    )
    target_span_length = last - first
    if target_span_length >= max_len:
        # we have to sacrifice something
        truncation_start = first
        truncation_end = first + max_len
    else:
        trunc_budget = max_len - target_span_length
        if first < trunc_budget:
            truncation_start = 0
            truncation_end = max_len
        else:
            if (length - last) < trunc_budget:
                truncation_start = length - max_len
                truncation_end = length
            else:
                context = int(trunc_budget / 2)
                truncation_start = first - context
                truncation_end = last + context
    trunc_segm_count += 1
    return trunc_segm_count, truncation_start, truncation_end


def _filter_t5(
        segments, token_type_ids, attention_mask, lengths, target_words, doc_id, max_len,
        total_segment_count, segment_nr, filtered_segment_count, trunc_segm_count,
    ):
    for segment, tti, am, length in zip(
            segments, token_type_ids, attention_mask, lengths
        ):
        if length > 0:
            segment_id = f"{doc_id}__{segment_nr}"
            segment_tokens = set(segment)
            target_words_found = segment_tokens.intersection(target_words)
            if target_words_found:
                if length > max_len:
                    trunc_segm_count, truncation_start, truncation_end = _find_truncation_indices(
                        segment, target_words_found, max_len, length, trunc_segm_count,
                    )
                    segment = segment[truncation_start:truncation_end]
                    tti = tti[truncation_start:truncation_end]
                    am = am[truncation_start:truncation_end]
                assert len(segment) <= max_len
                print(
                    json.dumps(
                        {
                            "s_id": segment_id,
                            "input_ids": segment,
                            "token_type_ids": tti,
                            "attention_mask": am,
                            "length": length,
                        }
                    )
                )
            else:
                filtered_segment_count += 1
        segment_nr += 1
        total_segment_count += 1
        return total_segment_count, segment_nr, filtered_segment_count, trunc_segm_count
    

def _filter_xlmr(
        segments, attention_mask, lengths, target_words, doc_id, max_len,
        total_segment_count, segment_nr, filtered_segment_count, trunc_segm_count,
    ):
    for segment, am, length in zip(
            segments, attention_mask, lengths
        ):
        if length > 0:
            segment_id = f"{doc_id}__{segment_nr}"
            segment_tokens = set(segment)
            target_words_found = segment_tokens.intersection(target_words)
            if target_words_found:
                if length > max_len:
                    trunc_segm_count, truncation_start, truncation_end = _find_truncation_indices(
                        segment, target_words_found, max_len, length, trunc_segm_count,
                    )
                    segment = segment[truncation_start:truncation_end]
                    am = am[truncation_start:truncation_end]
                assert len(segment) <= max_len
                print(
                    json.dumps(
                        {
                            "s_id": segment_id,
                            "input_ids": segment,
                            "attention_mask": am,
                            "length": length,
                        }
                    )
                )
            else:
                filtered_segment_count += 1
        segment_nr += 1
        total_segment_count += 1
        return total_segment_count, segment_nr, filtered_segment_count, trunc_segm_count


def main():
    """Script to process the tokenized data."""
    usage = "usage: %prog [options] arg"
    parser = OptionParser(usage)
    parser.add_option(
        "--language",
        type=str,
        default="eng_Latn",
        help="Dataset language code: default=%default",
    )
    parser.add_option(
        "--period", type=str, default="2011_2015", help="Time period: default=%default"
    )
    parser.add_option(
        "--data-dir",
        type=str,
        default="../data",
        help="Default directory where tokenized datasets by language are stored: default=%default",
    )
    parser.add_option(
        "--max_len",
        type=int,
        default=512,
        help="Maximum segment length to keep (in tokens): default=%default",
    )

    (options, args) = parser.parse_args()

    language = options.language
    period = options.period
    max_len = options.max_len
    assert period in ["2011_2015", "2020_2021", "2024_2025"]

    data_dir = options.data_dir

    # Load target words
    with open(os.path.join("languages", language, "target_words.json")) as f:
        target_words = json.load(f)

    target_word_ids = set([target_words[word] for word in target_words])

    with open(os.path.join(data_dir, language, f"{period}_token_count.json")) as f:
        counts = json.load(f)
        counts = {int(k): v for k, v in counts.items()}
    target_words = {key: counts[key] for key in counts if key in target_word_ids}

    infile = os.path.join(data_dir, language, f"{period}_tokens.jsonl.gz")

    total_doc_count, total_segment_count, filtered_segment_count, trunc_segm_count = (
        0,
        0,
        0,
        0,
    )

    with open(infile) as in_f:
        for line in tqdm(in_f, total=10**6, mininterval=60):
            total_doc_count += 1
            doc_dct = json.loads(line)
            segments = doc_dct["input_ids"]
            if "token_type_ids" in doc_dct:
                token_type_ids = doc_dct["token_type_ids"]
            attention_mask = doc_dct["attention_mask"]
            lengths = doc_dct["length"]
            doc_id = doc_dct["id"]

            # https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-equal
            doc_value_lengths = [len(doc_value) for doc_value in doc_dct.items()]
            assert (not doc_value_lengths or doc_value_lengths.count(doc_value_lengths[0]) == len(doc_value_lengths))

            segment_nr = 0
            if "token_type_ids" in doc_dct:
                total_segment_count, segment_nr, filtered_segment_count, trunc_segm_count = _filter_t5(
                    segments, token_type_ids, attention_mask, lengths, target_words, doc_id, max_len,
                    total_segment_count, segment_nr, filtered_segment_count, trunc_segm_count,
                )
            else:
                total_segment_count, segment_nr, filtered_segment_count, trunc_segm_count = _filter_xlmr(
                    segments, attention_mask, lengths, target_words, doc_id, max_len,
                    total_segment_count, segment_nr, filtered_segment_count, trunc_segm_count,
                )


    with open(
        os.path.join(
            data_dir,
            language,
            f"{period}__segment_filtering_info.json",
        ),
        "w",
    ) as f:
        json.dump(
            {
                "total_doc_count": total_doc_count,
                "total_segment_count": total_segment_count,
                "truncated_segment_count": trunc_segm_count,
                "filtered_segment_count": filtered_segment_count,
            },
            f,
        )


if __name__ == "__main__":
    main()

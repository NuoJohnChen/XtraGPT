import openai
from ai_scientist.perform_review import load_paper, perform_review
import json

PAPERS = ['3qo1pJHabg', 'rHzapPnCgT', 'fDZumshwym', 'etm456yoiq', 'M3QXCOTTk4', 'wOMy6J8epf', 'mhtD74Jgyw', 'xhEN0kJh4q', 'oDYXpvnv5f', 'f6KkyweyYh', 'jVuknNhGmV', 'CY9f6G89Rv', 'HnVtsfyvap', 'b3Cu426njo', 'Kn7tWhuetn', 'Ffjc8ApSbt', 'XkLMGx60aZ', 'yC2waD70Vj', 'EArTDUmILF', 'Gk75gOjtQh', 'uizIvVBY8P', 'ghyeMoj1gK', '98g9NdJPxm', '5T46w5X3Go', 'RUgBoMu0ad', 'N0nTk5BSvO']
# PAPERS = ['aTFPO9FHL3', '5T46w5X3Go', 'RUgBoMu0ad', 'N0nTk5BSvO']

client = openai.OpenAI(api_key="...")
model = "o1-2024-12-17"

BASE_PATH = "/shared/ssd/ConferenceQA_test/"

def parse_decision(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    raw_decision = data['content']['venueid']['value']
    if raw_decision == "ICLR.cc/2024/Conference":
        return raw_decision, "Accept"
    else:
        return raw_decision, "Reject"

if __name__ == "__main__":
    # this part is for getting the revised xtragpt version
    markdown_cache = {}
    with open('/shared/ssd/ConferenceQA_rating/rating_test_prediction_avg_with_decision_after_xtragpt_revision.jsonl', 'r', encoding='utf-8') as f:
        for row in f:
            data = json.loads(row)
            paper_id = data['article']
            if paper_id in PAPERS:
                markdown_cache[paper_id] = data['paper_content']

    res = []
    for idx, paper in enumerate(PAPERS):
        # Load paper from PDF file (raw text)
        paper_path = BASE_PATH + f"{paper}/paper.pdf"
        notes_path = BASE_PATH + f"{paper}/note.json"
        try:
            # this is for before xtragpt
            # paper_txt = load_paper(paper_path)
            # this is for after xtragpt
            paper_txt = markdown_cache[paper]

            # Get the review dictionary
            review = perform_review(
                paper_txt,
                model,
                client,
                num_reflections=1,
                num_fs_examples=1,
                num_reviews_ensemble=1,
                temperature=0.1,
            )
            raw_dec, dec = parse_decision(notes_path)
            details = {
                "paper_id": paper,
                "overall": review["Overall"],
                "ai_scientist_decision": review["Decision"],
                "raw_decision": raw_dec,
                "decision": dec,
                # "weaknesses": review["Weaknesses"],
                "soundness": review["Soundness"],
                "presentation": review["Presentation"],
                "contribution": review["Contribution"],
                "confidence": review["Confidence"]
            }
        except Exception as e:
            print(f"Error: {e}")
            # details = {
            #     "paper_id": paper,
            #     "overall": "FAILED",
            #     "ai_scientist_decision": "FAILED",
            #     "decision": "FAILED"
            #     # "weaknesses": review["Weaknesses"],
            # }
            

        res.append(details)

        print(f"Completed {idx+1} / {len(PAPERS)}.. Writing..")
        with open("./after_revision_scores.json", 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=2, ensure_ascii=False)


# conda activate /miniconda3/envs/openrlhf
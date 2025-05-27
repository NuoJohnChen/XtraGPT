import json

def evaluate_decisions(input_path: str):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    TP = TN = FP = FN = 0
    ignored = 0
    for entry in data:
        pred = entry.get("ai_scientist_decision", "").strip().lower()
        label = entry.get("decision", "").strip().lower()

        if pred == "accept" and label == "accept":
            TP += 1
        elif pred == "reject" and label == "reject":
            TN += 1
        elif pred == "accept" and label == "reject":
            FP += 1
        elif pred == "reject" and label == "accept":
            FN += 1
        else:
            ignored += 1

    print("Confusion Matrix (Case-Insensitive)")
    print("-----------------------------------")
    print(f"Num failed to be evaluated: {ignored}")
    print(f"True Positives (Accepted ↔ Accepted): {TP}")
    print(f"True Negatives (Rejected ↔ Rejected): {TN}")
    print(f"False Positives (Pred: Accepted, True: Rejected): {FP}")
    print(f"False Negatives (Pred: Rejected, True: Accepted): {FN}")
    print("-----------------------------------")
    print(f"Total Samples: {TP + TN + FP + FN}")
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Accuracy : {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall   : {recall:.2%}")
    print(f"F1 Score : {f1:.2%}")

if __name__ == "__main__":
    evaluate_decisions("./decisions_after_xtragpt.json")

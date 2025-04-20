
import re
from collections import Counter

data = {
    "student_scores_out_of_hundred_as_list": ":  \n    - Student 1: 100\n    - Student 2: 80\n    - Student 3: 60\n    - Student 4: 40\n    - Student 5: 70"
}

raw_scores = data["student_scores_out_of_hundred_as_list"]

scores = re.findall(r"-\s*Student\s*\d+:\s*(\d+)", raw_scores)

# Count frequency of each score
score_counts = Counter(map(int, scores))

# Convert to desired format
result = [{"score": score, "count": count} for score, count in score_counts.items()]

data["student_scores_out_of_hundred_as_list"] = result

print(result)
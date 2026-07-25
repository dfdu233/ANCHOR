import pandas as pd
import random
import string
import numpy as np
from utils_conversational import ORIGINAL_DATA_FOLDER, PROCESSED_DATA_FOLDER, SPECIAL_SEP_TOKEN
from tqdm import tqdm
from LLMs import LLM_Consolidate, LLM_Respond, LLM_Score

CHECK_SCORE = False

llm_consolidate = LLM_Consolidate()
llm_respond = LLM_Respond()
llm_score = LLM_Score()

# fix seeding
seed_val = 321
random.seed(seed_val)
np.random.seed(seed_val)
print("Fixed random seed on value: ", seed_val)

# define the unwanted follow up questions (case-insensitive)
UNWANTED_EXACT_IN_FOLLOWUP_QUESTIONS = \
    ["i dont know", "i dont know yet", "no", "i dont understand your question", "i am not sure", "im not sure", "i doubt it", "na", "did you find them"]
UNWANTED_PATTERNS_IN_FOLLOWUP_QUESTIONS = \
    ["picture", "pictures", "photo", "photos", "image", "images", "map"]

# filtering function for removing low-quality samples
def normalize(text):
    if not isinstance(text, str):
        return ""
    # lowercase, strip spaces, remove punctuation
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    # collapse multiple spaces (e.g., "i dont   know" → "i dont know")
    text = " ".join(text.split())
    return text

def drop_low_quality_rows(df):
    df = df.dropna()
    # build a boolean mask for rows to drop
    mask_exact = df.apply(
                    lambda col: col.map(normalize).isin(UNWANTED_EXACT_IN_FOLLOWUP_QUESTIONS))
    mask_pattern = df.apply(
                    lambda col: col.map(normalize).str.contains('|'.join(UNWANTED_PATTERNS_IN_FOLLOWUP_QUESTIONS), case=False, na=False))
    mask = mask_exact.any(axis=1) | mask_pattern.any(axis=1)

    df = df[~mask]
    df.reset_index(drop=True, inplace=True)

    print(f"Removed {mask.sum()} rows; remaining: {len(df)}")

    return df

# response generation function
def generate_y_set(x_set, llm_respond, label):
    y_set = []
    for x in tqdm(x_set, desc=f"Responding for {label} questions..."):
        y = []
        for x_component in x:
            y.append(llm_respond.respond(x_component))
        y_set.append(y)
    return y_set

# score generation function
def compute_score_set(x_set, y_set, llm_score, label):
    score_set, rationale_set = [], []
    for x, y in tqdm(zip(x_set, y_set),
                     total=len(x_set),
                     desc=f"Scoring for {label} question..."):
        assert len(x) == len(y)
        score, rationale = [], []
        for x_component, y_component in zip(x, y):
            score_component, rationale_component = llm_score.score(x_component, y_component, verbose=True)
            score.append(score_component)
            rationale.append(rationale_component)
        score_set.append(score)
        rationale_set.append(rationale)
    
    print(f"Avg score on {label} questions: ", np.mean(score_set, axis=0))
    return score_set, rationale_set

if __name__ == "__main__":
    """
    Load the raw data
    """
    # do not use the clarification_need label since even with labelled 1, the question is still with low quality
    # only use the multi-turn data and sample 0,1,2 turn questions
    df_orig = pd.read_csv(ORIGINAL_DATA_FOLDER / "multi_turn_human_generated_data.tsv", sep='\t')
    df_orig = df_orig[["initial_request", 
                    "question1", 
                    "answer1", 
                    "question2", 
                    "answer2", 
                    "question3", 
                    "answer3"]]

    # shuffle the rows 
    df_orig = df_orig.sample(frac=1,random_state=seed_val).reset_index(drop=True)

    """
    Remove low-quality quetion samples
    """
    print("Filtering low quality rows from the original complex questions...")
    df_orig = drop_low_quality_rows(df_orig)

    """
    Consolidate questions to get detailed stand-alone questions
    """
    (
        x_base, 
        x_follow1, 
        x_ext1,
        x_follow2,
        x_ext2,
        x_follow3,
        x_ext3
    ) = (
            df_orig["initial_request"], 
            df_orig["question1"], 
            df_orig["answer1"],
            df_orig["question2"],
            df_orig["answer2"],     
            df_orig["question3"],
            df_orig["answer3"]
        )
    # for each datapoint, has x as a list of four elements, with the later elements being the concatenation 
    # of all the questions and clarifications up to that time step
    x_set = []
    for (base, 
        follow1, 
        ext1, 
        follow2, 
        ext2, 
        follow3, 
        ext3) in tqdm(zip(x_base,
                        x_follow1, 
                        x_ext1,
                        x_follow2, 
                        x_ext2,
                        x_follow3, 
                        x_ext3), 
                        total=len(x_base),
                        desc="Consolidating for complex questions..."):
        consolidated1 = llm_consolidate.consolidate_question(
                            SPECIAL_SEP_TOKEN.join([base, follow1, ext1]))
        consolidated2 = llm_consolidate.consolidate_question(
                            SPECIAL_SEP_TOKEN.join([base, follow1, ext1, follow2, ext2]))
        consolidated3 = llm_consolidate.consolidate_question(
                            SPECIAL_SEP_TOKEN.join([base, follow1, ext1, follow2, ext2, follow3, ext3])) 
        x = [base, 
            consolidated1,
            consolidated2,    
            consolidated3]
        x_set.append(x)

    """
    Split question samples into 0-turn, 1-turn, 2-turn, and 3-turn evenly,
    using the consolidated questions
    """
    chunk_size = len(x_set) // 4

    (
        x_set_0_turn, 
        x_set_1_turn,
        x_set_2_turn,
        x_set_3_turn
    ) = (
            x_set[:chunk_size], 
            x_set[chunk_size:2*chunk_size],
            x_set[2*chunk_size:3*chunk_size],
            x_set[3*chunk_size:]
        )
    # the last consolidated question should require no further clarification
    x_set_0_turn = [[x[-1]] for x in x_set_0_turn]
    x_set_1_turn = [[x[0], x[-1]] for x in x_set_1_turn]
    x_set_2_turn = [[x[0], x[1], x[-1]] for x in x_set_2_turn]

    print(f"Generated {len(x_set_0_turn)} 0-turn questions; " +
            f"{len(x_set_1_turn)} 1-turn questions; " +
                f"{len(x_set_2_turn)} 2-turn questions; " +
                    f"{len(x_set_3_turn)} 3-turn questions.")
    
    """
    Zero-Turn questions
    """
    y_set_0_turn = generate_y_set(x_set_0_turn, 
                                  llm_respond, 
                                  "zero-turn")
    if CHECK_SCORE:
        score_set_0_turn, rationale_set_0_turn = compute_score_set(x_set_0_turn, 
                                                                y_set_0_turn, 
                                                                llm_score, 
                                                                "zero-turn")
    else:
        score_set_0_turn, rationale_set_0_turn = None, None

    admissibility_set_0_turn = [[1] for i in range(len(x_set_0_turn))]

    """
    One-Turn questions
    """
    y_set_1_turn = generate_y_set(x_set_1_turn, 
                                  llm_respond, 
                                  "one-turn")
    if CHECK_SCORE:
        score_set_1_turn, rationale_set_1_turn = compute_score_set(x_set_1_turn, 
                                                                y_set_1_turn, 
                                                                llm_score, 
                                                                "one-turn")
    else:
        score_set_1_turn, rationale_set_1_turn = None, None

    admissibility_set_1_turn = [[0, 1] for i in range(len(x_set_1_turn))]

    """
    Two-Turn questions
    """
    y_set_2_turn = generate_y_set(x_set_2_turn, 
                                  llm_respond, 
                                  "two-turn")
    if CHECK_SCORE:
        score_set_2_turn, rationale_set_2_turn = compute_score_set(x_set_2_turn, 
                                                                y_set_2_turn, 
                                                                llm_score, 
                                                                "two-turn")
    else:
        score_set_2_turn, rationale_set_2_turn = None, None

    admissibility_set_2_turn = [[0, 0, 1] for i in range(len(x_set_2_turn))]

    """
    Three-Turn questions
    """
    y_set_3_turn = generate_y_set(x_set_3_turn, 
                                 llm_respond, 
                                 "three-turn")
    if CHECK_SCORE:
        score_set_3_turn, rationale_set_3_turn = compute_score_set(x_set_3_turn, 
                                                                y_set_3_turn, 
                                                                llm_score, 
                                                                "three-turn")
    else:
        score_set_3_turn, rationale_set_3_turn = None, None

    admissibility_set_3_turn = [[0, 0, 0, 1] for i in range(len(x_set_3_turn))]

    """
    Calibrate-Test Set Split
    """
    x_cali, y_cali, a_cali = [], [], []
    x_test, y_test, a_test = [], [], []

    for x, y, a in zip([x_set_0_turn, x_set_1_turn, x_set_2_turn, x_set_3_turn],
                          [y_set_0_turn, y_set_1_turn, y_set_2_turn, y_set_3_turn], 
                          [admissibility_set_0_turn, admissibility_set_1_turn, admissibility_set_2_turn, admissibility_set_3_turn]):
        split_point = int(len(x)/2)
        x_cali += x[:split_point]
        x_test += x[split_point:]
        y_cali += y[:split_point]
        y_test += y[split_point:]
        a_cali += a[:split_point]
        a_test += a[split_point:]      

    print(f"Calibration set size: {len(x_cali)}; Test set size: {len(x_test)}") 

    """
    Save the processed data sets
    """
    import pickle

    with open(PROCESSED_DATA_FOLDER / "ClariQ_cali.pkl", "wb") as f:
        pickle.dump((x_cali, y_cali, a_cali), f)
    with open(PROCESSED_DATA_FOLDER / "ClariQ_test.pkl", "wb") as f:
        pickle.dump((x_test, y_test, a_test), f)

    print("[Conversational AI] Dataset preprocessing completed!")
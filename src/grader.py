import requests
from concurrent.futures import ThreadPoolExecutor
import time

STUDENTS = {
    "team00": "http://localhost:8000",
}

PROBLEMS = {
    "1": {"secret_word": "flame", "candidate_words": ["crane", "flame", "slate"]},
    "2": {"secret_word": "altar", "candidate_words": ["apple", "altar", "adapt"]},
}


def compute_feedback(secret, guess):
    feedback = [0] * 5  # Default: 0
    secret_chars = list(secret)
    guess_chars = list(guess)

    # First pass: assign 2 (correct position)
    for i in range(5):
        if guess_chars[i] == secret_chars[i]:
            feedback[i] = 2
            secret_chars[i] = None
            guess_chars[i] = None

    # Count remaining letters in secret
    remaining = {}
    for ch in secret_chars:
        if ch:
            remaining[ch] = remaining.get(ch, 0) + 1

    # Second pass: assign 1 (misplaced)
    for i in range(5):
        if guess_chars[i] and remaining.get(guess_chars[i], 0) > 0:
            feedback[i] = 1
            remaining[guess_chars[i]] -= 1

    return feedback


def verbalize_feedback(secret, guess, feedback):
    parts = []
    for i, f in enumerate(feedback):
        ch = guess[i]
        if f == 2:
            parts.append(f"'{ch}' is in the correct position.")
        elif f == 1:
            parts.append(f"'{ch}' is in the word but in the wrong position.")
        else:
            parts.append(f"'{ch}' is not in the word.")
    return " ".join(parts)


def run_for_team(team_name, base_url):
    print(f"[{team_name}] Starting evaluation")

    for problem_id, problem_data in PROBLEMS.items():
        candidate_words = problem_data["candidate_words"]
        secret = problem_data["secret_word"]

        try:
            requests.post(
                f"{base_url}/start_problem",
                json={"problem_id": problem_id, "candidate_words": candidate_words},
                timeout=10,
            )
        except Exception as e:
            print(f"[{team_name}] Failed to send start_problem: {e}")
            continue

        guess_count = 0
        feedback = None

        while True:
            try:
                payload = {
                    "problem_id": problem_id,
                    "verbal_feedback": feedback,
                    "turn": guess_count + 1,
                }
                r = requests.post(f"{base_url}/guess", json=payload, timeout=10)
                r.raise_for_status()
                guess = r.json()["guess"]
            except Exception as e:
                print(f"[{team_name}] Error: {e}")
                break

            guess_count += 1

            if guess == secret:
                print(f"[{team_name}] Solved {problem_id} in {guess_count} guesses.")
                break

            raw_feedback = compute_feedback(secret, guess)
            feedback = verbalize_feedback(secret, guess, raw_feedback)

    print(f"[{team_name}] Finished.")


def main():
    with ThreadPoolExecutor(max_workers=4) as executor:
        for team, url in STUDENTS.items():
            executor.submit(run_for_team, team, url)


if __name__ == "__main__":
    main()

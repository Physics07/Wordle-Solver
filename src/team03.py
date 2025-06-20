import atexit
import datetime
import json
import os

from http.server import BaseHTTPRequestHandler, HTTPServer

from dotenv import load_dotenv
from snowflake.snowpark import Session
from snowflake.cortex import complete

from userlib.guess import Guesser

load_dotenv()


class Solver:
    def __init__(self):
        self.session = self._init_snowflake()
        self.model = "claude-3-5-sonnet"
        self.problems = {}
        self.guessers = {}
        self.previous_guess = {}
        self.snowflake_calls = 0
        self.log_file = open("run.log", "a")
        atexit.register(self.cleanup)

    def _init_snowflake(self):
        connection_params = {
            "account": os.environ.get("SNOWFLAKE_ACCOUNT"),
            "user": os.environ.get("SNOWFLAKE_USER"),
            "password": os.environ.get("SNOWFLAKE_USER_PASSWORD"),
            "role": os.environ.get("SNOWFLAKE_ROLE"),
            "database": os.environ.get("SNOWFLAKE_DATABASE", ""),
            "schema": os.environ.get("SNOWFLAKE_SCHEMA", ""),
            "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE", ""),
        }
        return Session.builder.configs(connection_params).create()

    def cleanup(self):
        try:
            self.log_file.close()
        except:
            pass
        try:
            self.session.close()
        except:
            pass

    def start_problem(self, problem_id, candidate_words):
        self.problems[problem_id] = {
            "candidate_words": candidate_words,
            "feedback_history": [],
        }
        self.guessers[problem_id] = Guesser("./data/", candidate_words)
        self.previous_guess[problem_id] = None
        self._log(f"\n=== Starting Problem {problem_id} ===")
        self._log(f"Candidate words: {', '.join(candidate_words)}")

    def add_feedback(self, problem_id, verbal_feedback):
        if verbal_feedback:
            self.problems[problem_id]["feedback_history"].append(verbal_feedback)

    def choose_next_guess(self, problem_id, turn):
        candidates = self.problems[problem_id]["candidate_words"]
        history = self.problems[problem_id]["feedback_history"]

        if not history:
            guess = self.guessers[problem_id].find_guess()
            self.previous_guess[problem_id] = guess
            self._log(f"Turn {turn}: Received feedback: None (first turn)")
            self._log(f"Turn {turn}: Guess: {guess}")
            return guess
        
        prompt = (
            "I'm playing Wordle and guessed a word, and got a feedback for my guess. Based on the following feedback, return the result string generated by this feedback.\n"
            f"my guess: the first letter is {self.previous_guess[problem_id][0]},\n"
            f"the second letter is '{self.previous_guess[problem_id][1]}',\n"
            f"the third letter is '{self.previous_guess[problem_id][2]}',\n"
            f"the fourth letter is '{self.previous_guess[problem_id][3]}',\n"
            f"the fifth letter is '{self.previous_guess[problem_id][4]}',\n\n"
            f"feedback: {history[-1]}\n\n"
            "The result string is computed as follows:\n"
            "1. If the two words have the same letter at the i-th (i is between 1 and 5) position, the corresponding letter in the i-th position of the result string is 'B'.\n"
            "2. If the i-th letter of the guess is not in the answer, the corresponding letter in the i-th position of the result string is 'G'.\n"
            "3. If the i-th letter of the guess is in the answer but not in the right position, the i-th position of the result string is either 'Y' or 'G', such that:\n"
            " -The total number of B's and Y's assigned to a character cannot be more than its count in the answer. and\n"
            " -All G's, if any, must appear after all Y's for that character.\n"
            "Make sure to put all the Y's, G's, and B's in their corresponding spots.\n"
            # "For example, if the answer is ' t w e e t ', and the query is ' m e l e e ', then the result string is ' G Y G B G '. You should return only ' G Y G B G ' for this example.\n"
            "Return ONLY the result string and do not return any other text (your response needs to consist of 5 uppercase letters, with spaces inbetween each one)."
        )

        response = "".join(
            complete(
                model=self.model,
                prompt=[{"role": "user", "content": prompt}],
                options={"max_tokens": 10, "temperature": 0.0},
                session=self.session,
            )
            .strip()
            .lower()
            .split()
        )

        self.guessers[problem_id].update(Guesser.convert_feedback_to_int(response))

        self.snowflake_calls += 1

        guess = self.guessers[problem_id].find_guess()
        
        last_feedback = history[-1] if history else "None"
        self.previous_guess[problem_id] = guess
        self._log(f"Turn {turn}: Received feedback: {last_feedback}")
        self._log(f"Turn {turn}: Guess: {guess}")
        
        return guess

    def _log(self, msg):
        ts = datetime.datetime.now().isoformat()
        self.log_file.write(f"[{ts}] {msg}\n")
        self.log_file.flush()


solver = Solver()


class StudentHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length"))
        data = json.loads(self.rfile.read(length))

        if self.path == "/start_problem":
            problem_id = data["problem_id"]
            candidate_words = data["candidate_words"]
            solver.start_problem(problem_id, candidate_words)
            self.send_response(200)
            self.end_headers()
            return

        if self.path == "/guess":
            problem_id = data["problem_id"]
            verbal_feedback = data.get("verbal_feedback")
            turn = data["turn"]
            solver.add_feedback(problem_id, verbal_feedback)
            guess = solver.choose_next_guess(problem_id, turn)

            response = {"guess": guess}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            return

        self.send_response(404)
        self.end_headers()


def run():
    port = int(os.environ.get("PORT", 8000))
    server = HTTPServer(("0.0.0.0", port), StudentHandler)
    print(f"Student server running on port {port}...")
    server.serve_forever()


if __name__ == "__main__":
    run()
